"""Evaluate paired WaveNet and Aggregator predictions with explicit weights."""

import csv
from pathlib import Path

import numpy as np
import torch

from .data import DatasetIndex, sha256
from .losses import ARCSEC_PER_MICRON
from .models import load_model
from .training import CONTRACT, predict, wave_predictions


METHOD_NAMES = {
    "wavenet_single_stamp": "WaveNet single stamp",
    "wavenet_mean": "WaveNet CCD mean",
    "wavenet_median": "WaveNet CCD median",
    "aggregator": "Aggregator",
}


def weighted_metrics(prediction, truth, weights):
    """Pool squared coefficient errors before taking the square root."""
    prediction, truth, weights = (np.asarray(value, dtype=float) for value in (prediction, truth, weights))
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 25:
        raise ValueError("Expected matching (N,25) predictions and truth")
    if not len(truth) or weights.shape != (len(truth),) or not np.all(weights > 0):
        raise ValueError("Expected one positive weight per prediction")
    if not all(np.isfinite(value).all() for value in (prediction, truth, weights)):
        raise ValueError("Non-finite evaluation input")
    weights = weights / weights.sum()
    error = prediction - truth
    mse = np.sum(weights[:, None] * error**2, axis=0)
    p = prediction - np.sum(weights[:, None] * prediction, axis=0)
    t = truth - np.sum(weights[:, None] * truth, axis=0)
    covariance = np.sum(weights[:, None] * p * t, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * p**2, axis=0) * np.sum(weights[:, None] * t**2, axis=0))
    corr = [float(np.clip(c / s, -1, 1)) if s > 0 else None for c, s in zip(covariance, scale)]
    return dict(coefficient_rmse_um=float(np.sqrt(mse.mean())),
                mrsse_arcsec=float(np.sum(weights * np.linalg.norm(error * ARCSEC_PER_MICRON, axis=1))),
                per_mode_rmse_um=np.sqrt(mse).tolist(),
                per_mode_bias_um=np.sum(weights[:, None] * error, axis=0).tolist(),
                per_mode_corr=corr)


def compare_predictions(rows, predictions, groups, aggregated):
    """Use the same paired CCDs and stamps for every method and weighting."""
    by_id = {int(sample): i for i, sample in enumerate(predictions["sample_id"])}
    by_group = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(groups["state_index"], groups["detector_id"]))}
    selected = [row for row in rows if (int(row["state_index"]), int(row["detector_id"])) in by_group]
    group_index = np.array([by_group[int(row["state_index"]), int(row["detector_id"])] for row in selected])
    counts = np.bincount(group_index, minlength=len(by_group))
    if not np.array_equal(counts, groups["length"]):
        raise ValueError("Stamp counts do not match Aggregator groups")
    stamp_prediction = predictions["prediction_ccs_um"][[by_id[int(row["sample_id"])] for row in selected]]
    stamp_truth = groups["truth"][group_index]
    methods = {"wavenet_single_stamp": dict(
        ccd_equal=weighted_metrics(stamp_prediction, stamp_truth, 1.0 / counts[group_index]),
        stamp_equal=weighted_metrics(stamp_prediction, stamp_truth, np.ones(len(selected))))}
    for name, prediction in (("wavenet_mean", groups["mean"]),
                             ("wavenet_median", groups["median"]), ("aggregator", aggregated)):
        methods[name] = dict(ccd_equal=weighted_metrics(prediction, groups["truth"], np.ones(len(counts))),
                             stamp_equal=weighted_metrics(prediction, groups["truth"], counts))
    return dict(split="test", population="state/CCD groups with both intra and extra",
                ccd_groups=len(counts), stamps=int(counts.sum()), methods=methods, **CONTRACT)


def evaluate_pair(configs, variant="supervised", wave_checkpoint=None, aggregator_checkpoint=None,
                  device=None, zoom_quantile=0.002, index=None):
    """Read frozen-WaveNet caches and update both reports on matched test groups."""
    from .reporting import write_report

    config = configs["aggregator"]
    wave_path = Path(wave_checkpoint or config["wave_checkpoint"])
    agg_path = Path(aggregator_checkpoint or Path(config["output_dir"]) / "model.pt")
    torch.set_num_threads(config.get("torch_threads", 8))
    device = torch.device(device or config["device"])
    model, agg = load_model(agg_path, "aggregator", device)
    wave = torch.load(wave_path, map_location="cpu", weights_only=False, mmap=True)
    wave_hash = sha256(wave_path)
    if wave_hash != agg["wave_checkpoint_sha256"]:
        raise ValueError("Aggregator requires its matching frozen WaveNet")
    index = index or DatasetIndex(config["dataset_root"])
    if wave["dataset_identity"] != index.identity or agg["dataset_identity"] != index.identity:
        raise ValueError("Evaluate on the dataset recorded in both checkpoints")
    if config["wave_precision"] != agg["config"]["wave_precision"]:
        raise ValueError("Frozen WaveNet prediction precision changed")
    cache_dir = (Path(config["output_dir"]) / "cache").resolve()
    predictions = wave_predictions(index, "test", wave_path, cache_dir,
                                   device, config["wave_precision"], config["wave_batch_size"])
    dataset = index.aggregate("test", predictions, require_both_sides=True)
    aggregated = predict(model, dataset, "aggregator", device, config["precision"], config["batch_size"])
    groups = dataset.arrays
    result = compare_predictions(index.splits["test"], predictions, groups, aggregated)
    result.update(variant=variant, dataset_identity=index.identity,
                  models=dict(wavenet=dict(sha256=wave_hash, epoch=wave["epoch"], precision=config["wave_precision"]),
                              aggregator=dict(sha256=sha256(agg_path), epoch=agg["epoch"], precision=config["precision"])))
    plot_values = np.concatenate([groups["truth"], groups["mean"], aggregated], axis=0)
    bounds = np.quantile(plot_values, [zoom_quantile, 1 - zoom_quantile], axis=0)
    margin = np.maximum((bounds[1] - bounds[0]) * 0.07, 1e-4)
    bounds = np.stack([bounds[0] - margin, bounds[1] + margin], axis=1)
    for kind, checkpoint, prediction in (("wavenet", wave, groups["mean"]), ("aggregator", agg, aggregated)):
        out = Path(configs[kind]["output_dir"])
        history = []
        if (out / "history.csv").exists():
            with (out / "history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
        write_report(out, kind, history, checkpoint["epoch"], result, prediction, groups["truth"],
                     zoom_quantile, plot_bounds=bounds)
    print(f"{variant}: {result['ccd_groups']:,} test CCD groups, {result['stamps']:,} stamps", flush=True)
    for name, method in result["methods"].items():
        print(f"  {METHOD_NAMES[name]}: CCD-equal RMSE={method['ccd_equal']['coefficient_rmse_um']:.6f} um; "
              f"stamp-equal RMSE={method['stamp_equal']['coefficient_rmse_um']:.6f} um", flush=True)
    return result
