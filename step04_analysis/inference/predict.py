#!/usr/bin/env python3
"""Run a Step03 model on packaged WEP stamps, once per real visit/CCD."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "step03_TARTS_training"))
import numpy as np
import pandas as pd
import torch

from common.data import ArrayDataset, normalize_images, sequence_features, wave_metadata
from common.models import load_model
from common.training import predict
from step04_analysis.config import settings


def load_models(wavenet_path, aggregator_path, device):
    """Load the two Step03 checkpoints selected in config.yaml or on the CLI."""
    wave, wave_info = load_model(wavenet_path, "wavenet", device)
    aggregator, aggregator_info = load_model(aggregator_path, "aggregator", device)
    contract = {key: wave_info[key] for key in ("frame", "unit", "noll", "normalization")}
    contract.update(wavenet_file=str(Path(wavenet_path).absolute()),
                    aggregator_file=str(Path(aggregator_path).absolute()),
                    image_shape=[160, 160],
                    wave_precision=aggregator_info["config"]["wave_precision"])
    return wave.eval(), aggregator.eval(), contract


def infer_frame(frame, wave, aggregator, device, precision, batch_size):
    """Only images and acquisition metadata enter either network."""
    columns = ["visit", "band", "detector_id", "detector", "pair_index",
               "intra_donut_id", "extra_donut_id"] + [f"{side}_{field}"
               for side in ("intra", "extra") for field in ("image", "field_x", "field_y", "sn")]
    frame = frame[columns].reset_index(drop=True)
    if frame.empty or frame.visit.nunique() != 1 or set(frame.band) != {"i"}:
        raise ValueError("Expected one nonempty i-band visit")
    if frame.duplicated(["detector_id", "intra_donut_id", "extra_donut_id"]).any():
        raise ValueError("Duplicate pair identity")
    waves = {}
    for side in ("intra", "extra"):
        chunks = []
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start:start + batch_size]
            images = np.stack(rows[f"{side}_image"].to_numpy()).astype(np.float32)
            if images.shape[1:] != (200, 200):
                raise ValueError("Expected centered 200x200 WEP CCS stamps")
            images = normalize_images(images[:, 20:180, 20:180])
            meta = wave_metadata(rows[f"{side}_field_x"], rows[f"{side}_field_y"],
                                 np.full(len(rows), side == "intra"), np.full(len(rows), 3))
            chunks.append(predict(wave, ArrayDataset(dict(image=images, **meta)),
                                  "wavenet", device, precision, batch_size))
        waves[side] = np.concatenate(chunks)
    ids, names, values, means, medians, lengths = [], [], [], [], [], []
    for detector, group in frame.groupby("detector_id", sort=True):
        rows = group.index.to_numpy()
        predictions = np.concatenate([waves[side][rows] for side in ("intra", "extra")])
        fields = np.concatenate([group[[f"{side}_field_x", f"{side}_field_y"]].to_numpy(np.float32)
                                 for side in ("intra", "extra")])
        snr = np.concatenate([group[f"{side}_sn"].to_numpy(np.float32) for side in ("intra", "extra")])
        features, mean = sequence_features(predictions, fields, snr)
        result = predict(aggregator, ArrayDataset(dict(features=features[None], mean=mean[None])),
                         "aggregator", device, "fp32", 1)[0]
        if group.detector.nunique() != 1:
            raise ValueError("Detector ID/name mismatch")
        ids.append(int(detector)); names.append(str(group.detector.iloc[0]))
        values.append(result); means.append(mean); medians.append(np.median(predictions, axis=0))
        lengths.append(len(predictions))
    return dict(visit=np.int64(frame.visit.iloc[0]), noll=np.arange(4, 29),
                frame=np.asarray("CCS"), unit=np.asarray("micron"),
                detector_id=np.asarray(ids), detector_name=np.asarray(names),
                prediction_ccs_um=np.asarray(values), mean_ccs_um=np.asarray(means),
                median_ccs_um=np.asarray(medians), length=np.asarray(lengths),
                pair_detector_id=frame.detector_id.to_numpy(np.int64),
                pair_index=frame.pair_index.to_numpy(np.int64),
                intra_donut_id=frame.intra_donut_id.astype(str).to_numpy(dtype=str),
                extra_donut_id=frame.extra_donut_id.astype(str).to_numpy(dtype=str),
                wave_intra_ccs_um=waves["intra"], wave_extra_ccs_um=waves["extra"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--wavenet", type=Path, help="WaveNet model.pt")
    parser.add_argument("--aggregator", type=Path, help="Aggregator model.pt")
    parser.add_argument("--visits-file", type=Path)
    parser.add_argument("--visit", action="append", type=int)
    parser.add_argument("--max-detectors", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = settings(args.config)
    if args.visit:
        visits = args.visit
    else:
        visits = pd.read_csv(args.visits_file or config["visits_file"])["visit"].astype("int64").tolist()
        if not visits or len(set(visits)) != len(visits):
            raise ValueError("Expected a nonempty, unique visit list")
    device = torch.device(args.device or config["device"])
    torch.set_num_threads(config["torch_threads"])
    paths = config["models"][args.variant]
    wave, aggregator, contract = load_models(args.wavenet or paths["wavenet"],
                                             args.aggregator or paths["aggregator"], device)
    out = args.output_dir or config["output_root"] / "predictions" / args.variant
    out.mkdir(parents=True, exist_ok=False)
    record = dict(variant=args.variant, visits=visits, **contract)
    (out / "model.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for visit in visits:
        path = config["packages_dir"] / f"visit_{visit}.pkl"
        frame = pd.read_pickle(path)  # Trusted local WEP package, never a model or code input.
        if set(frame.visit) != {visit}:
            raise ValueError(f"Visit identity mismatch: {path}")
        if args.max_detectors:
            frame = frame[frame.detector_id.isin(sorted(frame.detector_id.unique())[:args.max_detectors])]
        result = infer_frame(frame, wave, aggregator, device, contract["wave_precision"], config["batch_size"])
        blur = "pair_fwhm" if "pair_fwhm" in frame else "fwhm"
        result["pair_fwhm"] = frame[blur].to_numpy(np.float32)
        np.savez(out / f"visit_{visit}.npz", **result)
        print(f"visit {visit}: {len(result['detector_id'])} CCDs, {len(frame)} pairs", flush=True)


if __name__ == "__main__":
    main()
