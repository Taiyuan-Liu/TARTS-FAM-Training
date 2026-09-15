"""Label-free real targets and per-stamp prediction caches."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import ArrayDataset, real_wave, sequence_features, sha256, wave_metadata
from common.models import load_model
from common.training import local_path, predict, prediction_cache


def visit_plan(root, heldout_manifest):
    with (Path(root) / "visit_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with Path(heldout_manifest).open(newline="") as stream:
        heldout = {int(row["visit"]) for row in csv.DictReader(stream)}
    visits = [int(row["visit"]) for row in rows]
    if not rows or len(visits) != len(set(visits)) or set(visits) & heldout:
        raise ValueError("Empty/duplicate real visit plan or overlap with held-out visits")
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Real adaptation requires train and val visits only")
    for row in rows:
        row["source_pickle"] = str(local_path(row["source_pickle"], root))
    return rows


def real_identity(root, rows, heldout_manifest):
    """Bind immutable target sources and the held-out visit registry to a run."""
    pickles = []
    for row in rows:
        stat = Path(row["source_pickle"]).stat()
        pickles.append((row["visit"], row["split"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    shards = [(str(path.relative_to(root)), sha256(path)) for path in sorted((Path(root) / "shards").glob("*/*.npz"))]
    value = dict(pickles=pickles, shards=shards, heldout_sha256=sha256(heldout_manifest))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wave_targets(root, rows):
    targets = {}
    for split in ("train", "val"):
        targets[split] = real_wave(root, split)
        visits = {int(row["visit"]) for row in rows if row["split"] == split}
        if set(targets[split].arrays["visit"]) != visits:
            raise ValueError(f"Real target shards do not match the frozen {split} visit plan")
    return targets


def aggregate_predictions(arrays):
    """Build CCD groups in memory from cached predictions and physical metadata."""
    groups = defaultdict(list)
    for i, key in enumerate(zip(arrays["visit"], arrays["detector_id"])):
        groups[key].append(i)
    output = defaultdict(list)
    for (visit, detector), indices in sorted(groups.items()):
        indices.sort(key=lambda i: (-int(arrays["intra"][i]), int(arrays["pair_row"][i])))
        pred = arrays["prediction_ccs_um"][indices]
        fields = np.column_stack([arrays["field_x_deg"][indices], arrays["field_y_deg"][indices]])
        features, mean = sequence_features(pred, fields, arrays["snr"][indices])
        for key, value in dict(features=features, mean=mean, visit=int(visit),
                               detector_id=int(detector), length=len(indices)).items():
            output[key].append(value)
    return ArrayDataset(output)


def aggregator_targets(rows, wave_checkpoint, cache_dir, device, precision, batch_size):
    """Cache one prediction per stamp, never Danish labels or pre-built sequences."""
    targets, model = {}, None
    columns = ["visit", "band", "detector_id"] + [f"{side}_{field}"
               for side in ("intra", "extra") for field in ("image", "field_x", "field_y", "sn")]
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        files = [(row["visit"], Path(row["source_pickle"]).stat().st_size,
                  Path(row["source_pickle"]).stat().st_mtime_ns,
                  Path(row["source_pickle"]).stat().st_ctime_ns) for row in selected]
        signature = dict(wave_checkpoint_sha256=sha256(wave_checkpoint), precision=precision,
                         visits=files, preprocessing="center160_population_zscore_v1")

        def build():
            nonlocal model
            if model is None:
                model, _ = load_model(wave_checkpoint, "wavenet", device)
            parts = defaultdict(list)
            for row in selected:
                path = Path(row["source_pickle"])
                stat_before = path.stat()
                # These trusted local pickles are reduced immediately to image/metadata columns.
                frame = pd.read_pickle(path)[columns].reset_index(drop=True)
                if set(frame["visit"]) != {int(row["visit"])} or set(frame["band"]) != {"i"}:
                    raise ValueError(f"Visit/band mismatch: {path}")
                for side in ("intra", "extra"):
                    for start in range(0, len(frame), batch_size):
                        chunk = frame.iloc[start:start + batch_size]
                        images = np.stack(chunk[f"{side}_image"].to_numpy()).astype(np.float32)
                        if images.shape[1:] != (200, 200):
                            raise ValueError("Expected packaged WEP 200x200 images")
                        images = images[:, 20:180, 20:180].copy()
                        images -= images.mean((1, 2), keepdims=True)
                        std = images.std((1, 2), keepdims=True)
                        if np.any(std <= 0) or not np.isfinite(std).all():
                            raise ValueError("Constant/non-finite real stamp")
                        images /= std
                        inputs = wave_metadata(chunk[f"{side}_field_x"], chunk[f"{side}_field_y"],
                                               np.full(len(chunk), side == "intra"), np.full(len(chunk), 3))
                        pred = predict(model, ArrayDataset(dict(image=images, **inputs)),
                                       "wavenet", device, precision, batch_size)
                        values = dict(prediction_ccs_um=pred, visit=np.full(len(chunk), int(row["visit"])),
                                      detector_id=chunk["detector_id"].to_numpy(np.int32),
                                      intra=np.full(len(chunk), side == "intra", np.int8),
                                      pair_row=chunk.index.to_numpy(np.int32),
                                      field_x_deg=chunk[f"{side}_field_x"].to_numpy(np.float32),
                                      field_y_deg=chunk[f"{side}_field_y"].to_numpy(np.float32),
                                      snr=chunk[f"{side}_sn"].to_numpy(np.float32))
                        for key, value in values.items():
                            parts[key].append(value)
                stat_after = path.stat()
                if (stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ctime_ns) != (
                        stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ctime_ns):
                    raise ValueError(f"Target changed while reading: {path}")
                print(f"Real predictions: {split} visit={row['visit']}", flush=True)
            return {key: np.concatenate(value) for key, value in parts.items()}

        arrays = prediction_cache(Path(cache_dir) / f"real_{split}.npz", signature, build)
        targets[split] = aggregate_predictions(arrays)
    return targets
