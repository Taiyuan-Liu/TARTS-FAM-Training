#!/usr/bin/env python
"""Build coarse train/val/test NPZ shards from labelled WEP stamps."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile

from astropy.table import QTable
import numpy as np

SPLITS = ("train", "val", "test")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"
ARRAY_SPECS = {
    "image": (np.float32, (160, 160)),
    "field_x_deg": (np.float32, ()),
    "field_y_deg": (np.float32, ()),
    "intra": (np.int8, ()),
    "band": (np.int8, ()),
    "seq_num": (np.int32, ()),
    "state_index": (np.int16, ()),
    "detector_id": (np.int16, ()),
    "source_stamp_index": (np.int16, ()),
    "sample_id": (np.int32, ()),
    "snr": (np.float32, ()),
    "rtp_deg": (np.float32, ()),
    "zk_true_ccs_um": (np.float32, (25,)),
    "zk_true_ocs_um": (np.float32, (25,)),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_splits(path):
    """Read persistent membership; never reassign states when inputs grow."""
    assignments, group_splits = {}, {}
    for row in read_csv(path):
        state, group, split = int(row["state_index"]), int(row["split_group_id"]), row["split"]
        if state in assignments or split not in SPLITS or not 0 <= state <= 32767:
            raise ValueError(f"Invalid or duplicate split row: {row}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Pointing-twin group {group} crosses splits")
        group_splits[group] = split
        assignments[state] = (split, group)
    if not assignments:
        raise ValueError("Empty split registry")
    return assignments


def collect_samples(inputs, assignments):
    """Select effective stamps and bind each row to its image and target."""
    from galsim.zernike import zernikeRotMatrix

    rows, input_keys, state_models = [], set(), {}
    if inputs.is_dir():
        entries = [{"stage2_dir": str(path.parent)}
                   for path in sorted(inputs.glob("state_*/*/det*/manifest.json"))]
        base = inputs
    else:
        entries, base = read_csv(inputs), inputs.parent
    for entry in entries:
        directory = (base / entry["stage2_dir"]).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        physics = manifest["physics"]
        if (physics["stamp_image_frame"] != "CCS" or physics["training_target_frame"] != "CCS"
                or physics["published_truth_frame"] != "entrance_pupil_OCS"
                or physics["units"] != "um" or physics["noll_indices"] != list(range(4, 29))
                or physics["fam_diversity_in_truth"] or physics["focal_side_sign_flip"]
                or physics["truth_granularity"] != "detector_center_shared_not_per_star"
                or not np.isclose(physics["annular_obscuration"], 0.61, rtol=0, atol=1e-12)):
            raise ValueError("Expected no-added-FAM CCS Z4--Z28 targets in microns")
        identity = manifest["image_identity"]
        detector, side = int(identity["detector_id"]), identity["side"]
        if side not in ("intra", "extra"):
            raise ValueError(f"Invalid focal side: {side}")
        state, remainder = divmod(int(identity["seq_num"]) - (1 if side == "intra" else 2), 2)
        if state < 0 or remainder:
            raise ValueError("Exposure sequence and focal side disagree")
        if not (identity["focus_mm"] < 0 if side == "intra" else identity["focus_mm"] > 0):
            raise ValueError("Focal side and FOCUSZ sign disagree")
        key = (state, detector, side)
        if key in input_keys:
            raise ValueError(f"Duplicate CCD-side input: {key}")
        input_keys.add(key)
        if state not in assignments:
            raise ValueError(f"State {state} has no fixed split; extend the registry explicitly")
        split, group = assignments[state]
        model = manifest["provenance_gate"]["base_telescope_state_sha256"]
        if state in state_models and state_models[state] != model:
            raise ValueError(f"State {state} has inconsistent telescope models")
        state_models[state] = model
        stage1 = ((base / entry["stage1_dir"]) if entry.get("stage1_dir")
                  else (directory / manifest["stage1"]["directory"])).resolve()
        if sha256(stage1 / "manifest.json") != manifest["stage1"]["manifest_sha256"]:
            raise ValueError(f"Stage-1 association changed: {stage1}")
        source = json.loads((stage1 / "manifest.json").read_text())
        if (source.get("state_index", state), source["detector_id"], source["side"]) != key:
            raise ValueError("Stage-1/2 CCD-side identities disagree")
        for name in ("detector_name", "seq_num", "day_obs", "band", "group_id", "focus_mm"):
            if source["input"][name] != identity[name]:
                raise ValueError(f"Stage-1/2 exposure metadata disagree: {name}")
        for name, association_key in (("donut_stamps.fits", "donut_stamps_sha256"),
                                      ("stamp_metadata.ecsv", "stamp_metadata_sha256")):
            checksum = sha256(stage1 / name)
            if checksum != source["products"][name]["sha256"] or checksum != manifest["stage1"][association_key]:
                raise ValueError(f"Changed stamp product: {stage1 / name}")
        for name in ("stamp_truth_ccs_um.npy", "stamp_truth_ocs_um.npy", "stamp_label_index.ecsv"):
            if sha256(directory / name) != manifest["products"][name]["sha256"]:
                raise ValueError(f"Changed label product: {directory / name}")
        metadata = QTable.read(directory / "stamp_label_index.ecsv")
        ccs = np.load(directory / "stamp_truth_ccs_um.npy", allow_pickle=False)
        ocs = np.load(directory / "stamp_truth_ocs_um.npy", allow_pickle=False)
        n = manifest["one_to_one_contract"]["row_count"]
        if len(metadata) != n or source["counts"]["stamps"] != n or manifest["stage1"]["count"] != n:
            raise ValueError("Stage-1/2 row counts disagree")
        for truth in (ccs, ocs):
            if truth.shape != (n, 25) or truth.dtype != np.float32 or not np.isfinite(truth).all():
                raise ValueError("Invalid per-stamp target array")
        if not np.array_equal(metadata["stamp_index"], np.arange(n)):
            raise ValueError("Label index must use contiguous FITS row numbers")
        if not np.isfinite(physics["rtp_deg"]):
            raise ValueError("Non-finite rotTelPos")
        if n:
            if not np.all(ocs == ocs[0]) or not np.all(ccs == ccs[0]):
                raise ValueError("Expected one CCD-center truth shared by all stamps")
            full = np.zeros(29)
            full[4:29] = ocs[0]
            expected = (full @ zernikeRotMatrix(28, np.deg2rad(physics["rtp_deg"])))[4:29]
            if not np.allclose(ccs[0], expected, rtol=1e-6, atol=1e-7):
                raise ValueError("Stored CCS targets disagree with OCS @ R(+rotTelPos)")
        for row in metadata:
            if (int(row["detector_id"]), str(row["side"])) != (detector, side):
                raise ValueError("Metadata and manifest identities disagree")
            if "state_index" in metadata.colnames and int(row["state_index"]) != state:
                raise ValueError("Metadata state disagrees with the exposure sequence")
            for name in ("detector_name", "seq_num", "day_obs", "band", "group_id", "focus_mm"):
                if row[name] != identity[name]:
                    raise ValueError(f"Metadata and image identity disagree: {name}")
            if int(row["source_truth_detector_row"]) != manifest["label_source"]["source_detector_row"]:
                raise ValueError("Metadata and CCD truth row disagree")
            if not np.isclose(row["rtp_deg"], physics["rtp_deg"], rtol=0, atol=1e-12):
                raise ValueError("Metadata and target rotation disagree")
            i = int(row["stamp_index"])
            if i != int(row["donut_stamps_index"]) or i != int(row["stamp_truth_array_row"]):
                raise ValueError("Image/label row mapping differs")
            if int(row["effective"]) != 1:
                continue
            rows.append({
                "state_index": state, "detector_id": detector, "side": side,
                "split": split, "split_group_id": group, "detector_name": str(row["detector_name"]),
                "seq_num": int(row["seq_num"]), "band": str(row["band"]),
                "band_index": "ugrizy".index(str(row["band"])), "group_id": str(row["group_id"]),
                "source_stamp_index": i, "donut_id": str(row["donut_id"]),
                "field_x_deg": float(row["field_ccs_x_deg"]),
                "field_y_deg": float(row["field_ccs_y_deg"]),
                "snr": float(row["sn"]), "rtp_deg": float(row["rtp_deg"]),
                "stage1_dir": str(stage1), "stage2_dir": str(directory),
                "_ccs": ccs[i], "_ocs": ocs[i],
            })
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["state_index"], r["detector_id"],
                            0 if r["side"] == "intra" else 1, r["source_stamp_index"]))
    for i, row in enumerate(rows):
        row["sample_id"] = i
    if not rows:
        raise ValueError("No effective stamps in the input list")
    return rows, input_keys


def plan_shards(samples, capacity):
    plans = []
    for split in SPLITS:
        selected = [row for row in samples if row["split"] == split]
        for number, start in enumerate(range(0, len(selected), capacity)):
            chunk = selected[start:start + capacity]
            relative = f"{split}/part-{number:04d}.npz"
            plans.append((relative, chunk))
    return plans


def write_shard(task):
    """One worker writes one bounded-size shard using WEP's CCS image view."""
    from lsst.ts.wep.task.donutStamps import DonutStamps

    root, relative, rows = task
    arrays = {name: np.empty((len(rows), *shape), dtype=dtype)
              for name, (dtype, shape) in ARRAY_SPECS.items()}
    by_fits = defaultdict(list)
    for i, row in enumerate(rows):
        by_fits[row["stage1_dir"]].append((i, row))
    for directory, selected in by_fits.items():
        stamps = DonutStamps.readFits(str(Path(directory) / "donut_stamps.fits"))
        for i, row in selected:
            stamp = stamps[row["source_stamp_index"]]
            image = np.asarray(stamp.wep_im.image, dtype=np.float32)
            if image.shape != (200, 200) or not np.isfinite(image).all():
                raise ValueError("Expected a finite 200x200 CCS image")
            if str(stamp.donut_id) != row["donut_id"]:
                raise ValueError("Packed image and target donut IDs differ")
            if not np.allclose(stamp.wep_im.fieldAngle,
                               [row["field_x_deg"], row["field_y_deg"]], rtol=0, atol=1e-7):
                raise ValueError("Packed image and metadata fields differ")
            arrays["image"][i] = image[20:180, 20:180]
            for name in ("field_x_deg", "field_y_deg", "seq_num", "state_index",
                         "detector_id", "source_stamp_index", "sample_id", "snr", "rtp_deg"):
                arrays[name][i] = row[name]
            arrays["intra"][i] = int(row["side"] == "intra")
            arrays["band"][i] = row["band_index"]
            arrays["zk_true_ccs_um"][i], arrays["zk_true_ocs_um"][i] = row["_ccs"], row["_ocs"]
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("Non-finite packed values")
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, split=np.asarray(rows[0]["split"]), **arrays)
    return {"path": relative, "samples": len(rows)}


def build_dataset(inputs, splits, output, samples_per_shard=512, workers=1):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs, splits, output = (p.expanduser().resolve() for p in (inputs, splits, output))
    if output.exists():
        raise FileExistsError(output)
    if samples_per_shard < 1 or workers < 1:
        raise ValueError("Shard size and worker count must be positive")
    assignments = load_splits(splits)
    samples, input_keys = collect_samples(inputs, assignments)
    plans = plan_shards(samples, samples_per_shard)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        root = Path(temporary)
        for split in SPLITS:
            (root / split).mkdir()
        tasks = [(str(root), relative, rows) for relative, rows in plans]
        if workers == 1:
            products = [write_shard(task) for task in tasks]
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                products = list(pool.map(write_shard, tasks))
        write_csv(root / "splits.csv", [
            dict(state_index=state, split_group_id=group, split=split)
            for state, (split, group) in sorted(assignments.items())
        ], ("state_index", "split_group_id", "split"))
        counts = Counter(row["split"] for row in samples)
        summary = {
            "samples": len(samples),
            "sample_counts": {split: counts[split] for split in SPLITS},
            "ccd_sides": len(input_keys), "shards": products,
        }
        root.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=OUTPUT_ROOT / "labels",
                        help="Label root (default: output/labels) or CSV with a stage2_dir column")
    parser.add_argument("--splits", type=Path, default=Path(__file__).with_name("splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "dataset")
    parser.add_argument("--samples-per-shard", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = build_dataset(args.inputs, args.splits, args.output_dir, args.samples_per_shard,
                           args.workers)
    print(f"samples={result['samples']} shards={len(result['shards'])} "
          f"splits={result['sample_counts']}")


if __name__ == "__main__":
    main()
