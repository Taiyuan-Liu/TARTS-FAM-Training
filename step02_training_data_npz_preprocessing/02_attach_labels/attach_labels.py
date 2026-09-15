#!/usr/bin/env python
"""Attach CCD-center OCS and CCS Z4--Z28 targets to WEP stamps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from astropy.table import QTable
import galsim
import numpy as np

AXISYMMETRIC_NOLL = (4, 11, 22)
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"
SIMULATION_OUTPUT = OUTPUT_ROOT.parent.parent / "step01_simulation_dataset_generation/output/states"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_stamps(directory: Path):
    from lsst.ts.wep.task.donutStamps import DonutStamps

    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("donut_stamps.fits", "stamp_metadata.ecsv"):
        if sha256(directory / name) != manifest["products"][name]["sha256"]:
            raise ValueError(f"Changed stage-1 product: {name}")
    metadata = QTable.read(directory / "stamp_metadata.ecsv")
    stamps = DonutStamps.readFits(str(directory / "donut_stamps.fits"))
    if len(stamps) != len(metadata) or len(stamps) != manifest["counts"]["stamps"]:
        raise ValueError("FITS, metadata and manifest row counts differ")
    if not np.array_equal(metadata["stamp_index"], np.arange(len(stamps))):
        raise ValueError("stamp_index must equal the FITS row")
    for i, stamp in enumerate(stamps):
        row = metadata[i]
        if int(row["donut_stamps_index"]) != i:
            raise ValueError("Metadata must retain the FITS row order")
        if (int(row["detector_id"]), str(row["side"])) != (manifest["detector_id"], manifest["side"]):
            raise ValueError("Metadata and stamp detector/side disagree")
        for key in ("detector_name", "seq_num", "day_obs", "band", "group_id", "focus_mm"):
            if row[key] != manifest["input"][key]:
                raise ValueError(f"Metadata and stamp exposure disagree: {key}")
        if str(stamp.donut_id) != str(row["donut_id"]):
            raise ValueError("FITS and metadata donut IDs differ")
        image = np.asarray(stamp.wep_im.image)
        if image.shape != (200, 200) or not np.isfinite(image).all():
            raise ValueError("Expected a finite 200x200 CCS image")
        field = [row["field_ccs_x_deg"], row["field_ccs_y_deg"]]
        if not np.allclose(stamp.wep_im.fieldAngle, field, rtol=0, atol=1e-12):
            raise ValueError("FITS and metadata CCS field coordinates differ")
    return metadata, manifest


def load_truth(directory: Path, detector_id: int, detector_name: str):
    """Select by detector identity, not by an assumed fixed row order."""
    manifest = json.loads((directory / "manifest.json").read_text())
    ids = [int(value) for value in manifest["selection"]["detector_ids"]]
    if ids.count(detector_id) != 1:
        raise ValueError(f"Detector {detector_id} must occur exactly once in truth")
    index = ids.index(detector_id)
    detector = manifest["selection"]["detectors"][index]
    if detector["detector_id"] != detector_id or detector["detector_name"] != detector_name:
        raise ValueError("Truth detector ID/name mapping differs from the image")
    physics = manifest["physics"]
    basis = manifest["truth_contract"]
    if not physics["fam_excluded_from_label"] or abs(physics["label_focus_m"]) > 1e-15:
        raise ValueError("Truth must exclude added FAM diversity")
    if basis["stored_noll"] != [4, 28] or basis["stored_units"] != "um":
        raise ValueError("Truth must be Z4--Z28 in microns")
    if not np.isclose(basis["obscuration"], 0.61, rtol=0, atol=1e-12):
        raise ValueError("Expected annular obscuration 0.61")
    if basis["zernike_rotation_applied"] or basis.get("coordinate_frame", "OCS") != "OCS":
        raise ValueError("The source truth must be unrotated OCS coefficients")
    truth = np.load(directory / "zk_true.npy", allow_pickle=False)
    if truth.shape != (len(ids), 25) or truth.dtype != np.float32 or not np.isfinite(truth).all():
        raise ValueError("Expected finite float32 truth with shape (CCD count, 25)")
    return truth[index].copy(), manifest, detector, index


def check_image_truth(image_path: Path, stamp_manifest: dict, label_dir: Path, truth: dict):
    """Verify the simulation's direct image/label association."""
    image = json.loads(image_path.read_text())
    if image["label_manifest_sha256"] != sha256(label_dir / "manifest.json"):
        raise ValueError("Image was generated with a different truth manifest")
    if image["base_telescope_state_sha256"] != truth["physics"]["base_telescope_state_sha256"]:
        raise ValueError("Image and truth use different base telescope states")
    identity = stamp_manifest["input"]
    ids = [int(value) for value in image["detector_ids"]]
    detector_id = stamp_manifest["detector_id"]
    if ids.count(detector_id) != 1 or image["detector_names"][ids.index(detector_id)] != identity["detector_name"]:
        raise ValueError("Image and stamp detector identities disagree")
    if image["seqnum"] != identity["seq_num"] or image["fam_side"] != stamp_manifest["side"]:
        raise ValueError("Image and stamp exposure/side identities disagree")
    if not np.isclose(image["focus_m"] * 1000, identity["focus_mm"], rtol=0, atol=1e-9):
        raise ValueError("Image and stamp FAM offsets disagree")
    return image


def rotate_ocs_to_ccs(
    zk_ocs_um: np.ndarray, rtp_deg: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Rotate a row-vector Z4..Z28 from OCS into CCS.

    Rubin production aggregation uses ``CCS @ R(-RTP) -> OCS``.  This is its
    inverse: ``OCS @ R(+RTP) -> CCS``.  The full Noll matrix is used before
    slicing so the convention is explicit.
    """
    if zk_ocs_um.shape != (25,):
        raise ValueError(f"Expected Z4..Z28, got {zk_ocs_um.shape}")
    rtp_rad = np.deg2rad(rtp_deg)
    full_ocs = np.zeros(29, dtype=np.float64)
    full_ocs[4:29] = np.asarray(zk_ocs_um, dtype=np.float64)
    plus = galsim.zernike.zernikeRotMatrix(28, rtp_rad)
    minus = galsim.zernike.zernikeRotMatrix(28, -rtp_rad)
    full_ccs = full_ocs @ plus
    recovered = full_ccs @ minus
    zk_ccs = full_ccs[4:29]
    roundtrip = float(np.max(np.abs(recovered[4:29] - full_ocs[4:29])))
    # Compare like with like in float64.  The published source is float32, but
    # ``full_ocs`` is its exact float64 promotion; comparing against a fresh
    # float32 norm would measure dtype rounding rather than rotation fidelity.
    norm_error = float(abs(np.linalg.norm(full_ccs) - np.linalg.norm(full_ocs)))
    axis_error = float(
        max(abs(zk_ccs[j - 4] - float(zk_ocs_um[j - 4])) for j in AXISYMMETRIC_NOLL)
    )
    if roundtrip > 1e-12:
        raise RuntimeError(f"OCS -> CCS -> OCS round trip failed: {roundtrip}")
    if norm_error > 1e-12:
        raise RuntimeError(f"Zernike rotation did not preserve norm: {norm_error}")
    if axis_error > 1e-12:
        raise RuntimeError(f"Axisymmetric modes changed under rotation: {axis_error}")
    return zk_ccs.astype(np.float32), {
        "float64_roundtrip_max_abs_um": roundtrip,
        "float64_norm_abs_difference_um": norm_error,
        "axisymmetric_z4_z11_z22_max_abs_difference_um": axis_error,
    }


def attach_labels(stage1: Path, truth_dir: Path | None = None, output: Path | None = None,
                  image_manifest: Path | None = None) -> dict:
    """Match detector -> rotate OCS to CCS -> repeat target over stamp rows."""
    stage1 = stage1.expanduser().resolve()
    metadata, source = read_stamps(stage1)
    identity = source["input"]
    side = source["side"]
    if side not in ("intra", "extra"):
        raise ValueError("Expected intra or extra")
    state, remainder = divmod(int(identity["seq_num"]) - (1 if side == "intra" else 2), 2)
    if state < 0 or remainder or source.get("state_index", state) != state:
        raise ValueError("State, exposure sequence and focal side disagree")
    if not (identity["focus_mm"] < 0 if side == "intra" else identity["focus_mm"] > 0):
        raise ValueError("Focal side and FOCUSZ sign disagree")
    if output is None:
        output = OUTPUT_ROOT / "labels" / f"state_{state:03d}" / side / f"det{source['detector_id']:03d}_{identity['detector_name']}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.expanduser().resolve()
    truth_dir = (truth_dir or SIMULATION_OUTPUT / f"state_{state:03d}" / "truth").expanduser().resolve()
    ocs, truth, detector, truth_row = load_truth(
        truth_dir, source["detector_id"], source["input"]["detector_name"]
    )
    if image_manifest is None:
        image_manifest = SIMULATION_OUTPUT / f"state_{state:03d}" / side / "image_pass.yaml.manifest.json"
    image_manifest = image_manifest.expanduser().resolve()
    association = check_image_truth(image_manifest, source, truth_dir, truth)
    if association.get("state_index", state) != state or truth.get("state_index", state) != state:
        raise ValueError("Truth, image and stamp state indices disagree")
    rtp_deg = float(truth["physics"]["opd_rotTelPos_deg"])
    if not np.isfinite(rtp_deg):
        raise ValueError("Non-finite rotTelPos")
    ccs, rotation_checks = rotate_ocs_to_ccs(ocs, rtp_deg)
    n = len(metadata)
    metadata["rtp_deg"] = np.full(n, rtp_deg)
    metadata["source_truth_detector_row"] = np.full(n, truth_row, dtype=np.int32)
    metadata["label_reference_field_ccs_x_deg"] = np.full(n, detector["field_x_deg"])
    metadata["label_reference_field_ccs_y_deg"] = np.full(n, detector["field_y_deg"])
    metadata["stamp_truth_array_row"] = np.arange(n, dtype=np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        np.save(directory / "stamp_truth_ocs_um.npy", np.repeat(ocs[None], n, axis=0))
        np.save(directory / "stamp_truth_ccs_um.npy", np.repeat(ccs[None], n, axis=0))
        metadata.write(directory / "stamp_label_index.ecsv", format="ascii.ecsv")
        manifest = {
            "stage1": {"directory": os.path.relpath(stage1, output), "count": n,
                       "manifest_sha256": sha256(stage1 / "manifest.json"),
                       "donut_stamps_sha256": sha256(stage1 / "donut_stamps.fits"),
                       "stamp_metadata_sha256": sha256(stage1 / "stamp_metadata.ecsv")},
            "label_source": {"directory": str(truth_dir), "source_detector_row": truth_row,
                             "reference_field_ccs_deg": [detector["field_x_deg"], detector["field_y_deg"]],
                             "manifest_sha256": sha256(truth_dir / "manifest.json"),
                             "zk_true_sha256": sha256(truth_dir / "zk_true.npy")},
            "image_identity": {**{key: identity[key] for key in (
                "seq_num", "day_obs", "band", "group_id", "focus_mm", "detector_name")},
                "detector_id": source["detector_id"], "side": side},
            "one_to_one_contract": {"row_count": n,
                                    "mapping": "stamp_index == donut_stamps_index == stamp_truth array row"},
            "physics": {"noll_indices": list(range(4, 29)), "units": "um",
                        "annular_obscuration": 0.61, "stamp_image_frame": "CCS",
                        "training_target_frame": "CCS", "published_truth_frame": "entrance_pupil_OCS",
                        "rtp_deg": rtp_deg, "ocs_to_ccs_row_formula": "Z_CCS = Z_OCS @ R(+rotTelPos)",
                        "rotation_checks": rotation_checks, "focal_side_sign_flip": False,
                        "truth_granularity": "detector_center_shared_not_per_star",
                        "fam_diversity_in_truth": False},
            "provenance_gate": {"image_pass_manifest": str(image_manifest),
                                "image_pass_manifest_sha256": sha256(image_manifest),
                                "base_telescope_state_sha256": truth["physics"]["base_telescope_state_sha256"]},
            "products": {name: {"bytes": (directory / name).stat().st_size,
                                "sha256": sha256(directory / name)} for name in (
                "stamp_truth_ocs_um.npy", "stamp_truth_ccs_um.npy", "stamp_label_index.ecsv")},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        directory.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path,
                        help="Default: Step01 output/states/state_NNN/truth")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: output/labels/state_NNN/SIDE/detNNN_NAME")
    parser.add_argument("--image-manifest", type=Path,
                        help="Default: Step01 output/states/state_NNN/SIDE/image_pass.yaml.manifest.json")
    args = parser.parse_args()
    result = attach_labels(args.stage1_dir, args.label_dir, args.output_dir, args.image_manifest)
    identity = result["image_identity"]
    print(f"seq={identity['seq_num']} detector={identity['detector_id']} "
          f"side={identity['side']} labelled_stamps={result['one_to_one_contract']['row_count']}")


if __name__ == "__main__":
    main()
