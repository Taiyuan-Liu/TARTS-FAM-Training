#!/usr/bin/env python
"""Generate no-FAM CCD-center OPDs and OCS Z4--Z28 labels from one state."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.utils import iers
import batoid
import galsim
from imsim.camera import get_camera
from lsst.afw import cameraGeom
import numpy as np
from scipy.optimize import least_squares
import yaml

from opd_to_zernikes import fit_opd_fits


def canonical_sha256(payload) -> str:
    """Identify the bound optical configuration for image/label association."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_config_for_state(config_path: Path, states_path: Path, state_index: int):
    with states_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["state_index"]) == state_index]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for state {state_index}, found {len(rows)}")
    state = rows[0]
    config = yaml.safe_load(config_path.read_text())
    config["eval_variables"].update(
        cboresight={"type": "RADec", "ra": f"{state['ra_deg']} deg",
                    "dec": f"{state['dec_deg']} deg"},
        sband=state["band"], azenith=f"{state['zenith_deg']} deg",
        artp=f"{state['rtp_deg']} deg", fmjd=float(state["mjd"]),
        frawSeeing=float(state["seeing"]), iseqnum=int(state["seqid"]),
        ldofs=json.loads(state["dof_json"]),
    )
    config["input"]["telescope"]["file_name"] = f"LSST_{state['band']}.yaml"
    return config, state


def focal_to_field(
    telescope: batoid.Optic,
    *,
    focal_x_mm: float,
    focal_y_mm: float,
    z_offset_m: float,
    wavelength_nm: float,
) -> tuple[float, float, float]:
    """Invert imSim's field-to-focal trace for one detector center.

    Focal coordinates use Rubin DVCS.  Batoid uses EDCS, hence the x/y swap,
    exactly as in ``imsim.batoid_wcs.BatoidWCSFactory._field_to_focal``.
    Returned field angles are radians in the camera-fixed, unrotated frame.
    """
    if z_offset_m:
        det_telescope = telescope.withLocallyShiftedOptic(
            "Detector", [0.0, 0.0, -z_offset_m]
        )
    else:
        det_telescope = telescope

    def residual(field: np.ndarray) -> np.ndarray:
        thx, thy = float(field[0]), float(field[1])
        rays = batoid.RayVector.fromFieldAngles(
            np.asarray([thx]),
            np.asarray([thy]),
            projection="gnomonic",
            optic=telescope,
            wavelength=float(wavelength_nm) * 1.0e-9,
        )
        det_telescope.trace(rays)
        traced_x_mm = float(rays.y[0]) * 1.0e3
        traced_y_mm = float(rays.x[0]) * 1.0e3
        return np.asarray(
            [traced_x_mm - focal_x_mm, traced_y_mm - focal_y_mm]
        )

    result = least_squares(residual, np.zeros(2, dtype=float))
    if not result.success:
        raise RuntimeError(f"Detector center field inversion failed: {result.message}")
    residual_mm = float(np.linalg.norm(residual(result.x)))
    if residual_mm > 1.0e-6:
        raise RuntimeError(
            f"Detector center field inversion residual is {residual_mm} mm"
        )
    return float(result.x[0]), float(result.x[1]), residual_mm


def resolve_detectors(
    detector_ids: list[int],
    *,
    camera_name: str,
    nominal_telescope: batoid.Optic,
    wavelength_nm: float,
) -> list[dict[str, Any]]:
    camera = get_camera(camera_name)
    records: list[dict[str, Any]] = []
    for detector_id in detector_ids:
        try:
            detector = camera[detector_id]
        except Exception as exc:
            raise ValueError(
                f"Detector ID {detector_id} is not valid for {camera_name}"
            ) from exc
        detector_name = detector.getName()
        focal_x_mm, focal_y_mm = detector.getCenter(cameraGeom.FOCAL_PLANE)
        orientation = detector.getOrientation()
        height_mm = (
            float(orientation.getHeight())
            if hasattr(orientation, "getHeight")
            else 0.0
        )
        thx_rad, thy_rad, inversion_residual_mm = focal_to_field(
            nominal_telescope,
            focal_x_mm=float(focal_x_mm),
            focal_y_mm=float(focal_y_mm),
            z_offset_m=height_mm * 1.0e-3,
            wavelength_nm=wavelength_nm,
        )
        thx_deg = float(np.rad2deg(thx_rad))
        thy_deg = float(np.rad2deg(thy_rad))
        if abs(thx_deg) < 1.0e-12:
            thx_deg = 0.0
        if abs(thy_deg) < 1.0e-12:
            thy_deg = 0.0
        records.append(
            {
                "opd_hdu": len(records),
                "detector_id": int(detector_id),
                "detector_name": detector_name,
                "detector_type": str(detector.getType()),
                "focal_x_mm": float(focal_x_mm),
                "focal_y_mm": float(focal_y_mm),
                "camera_height_mm": height_mm,
                "field_x_deg": thx_deg,
                "field_y_deg": thy_deg,
                "field_inversion_residual_mm": inversion_residual_mm,
            }
        )
    return records


def build_bandpass_and_inputs(config: dict[str, Any]) -> float:
    galsim.config.ProcessInput(config)
    bandpass, _ = galsim.config.BuildBandpass(
        config["image"], "bandpass", config, None
    )
    wavelength_nm = float(bandpass.effective_wavelength)
    config["output"]["opd"]["wavelength"] = wavelength_nm
    return wavelength_nm


def generate_opd(
    config: dict[str, Any], detector_records: list[dict[str, Any]]
) -> Path:
    config["output"]["opd"]["fields"] = [
        {
            "thx": f"{record['field_x_deg']:.17g} deg",
            "thy": f"{record['field_y_deg']:.17g} deg",
        }
        for record in detector_records
    ]
    galsim.config.SetupExtraOutput(config)
    galsim.config.SetupConfigFileNum(config, 0, 0, 0)
    galsim.config.extra.WriteExtraOutputs(config, None)
    output = (
        Path(config["output"]["dir"])
        / config["output"]["opd"]["file_name"]
    )
    return output


def annotate_opd(opd_path: Path, detectors: list[dict]) -> None:
    """Record which detector each OPD image represents."""
    with fits.open(opd_path, mode="update", memmap=False) as hdus:
        for hdu, record in zip(hdus, detectors, strict=True):
            hdu.header["DET_NUM"] = (record["detector_id"], "LsstCamSim detector ID")
            hdu.header["DET_NAME"] = (record["detector_name"], "detector name")
            hdu.header["FLDPOL"] = ("CCD_CENTER", "truth field policy")
            hdu.header["LABFOCUS"] = (0.0, "label focusZ (m); FAM excluded")
            hdu.header["ZKSOURCE"] = ("OPD_PIX", "Zernike truth is fit from OPD pixels")


def generate_truth(config: dict, state: dict, detector_ids: list[int],
                   camera_name: str, output_dir: Path) -> dict:
    """Build the optical model, trace CCD-center OPDs, fit and save labels."""
    telescope = config["input"]["telescope"]
    opd = config["output"]["opd"]
    if float(telescope.get("focusZ", 0.0)) != 0.0:
        raise ValueError("Truth must have zero added FAM focus")
    if len(config["eval_variables"]["ldofs"]) != 50:
        raise ValueError("Expected 50 telescope DOFs")
    if not detector_ids or len(set(detector_ids)) != len(detector_ids):
        raise ValueError("Select a nonempty list of unique detector IDs")
    if (opd["jmax"], opd["eps"], opd["projection"]) != (28, 0.61, "gnomonic"):
        raise ValueError("Expected Z1--Z28, obscuration 0.61 and gnomonic fields")

    # Freeze the physical inputs before GalSim adds runtime objects to config.
    base_telescope = deepcopy(telescope)
    base_telescope.pop("focusZ", None)
    model = {
        "eval_variables": deepcopy(config["eval_variables"]),
        "bandpass": deepcopy(config["image"]["bandpass"]),
        "telescope": base_telescope,
    }
    parameters_sha = canonical_sha256({
        "eval_variables": model["eval_variables"], "bandpass": model["bandpass"],
    })
    model_sha = canonical_sha256({
        "state_parameters_sha256": parameters_sha,
        "telescope_config_without_focusZ": base_telescope,
    })

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    wavelength_nm = build_bandpass_and_inputs(config)
    rotation, _ = galsim.config.ParseValue(opd, "rotTelPos", config, galsim.Angle)
    nominal = batoid.Optic.fromYaml(base_telescope["file_name"])
    detectors = resolve_detectors(
        detector_ids, camera_name=camera_name,
        nominal_telescope=nominal, wavelength_nm=wavelength_nm,
    )
    if len(detectors) == 1:
        first = detectors[0]
        opd_name = f"opd_base_det{first['detector_id']:03d}_{first['detector_name']}_center.fits"
    else:
        opd_name = f"opd_base_{len(detectors):03d}_detectors_center.fits"

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config["output"]["dir"] = str(output_dir)
    opd["file_name"] = opd_name
    opd_path = generate_opd(config, detectors)
    annotate_opd(opd_path, detectors)
    truth = fit_opd_fits(opd_path)
    np.save(output_dir / "zk_true.npy", truth, allow_pickle=False)
    with (output_dir / "detector_fields.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detectors[0]))
        writer.writeheader()
        writer.writerows(detectors)

    manifest = {
        "state_index": int(state["state_index"]),
        "author_config": {"path": state["author_config"], "sha256": state["author_config_sha256"]},
        "state_parameters_sha256": parameters_sha,
        "model_config": model,
        "selection": {
            "camera": camera_name, "detector_ids": detector_ids, "detectors": detectors,
            "field_policy": "ccd_center", "field_coordinate_frame": "camera_fixed_unrotated",
        },
        "physics": {
            "base_telescope_state_sha256": model_sha,
            "base_telescope_state_hash_algorithm": "sha256_of_bound_optical_config",
            "aos_dof_count": 50, "label_focus_m": 0.0, "fam_excluded_from_label": True,
            "image_focus_m": float(state["intra_focus_mm"]) / 1000.0,
            "image_focus_header_mm": float(state["intra_focus_mm"]),
            "effective_wavelength_nm": wavelength_nm,
            "opd_rotTelPos_deg": float(rotation / galsim.degrees),
        },
        "truth_contract": {
            "mode": "annular_zernike_pixel_fit", "fit_noll": [1, 28],
            "stored_noll": [4, 28], "obscuration": 0.61,
            "input_units": "nm", "stored_units": "um", "stored_dtype": "float32",
            "coordinate_frame": "OCS", "zernike_rotation_applied": False,
        },
        "outputs": {
            opd_name: {"planes": len(detectors)},
            "zk_true.npy": {"shape": list(truth.shape), "dtype": str(truth.dtype)},
            "detector_fields.csv": {"rows": len(detectors)},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=here / "truth.yaml")
    parser.add_argument("--states", type=Path, default=(
        here.parent / "output/selection/states.csv"
    ))
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--detector-id", action="append", type=int)
    parser.add_argument("--camera", default="LsstCamSim")
    parser.add_argument("--fea-dir", type=Path)
    parser.add_argument("--bend-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: output/states/state_NNN/truth under this step")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "output/states" / f"state_{args.state_index:03d}" / "truth"
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    config, state = load_config_for_state(args.config, args.states, args.state_index)
    for key, directory in (("fea_dir", args.fea_dir), ("bend_dir", args.bend_dir)):
        if directory is not None:
            config["input"]["telescope"]["fea"][key] = str(directory.expanduser().resolve())
    ids = args.detector_id or list(map(int, state["detector_ids"].split(";")))
    result = generate_truth(config, state, ids, args.camera, args.output_dir)
    print(f"state={args.state_index} CCDs={len(ids)} truth={result['outputs']['zk_true.npy']['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
