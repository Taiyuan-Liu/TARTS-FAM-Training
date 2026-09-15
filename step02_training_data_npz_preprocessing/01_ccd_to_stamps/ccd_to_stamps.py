#!/usr/bin/env python
"""Extract 200x200 WEP donuts from one imSim amplifier raw."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
import numpy as np

CAMERA_NAME = "LSSTCamSim"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"
METADATA_SCHEMA = [
    ("stamp_index", "i8"),
    ("donut_id", "U128"),
    ("source_catalog_row", "i8"),
    ("detector_id", "i8"),
    ("detector_name", "U16"),
    ("camera", "U32"),
    ("side", "U8"),
    ("focus_mm", "f8"),
    ("band", "U8"),
    ("day_obs", "i8"),
    ("seq_num", "i8"),
    ("group_id", "U128"),
    ("donut_stamps_index", "i8"),
    ("stamp_height", "i8"),
    ("stamp_width", "i8"),
    ("bbox_x0", "i8"),
    ("bbox_y0", "i8"),
    ("detected_centroid_x", "f8"),
    ("detected_centroid_y", "f8"),
    ("final_centroid_x", "f8"),
    ("final_centroid_y", "f8"),
    ("ra_deg", "f8"),
    ("dec_deg", "f8"),
    ("field_dvcs_x_deg", "f8"),
    ("field_dvcs_y_deg", "f8"),
    ("field_ccs_x_deg", "f8"),
    ("field_ccs_y_deg", "f8"),
    ("source_flux_njy", "f8"),
    ("sn", "f8"),
    ("effective", "i8"),
    ("entropy", "f8"),
    ("recenter_flag", "i8"),
    ("bad_pixel_fraction", "f8"),
]


def first_header_value(hdus: fits.HDUList, *keys: str) -> Any:
    for key in keys:
        for hdu in hdus:
            if key in hdu.header:
                return hdu.header[key]
    raise KeyError(f"None of {keys!r} occurs in the FITS headers")


def inspect_amp_raw(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.name.startswith("amp_"):
        raise ValueError(f"The primary input must be amp_*.fits.fz, got {path.name}")
    with fits.open(path, memmap=False) as hdus:
        if len(hdus) != 17:
            raise ValueError(f"Expected primary + 16 amplifier HDUs, got {len(hdus)}")
        shapes = [tuple(int(v) for v in hdu.data.shape) for hdu in hdus[1:]]
        if any(shape != (2048, 576) for shape in shapes):
            raise ValueError(f"Unexpected amplifier shapes: {sorted(set(shapes))}")
        detector_name = str(first_header_value(hdus, "DET_NAME", "CHIPID")).strip()
        instrument = str(first_header_value(hdus, "INSTRUME")).strip()
        if instrument != CAMERA_NAME:
            raise ValueError(f"Expected INSTRUME={CAMERA_NAME}, got {instrument!r}")
        identity = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "hdu_count": len(hdus),
            "amplifier_shapes_yx": [list(shape) for shape in shapes],
            "instrument": instrument,
            "detector_name": detector_name,
            "day_obs": int(first_header_value(hdus, "DAYOBS")),
            "seq_num": int(first_header_value(hdus, "SEQNUM")),
            "focus_mm": float(first_header_value(hdus, "FOCUSZ")),
            "physical_filter": str(first_header_value(hdus, "FILTER")).strip(),
            "group_id": str(first_header_value(hdus, "GROUPID")).strip(),
            "exposure_time_s": float(first_header_value(hdus, "EXPTIME")),
            "mjd_obs_tai": float(first_header_value(hdus, "MJD-OBS")),
            "boresight_alt_deg": float(first_header_value(hdus, "ELSTART")),
            "boresight_az_deg": float(first_header_value(hdus, "AZSTART")),
        }
    band = identity["physical_filter"].split("_", 1)[0]
    if band not in "ugrizy":
        raise ValueError(f"Cannot derive Rubin band from {identity['physical_filter']!r}")
    identity["band"] = band
    return identity


def run_matched_isr(path: Path, detector_id: int, identity: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Assemble amp raw with overscan, nominal gains and camera crosstalk."""

    from lsst.afw.image import FilterLabel
    from lsst.ip.isr import CrosstalkCalib, IsrTaskLSST, PhotonTransferCurveDataset
    from lsst.obs.lsst import LsstCamSim
    from lsst.obs.lsst.utils import readRawFile

    camera = LsstCamSim.getCamera()
    detector = camera[detector_id]
    if detector.getName() != identity["detector_name"]:
        raise ValueError(
            f"detector {detector_id} is {detector.getName()}, raw says {identity['detector_name']}"
        )

    data_id = {
        "instrument": CAMERA_NAME,
        "detector": detector_id,
        "day_obs": identity["day_obs"],
        "seq_num": identity["seq_num"],
    }
    raw_exposure = readRawFile(str(path), detector, dataId=data_id)

    config = IsrTaskLSST.ConfigClass()
    crosstalk_coefficients = np.asarray(detector.getCrosstalk(), dtype=float)
    has_crosstalk = crosstalk_coefficients.size == len(detector) ** 2
    if crosstalk_coefficients.size not in (0, len(detector) ** 2):
        raise RuntimeError(
            f"Unexpected camera crosstalk size {crosstalk_coefficients.size} for {len(detector)} amps"
        )
    settings = {
        "doApplyGains": True,
        "useGainsFrom": "PTC",
        "doCorrectGains": False,
        "doBias": False,
        "doDeferredCharge": False,
        "doLinearize": False,
        # imSim's CcdReadout also skips crosstalk when this camera-model
        # matrix is absent, so the inverse ISR setting must follow it.
        "doCrosstalk": has_crosstalk,
        "doDefect": False,
        "doDark": False,
        "doFlat": False,
        "doBrighterFatter": False,
        "doInterpolate": False,
        "doAmpOffset": False,
        "doVariance": True,
        "doSaturation": True,
        "defaultSaturationSource": "CAMERAMODEL",
        "doSuspect": False,
        "doSetBadRegions": False,
        "doBootstrap": False,
        "doStandardStatistics": False,
        "doCheckUnprocessableData": False,
        "doITLEdgeBleedMask": False,
        "doITLDipMask": False,
        "doE2VEdgeBleedMask": False,
    }
    applied: dict[str, Any] = {}
    for name, value in settings.items():
        if hasattr(config, name):
            setattr(config, name, value)
            applied[name] = value
    if hasattr(config, "qa") and hasattr(config.qa, "saveStats"):
        config.qa.saveStats = False
        applied["qa.saveStats"] = False

    # Use the same nominal camera gains and read noise as imSim.
    ptc = PhotonTransferCurveDataset(
        [amp.getName() for amp in detector], "NOMINAL_PTC", 1
    )
    for amp in detector:
        name = amp.getName()
        ptc.gain[name] = float(amp.getGain())
        ptc.noise[name] = float(amp.getReadNoise())
        ptc.ptcTurnoff[name] = math.inf
    crosstalk = CrosstalkCalib().fromDetector(detector) if has_crosstalk else None
    task = IsrTaskLSST(config=config)
    result = task.run(raw_exposure, ptc=ptc, crosstalk=crosstalk, camera=camera)
    exposure = result.outputExposure
    exposure.setFilter(FilterLabel(band=identity["band"], physical=identity["physical_filter"]))

    if exposure.getDetector() is None or exposure.getDetector().getId() != detector_id:
        raise RuntimeError("ISR output lost or changed detector geometry")
    if exposure.getWcs() is None:
        raise RuntimeError("ISR output has no WCS")
    expected = detector.getBBox().getDimensions()
    expected_shape = (int(expected.getY()), int(expected.getX()))
    if tuple(exposure.image.array.shape) != expected_shape:
        raise RuntimeError(
            f"ISR output shape {exposure.image.array.shape} != camera geometry {expected_shape}"
        )
    if not np.all(np.isfinite(exposure.image.array)):
        raise RuntimeError("ISR output image contains non-finite pixels")
    return exposure, {
        "task": "lsst.ip.isr.IsrTaskLSST",
        "profile": "imsim_readout_matched",
        "applied_config": applied,
        "units": str(exposure.metadata.get("LSST ISR UNITS", "unknown")),
        "camera_model_has_crosstalk": has_crosstalk,
        "shape_yx": list(exposure.image.array.shape),
        "median": float(np.median(exposure.image.array)),
        "standard_deviation": float(np.std(exposure.image.array)),
    }


def make_detection_task(source_limit: int) -> Any:
    from lsst.ts.wep.task.generateDonutDirectDetectTask import (
        GenerateDonutDirectDetectTask,
        GenerateDonutDirectDetectTaskConfig,
    )

    config = GenerateDonutDirectDetectTaskConfig()
    config.donutSelector.sourceLimit = source_limit
    config.initialCutoutPadding = 40
    return GenerateDonutDirectDetectTask(config=config)


def make_cutout_task(stamp_size: int, padding: int) -> Any:
    from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
        CutOutDonutsScienceSensorTask,
        CutOutDonutsScienceSensorTaskConfig,
    )

    config = CutOutDonutsScienceSensorTaskConfig()
    config.donutStampSize = stamp_size
    config.initialCutoutPadding = padding
    return CutOutDonutsScienceSensorTask(config=config)


def quantity_value(value: Any, unit: Any | None = None) -> float:
    if hasattr(value, "to_value"):
        return float(value.to_value(unit) if unit is not None else value.value)
    return float(value)


def make_metadata_table(
    catalog: QTable,
    stamps: Any,
    identity: dict[str, Any],
    detector_id: int,
    side: str,
) -> QTable:
    if len(catalog) != len(stamps):
        raise RuntimeError(f"catalog/stamp count mismatch: {len(catalog)} != {len(stamps)}")
    meta = stamps.metadata
    rows = []
    for index, (source, stamp) in enumerate(zip(catalog, stamps)):
        field_dvcs = stamp.calcFieldXY()
        field_ccs = np.asarray(stamp.wep_im.fieldAngle, dtype=float)
        xy0 = stamp.stamp_im.getXY0()
        rows.append(
            {
                "stamp_index": index,
                "donut_id": str(stamp.donut_id),
                "source_catalog_row": index,
                "detector_id": detector_id,
                "detector_name": identity["detector_name"],
                "camera": CAMERA_NAME,
                "side": side,
                "focus_mm": identity["focus_mm"],
                "band": identity["band"],
                "day_obs": identity["day_obs"],
                "seq_num": identity["seq_num"],
                "group_id": identity["group_id"],
                "donut_stamps_index": index,
                "stamp_height": int(stamp.stamp_im.image.array.shape[0]),
                "stamp_width": int(stamp.stamp_im.image.array.shape[1]),
                "bbox_x0": int(xy0.getX()),
                "bbox_y0": int(xy0.getY()),
                "detected_centroid_x": quantity_value(source["centroid_x"]),
                "detected_centroid_y": quantity_value(source["centroid_y"]),
                "final_centroid_x": float(stamp.centroid_position.getX()),
                "final_centroid_y": float(stamp.centroid_position.getY()),
                "ra_deg": float(stamp.sky_position.getRa().asDegrees()),
                "dec_deg": float(stamp.sky_position.getDec().asDegrees()),
                "field_dvcs_x_deg": float(field_dvcs[0]),
                "field_dvcs_y_deg": float(field_dvcs[1]),
                "field_ccs_x_deg": float(field_ccs[0]),
                "field_ccs_y_deg": float(field_ccs[1]),
                "source_flux_njy": quantity_value(source["source_flux"], u.nJy),
                "sn": float(meta.getArray("SN")[index]),
                "effective": int(meta.getArray("EFFECTIVE")[index]),
                "entropy": float(meta.getArray("ENTROPY")[index]),
                "recenter_flag": int(meta.getArray("RECENTER_FLAGS")[index]),
                "bad_pixel_fraction": float(meta.getArray("FRAC_BAD_PIX")[index]),
            }
        )
    names = [name for name, _ in METADATA_SCHEMA]
    dtypes = [dtype for _, dtype in METADATA_SCHEMA]
    return QTable(rows=rows, names=names, dtype=dtypes)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_stamps(amp: Path, state_index: int, side: str, output: Path | None = None,
                   source_limit: int = 40) -> dict:
    """Read amplifier raw -> ISR -> detect -> cut -> save stamps and metadata."""
    from lsst.obs.lsst import LsstCamSim
    from lsst.ts.wep.task.donutStamps import DonutStamps
    from lsst.ts.wep.utils import DefocalType

    amp = amp.expanduser().resolve()
    if state_index < 0 or source_limit <= 0:
        raise ValueError("state_index must be nonnegative and source_limit positive")
    if side not in ("intra", "extra"):
        raise ValueError("side must be intra or extra")
    identity = inspect_amp_raw(amp)
    detector = LsstCamSim.getCamera()[identity["detector_name"]]
    detector_id = int(detector.getId())
    if output is None:
        output = OUTPUT_ROOT / "stamps" / f"state_{state_index:03d}" / side / f"det{detector_id:03d}_{identity['detector_name']}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.expanduser().resolve()
    expected_seq = 2 * state_index + (1 if side == "intra" else 2)
    if identity["seq_num"] != expected_seq:
        raise ValueError(f"State/side expects SEQNUM={expected_seq}, got {identity['seq_num']}")
    if (side == "intra" and identity["focus_mm"] >= 0) or (
        side == "extra" and identity["focus_mm"] <= 0
    ):
        raise ValueError("Focal side and FOCUSZ sign disagree")

    exposure, isr = run_matched_isr(amp, detector_id, identity)
    catalog = make_detection_task(source_limit).run(
        exposure.clone(), LsstCamSim.getCamera()
    ).donutCatalog
    catalog["donut_id"] = np.asarray([
        f"{identity['day_obs']}-{identity['seq_num']:06d}-{detector_id:03d}-{i:03d}"
        for i in range(len(catalog))
    ], dtype=str)
    cutout = make_cutout_task(stamp_size=200, padding=40)
    defocal_type = DefocalType.Intra if side == "intra" else DefocalType.Extra
    cut_exposure = exposure.clone()
    stamps = cutout.cutOutStamps(cut_exposure, catalog, defocal_type, CAMERA_NAME)
    if not len(stamps):
        stamps = cutout.addVisitLevelMetadata(
            cut_exposure, DonutStamps([]), catalog, defocal_type
        )
        stamps.use_archive = False
    metadata = make_metadata_table(catalog, stamps, identity, detector_id, side)
    metadata["state_index"] = np.full(len(metadata), state_index, dtype=np.int64)
    for stamp in stamps:
        image = np.asarray(stamp.wep_im.image)
        if image.shape != (200, 200) or not np.isfinite(image).all():
            raise ValueError("WEP returned an invalid 200x200 CCS image")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        stamps.writeFits(str(directory / "donut_stamps.fits"))
        metadata.write(directory / "stamp_metadata.ecsv", format="ascii.ecsv")
        manifest = {
            "state_index": state_index, "detector_id": detector_id, "side": side,
            "input": identity,
            "counts": {"detected": len(catalog), "stamps": len(stamps)}, "isr": isr,
            "wep": {"source_limit": source_limit, "stamp_size": 200, "padding": 40,
                    "stored_frame": "DVCS", "training_view": "DonutStamp.wep_im.image",
                    "training_frame": "CCS"},
            "products": {name: {"bytes": (directory / name).stat().st_size,
                                "sha256": sha256(directory / name)}
                         for name in ("donut_stamps.fits", "stamp_metadata.ecsv")},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        directory.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--side", choices=("intra", "extra"), required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: output/stamps/state_NNN/SIDE/detNNN_NAME")
    parser.add_argument("--source-limit", type=int, default=40)
    args = parser.parse_args()
    result = extract_stamps(args.amp, args.state_index, args.side, args.output_dir, args.source_limit)
    print(f"state={result['state_index']} detector={result['detector_id']} "
          f"side={result['side']} stamps={result['counts']['stamps']}")


if __name__ == "__main__":
    main()
