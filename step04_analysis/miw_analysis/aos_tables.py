"""Build standard AOS tables from TARTS CCD predictions.

Clipping, float32 nm round-trip, averaging and rotation follow the existing
MIW AIDonut/TARTS visit-table writer. No neural network is loaded here.
"""

from copy import deepcopy
import galsim
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table
from lsst.ts.wep.utils import conditionalSigmaClip
from step04_analysis.config import NOLL, MODE_INDEX

NOLL_INDICES = np.asarray(NOLL)
ZK_CATEGORIES = ("zk", "zk_intrinsic", "zk_deviation")
SIGMA_CLIP_KWARGS = {"sigma": 3.0, "maxiters": 1, "stdfunc": "mad_std"}
STD_MIN_NM = 0.005
MAX_ZERN_CLIP = 3


def _to_qtable_um(values_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round model float32 microns through CalcZernikes' float32 nm columns."""

    values_nm = (np.asarray(values_um, dtype=np.float32) * np.float32(1000.0)).astype(np.float32)
    roundtrip_um = values_nm * np.float32(0.001)
    return values_nm, roundtrip_um


def _clip_detector(
    deviation_nm: np.ndarray,
    fwhm: np.ndarray,
    fit_success: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce CombineZernikesSigmaClipTask followed by blurClip."""

    n_rows = len(deviation_nm)
    used = np.zeros(n_rows, dtype=bool)
    blur_clipped = np.zeros(n_rows, dtype=bool)
    valid = np.asarray(fit_success, dtype=bool) & np.isfinite(deviation_nm).all(axis=1)
    clipped_input = np.asarray(deviation_nm, dtype=np.float32).copy()
    clipped_input[~valid] = np.nan
    clipped = conditionalSigmaClip(
        clipped_input,
        sigmaClipKwargs=SIGMA_CLIP_KWARGS,
        stdMin=STD_MIN_NM,
    )

    num_rejected = len(clipped)
    effective_max = MAX_ZERN_CLIP + 1
    while num_rejected == len(clipped) and effective_max > 1:
        effective_max -= 1
        rejected = np.any(np.isnan(clipped[:, :effective_max]), axis=1)
        num_rejected = int(rejected.sum())
    used[~rejected] = True
    used &= valid

    use_indices = np.flatnonzero(used)
    if len(use_indices):
        blur_result = sigma_clip(
            np.asarray(fwhm, dtype=np.float32)[use_indices],
            stdfunc="mad_std",
            sigma_lower=99,
        )
        local_blur_mask = np.ma.getmaskarray(blur_result)
        drop_indices = use_indices[local_blur_mask]
        used[drop_indices] = False
        blur_clipped[drop_indices] = True

    return used, blur_clipped


def _rotate_table(table: Table) -> None:
    noll_indices = np.asarray(table.meta["nollIndices"], dtype=np.int64)
    if not np.array_equal(noll_indices, NOLL_INDICES):
        raise ValueError(f"Unexpected Noll indices: {noll_indices.tolist()}")

    jmin = int(noll_indices.min())
    jmax = int(noll_indices.max())
    rot_ocs = galsim.zernike.zernikeRotMatrix(jmax, -float(table.meta["rotTelPos"]))[4:, 4:]
    rot_nw = galsim.zernike.zernikeRotMatrix(jmax, -float(table.meta["parallacticAngle"]))[4:, 4:]

    for category in ZK_CATEGORIES:
        full_ccs = np.zeros((len(table), jmax - jmin + 1))
        full_ccs[:, noll_indices - 4] = np.asarray(table[f"{category}_CCS"])
        table[f"{category}_OCS"] = (full_ccs @ rot_ocs)[:, noll_indices - 4]
        table[f"{category}_NW"] = (full_ccs @ rot_nw)[:, noll_indices - 4]


def broadcast_prediction(source_raw, prediction):
    """Align by pair identities, then broadcast each CCD's 25-mode estimate."""
    import pandas as pd
    if (str(prediction["frame"]), str(prediction["unit"])) != ("CCS", "micron"):
        raise ValueError("Expected physical CCS predictions in microns")
    if int(prediction["visit"]) != int(source_raw.meta["visit"]):
        raise ValueError("Source and prediction visit differ")
    if not np.array_equal(prediction["noll"], np.arange(4, 29)):
        raise ValueError("Expected Z4--Z28 in Noll order")
    names = dict(zip(prediction["detector_id"], prediction["detector_name"]))
    pair_keys = pd.MultiIndex.from_arrays([
        [str(names[k]) for k in prediction["pair_detector_id"]],
        np.asarray(prediction["intra_donut_id"]).astype(str),
        np.asarray(prediction["extra_donut_id"]).astype(str)])
    source_keys = pd.MultiIndex.from_arrays([np.asarray(source_raw[k]).astype(str)
                    for k in ("detector", "intra_donut_id", "extra_donut_id")])
    if not pair_keys.is_unique or not source_keys.is_unique:
        raise ValueError("Duplicate input pair identities")
    order = pair_keys.get_indexer(source_keys)
    if (order < 0).any() or len(pair_keys) != len(source_keys):
        raise ValueError("Predictions must cover exactly the source AOS pairs")
    lookup = pd.DataFrame(prediction["prediction_ccs_um"][:, MODE_INDEX],
                          index=np.asarray(prediction["detector_name"]).astype(str))
    zernikes = lookup.loc[np.asarray(source_raw["detector"]).astype(str)].to_numpy(np.float32)
    fwhm = np.asarray(prediction["pair_fwhm"], np.float32)[order]
    if not np.isfinite(zernikes).all() or not np.isfinite(fwhm).all():
        raise ValueError("Non-finite predictions or packaged FWHM")
    return zernikes, fwhm


def build_visit_tables(source_raw, source_avg, zernikes_um, fwhm, selection="native"):
    """Return standard Raw/Avg AOS tables, with native clipping by default."""
    raw, avg = source_raw.copy(copy_data=True), source_avg.copy(copy_data=True)
    raw.meta, avg.meta = deepcopy(source_raw.meta), deepcopy(source_avg.meta)
    zernikes_um, fwhm = np.asarray(zernikes_um, np.float32), np.asarray(fwhm, np.float32)
    opd_nm, opd_um = _to_qtable_um(zernikes_um)
    intrinsic_um = np.asarray(source_raw["zk_intrinsic_CCS"], np.float32)
    intrinsic_nm = (intrinsic_um * np.float32(1000)).astype(np.float32)
    deviation_nm = ((zernikes_um.astype(float) - intrinsic_um.astype(float)) * 1000).astype(np.float32)
    raw["zk_CCS"] = opd_um
    raw["zk_intrinsic_CCS"] = intrinsic_um
    raw["zk_deviation_CCS"] = deviation_nm * np.float32(.001)
    used, blur_clipped = np.zeros(len(raw), bool), np.zeros(len(raw), bool)
    detectors = np.asarray(raw["detector"]).astype(str)
    for detector in np.unique(detectors):
        indices = np.flatnonzero(detectors == detector)
        used[indices], blur_clipped[indices] = _clip_detector(
            deviation_nm[indices], fwhm[indices], np.ones(len(indices), bool))
    if selection == "matched_danish":
        used = np.asarray(source_raw["used"], bool).copy()
        blur_clipped[:] = False
    elif selection != "native":
        raise ValueError(selection)
    raw["used"] = used
    for category in ZK_CATEGORIES:
        avg[f"{category}_CCS"] = np.full((len(avg), len(NOLL)), np.nan, np.float32)
    avg_detectors = np.asarray(avg["detector"]).astype(str)
    for detector in np.unique(detectors):
        rows = np.flatnonzero((detectors == detector) & used)
        target = np.flatnonzero(avg_detectors == detector)
        if len(target) != 1:
            raise ValueError(f"Expected one average row for {detector}")
        if not len(rows):
            continue
        for column, values in (("zk_CCS", opd_nm), ("zk_intrinsic_CCS", intrinsic_nm),
                               ("zk_deviation_CCS", deviation_nm)):
            for mode in range(len(NOLL)):
                avg[column][target[0], mode] = np.float32(np.nanmean(values[rows, mode])) * np.float32(.001)
    _rotate_table(raw)
    _rotate_table(avg)
    raw.meta["estimatorInfo"] = dict(fwhm=fwhm.tolist(), fit_success=[True] * len(raw),
                                     blur_clipped=blur_clipped.tolist())
    avg.meta["estimatorInfo"] = dict(fwhm=float(np.nanmedian(fwhm.astype(float))))
    raw.meta["estimator"], avg.meta["estimator"] = "TARTS", "TARTS"
    raw.meta["selection"], avg.meta["selection"] = selection, selection
    return raw, avg
