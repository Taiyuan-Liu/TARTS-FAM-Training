#!/usr/bin/env python3
"""Build MIW reference and time-block maps; save arrays without correlation analysis."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.table import QTable
from step04_analysis.config import NOLL, settings

TERMS = [5, 6, 7, 8]


def read_pairs(path, visits):
    filters = [[("day_obs", "=", int(v) // 100000), ("seq_num", "=", int(v) % 100000)] for v in visits]
    return pq.read_table(path, filters=filters).to_pandas()


def fit_grid(frame, basis):
    from lsst.ts.intrinsic.wavefront.measured_intrinsic import build_measured_intrinsic_uconstrained
    result = build_measured_intrinsic_uconstrained(frame, None, "OCS", NOLL,
        kj_grid=basis.kj_grid, U_eff=basis.U_eff, n_iter=3, n_bins=73,
        fp_radius_basis=1.8, fp_radius_grid=1.8, min_donuts=100, bad_fit_threshold=2.0)
    grid = result["iter_results"][-1]["measured_grid"]
    return np.stack([grid[j] for j in TERMS]), result["xbins"]


def polar_reference(grids, edges, thetas):
    """Five-rotation, spin-aware OCS-only reference for Z5--Z8."""
    from lsst.ts.intrinsic.wavefront import intrinsic_split as isp
    radius, azimuth, x, y = isp.make_polar_grid(r_min=.06, r_max=1.75, n_r=80, n_az=180)
    centers = (edges[1:] + edges[:-1]) / 2
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    samples = {}
    for k, j in enumerate(TERMS):
        maps = []
        for grid in grids:
            mask = np.isfinite(grid[k])
            maps.append((xx[mask], yy[mask], grid[k][mask]))
        values, valid, _ = isp.sample_maps_polar(maps, x, y, hole_dist=.06)
        samples[j] = np.nan_to_num(values), valid
    fields = {}
    for group in isp.group_zernikes(TERMS):
        jc, js = group["j_cos"], group["j_sin"]
        zc, vc = samples[jc]
        zs, vs = samples[js]
        split = isp.decompose_spin_lsq(zc + 1j * zs, vc & vs, np.deg2rad(thetas), azimuth,
            n_spin=group["spin"], s=1, m_max=12, ridge=1e-3, degen_assignment="ocs", ocs_only=True)
        fields[jc], fields[js] = split["O_pol"].real, split["O_pol"].imag
    return dict(fields=np.stack([fields[j] for j in TERMS]), X=x, Y=y, R=radius,
                A=azimuth, terms=np.asarray(TERMS), thetas=np.asarray(thetas))


def sample_reference(frame, reference):
    """Evaluate at actual pair positions, then median-bin onto the block grid."""
    from scipy.spatial import Delaunay
    from lsst.ts.intrinsic.wavefront.measured_intrinsic import bin_median_focal
    tri = Delaunay(np.column_stack([reference["X"].ravel(), reference["Y"].ravel()]))
    points = np.column_stack([np.rad2deg(frame.thx_OCS), np.rad2deg(frame.thy_OCS)])
    simplex = tri.find_simplex(points)
    transform = tri.transform[simplex]
    bary = np.einsum("nij,nj->ni", transform[:, :2, :], points - transform[:, 2, :])
    weights = np.column_stack([bary, 1 - bary.sum(axis=1)])
    vertices = tri.simplices[simplex]
    values = np.zeros((len(frame), len(NOLL)))
    for k, j in enumerate(TERMS):
        values[:, NOLL.index(j)] = np.where(simplex >= 0,
            (reference["fields"][k].ravel()[vertices] * weights).sum(axis=1), np.nan)
    grids, *_ = bin_median_focal(points[:, 0], points[:, 1], values,
                                 {j: k for k, j in enumerate(NOLL)}, n_bins=73, fp_radius=1.8)
    return np.stack([grids[j] for j in TERMS])


def build_reference(root, output, membership, basis):
    output.mkdir(parents=True, exist_ok=True)
    metadata = QTable.read(root / "danish/visits.parquet")
    angles = dict(zip(np.asarray(metadata["visit"], int), np.asarray(metadata["rotator_angle"], float)))
    thetas = [np.mean([angles[v] for v in group["visits"]]) for group in membership["bins"]]
    for branch in ("danish", "tarts"):
        grids = []
        for group in membership["bins"]:
            frame = read_pairs(root / branch / "donuts.parquet", group["visits"])
            grid, edges = fit_grid(frame, basis)
            grids.append(grid)
        np.savez_compressed(output / f"{branch}_polar.npz", **polar_reference(grids, edges, thetas))

def build_block(root, output, row, basis):
    arrays = {}
    for branch in ("danish", "tarts"):
        frame = read_pairs(root / branch / "donuts.parquet", row["visits"])
        grids, edges = fit_grid(frame, basis)
        with np.load(output / "reference" / f"{branch}_polar.npz") as reference:
            arrays[branch + "_reference"] = sample_reference(frame, reference)
        arrays[branch] = grids
        arrays[branch + "_pairs"] = len(frame)
    folder = output / "blocks"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"block_{row['ordinal']:03d}.npz",
        **arrays, edges=edges, terms=TERMS, frame="OCS", unit="micron",
        ordinal=row["ordinal"], day_obs=row["day_obs"], build_used=row["build_used"],
        visits=np.asarray(row["visits"], np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reference", "blocks"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--block", type=int, action="append")
    args = parser.parse_args()
    cfg = settings(args.config)
    from lsst.ts.intrinsic.wavefront import ofc_svd
    basis = ofc_svd.build_ofc_svd(NOLL, 1, 6, 34, n_dof=50,
                                ofc_normalization_yaml=cfg["miw"]["normalization_file"])
    root = cfg["output_root"] / "miw_analysis" / args.variant
    source = args.input_dir or root / "inputs"
    output = args.output_dir or root / "time_consistency"
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "reference":
        build_reference(source, output / "reference",
                        json.loads(cfg["miw"]["reference_file"].read_text()), basis)
    else:
        rows = json.loads(cfg["miw"]["time_blocks_file"].read_text())
        selected = args.block if args.block is not None else [r["ordinal"] for r in rows]
        by_id = {r["ordinal"]: r for r in rows}
        for index in selected:
            build_block(source, output, by_id[index], basis)
            print(f"block {index}: maps written", flush=True)


if __name__ == "__main__":
    main()
