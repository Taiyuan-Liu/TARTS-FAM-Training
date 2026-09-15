#!/usr/bin/env python3
"""Compare independent CCD predictions with Danish CCD averages in physical CCS."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step04_analysis.config import MODE_INDEX, NOLL, settings


def mode_metrics(prediction, reference):
    """Equal weight per row; sample standard deviation, not standard error."""
    records = []
    for k, j in enumerate(NOLL):
        x, y = reference[:, k], prediction[:, k]
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        error = y - x
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else None
        records.append(dict(noll=j, count=len(x), bias_um=float(error.mean()) if len(x) else None,
                            rmse_um=float(np.sqrt(np.mean(error**2))) if len(x) else None,
                            std_um=float(error.std(ddof=1)) if len(x) > 1 else None, correlation=corr))
    return records


def mode_title(j):
    """Name and signed pupil symmetry of a Noll-indexed Zernike mode."""
    from galsim.zernike import noll_to_zern
    names = {(2, 0): "Defocus", (2, 2): "Astigmatism", (3, 1): "Coma",
             (3, 3): "Trefoil", (4, 0): "Primary spherical",
             (4, 2): "Secondary astigmatism", (4, 4): "Tetrafoil",
             (5, 1): "Secondary coma", (5, 3): "Secondary trefoil",
             (6, 0): "Secondary spherical", (6, 2): "Tertiary astigmatism",
             (6, 4): "Secondary tetrafoil"}
    n, m = noll_to_zern(j)
    if m == 0:
        symmetry = "m=0 · Axisymmetric"
    else:
        order = abs(m)
        angular = ("cos" if m > 0 else "sin") + f" {order if order > 1 else ''}θ"
        rotation = "180° sign flip" if order == 1 else f"{360 // order}° symmetry"
        symmetry = f"{angular} · {rotation}"
    return f"Z{j} · {names[n, abs(m)]}\n{symmetry}"


def prediction_table(directory):
    records = []
    files = sorted(Path(directory).glob("visit_*.npz"))
    if not files:
        raise FileNotFoundError(f"No visit predictions in {directory}")
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if (str(data["frame"]), str(data["unit"])) != ("CCS", "micron"):
                raise ValueError("Expected CCS predictions in microns")
            values = data["prediction_ccs_um"][:, MODE_INDEX]
            for k, detector in enumerate(data["detector_id"]):
                records.append(dict(visit=int(data["visit"]), detector_id=int(detector),
                                    **{f"Z{j}": float(values[k, i]) for i, j in enumerate(NOLL)}))
    table = pd.DataFrame(records).set_index(["visit", "detector_id"])
    if not table.index.is_unique:
        raise ValueError("Duplicate visit/CCD predictions")
    return table.sort_index()


def danish_table(packages, keys):
    """Read references only here; never average broadcast pair predictions."""
    records = []
    for visit in sorted(set(keys.get_level_values("visit"))):
        frame = pd.read_pickle(Path(packages) / f"visit_{visit}.pkl")[["visit", "detector_id", "zk_avg"]]
        if set(frame.visit) != {visit}:
            raise ValueError("Packaged reference visit mismatch")
        for detector, group in frame.groupby("detector_id"):
            if (visit, detector) not in keys:
                continue
            values = np.stack(group.zk_avg).astype(float)
            if values.shape[1] != len(NOLL) or not np.allclose(values, values[0], rtol=0, atol=5e-7):
                raise ValueError("Danish zk_avg must be one 21-mode vector per visit/CCD")
            records.append(dict(visit=visit, detector_id=int(detector),
                                **{f"Z{j}": values[0, i] for i, j in enumerate(NOLL)}))
    return pd.DataFrame(records).set_index(["visit", "detector_id"]).sort_index().loc[keys]


def parity(prediction, reference, bounds, path):
    stats = mode_metrics(prediction, reference)
    fig, axes = plt.subplots(3, 7, figsize=(21, 10.5), layout="constrained")
    for k, axis in enumerate(axes.flat):
        lo, hi = bounds[k]
        axis.scatter(reference[:, k], prediction[:, k], s=2, alpha=0.18, rasterized=True)
        axis.plot([lo, hi], [lo, hi], color="0.4", lw=0.7)
        r = stats[k]["correlation"]
        axis.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        axis.set_title(mode_title(NOLL[k]), fontsize=9.5)
        corr = f"{r:.3f}" if r is not None else "n/a"
        axis.text(.03, .96, f"n={stats[k]['count']:,}  r={corr}\n"
                  f"RMSE={stats[k]['rmse_um']:.3f} µm\n"
                  f"bias={stats[k]['bias_um']:+.3f} µm", transform=axis.transAxes,
                  va="top", fontsize=8)
        axis.tick_params(labelsize=7)
    fig.supxlabel("Danish [µm, CCS]"); fig.supylabel("TARTS [µm, CCS]")
    fig.savefig(Path(path).with_suffix(".png"), dpi=180)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def residual_plots(statistics, out):
    for errorbars in (True, False):
        fig, axis = plt.subplots(figsize=(12, 4.5), layout="constrained")
        x = np.arange(len(NOLL))
        for index, (name, rows) in enumerate(statistics.items()):
            mean = [row["bias_um"] for row in rows]
            label, color = ("Before DARE", "0.25") if name == "supervised" else ("After DARE", "#cf4559")
            if errorbars:
                axis.errorbar(x + (index - (len(statistics) - 1) / 2) * .15, mean,
                              yerr=[row["std_um"] for row in rows], fmt="o", ms=3,
                              capsize=2, lw=.8, label=label, color=color)
            else:
                axis.plot(x, mean, "o-", ms=3, lw=1, label=label, color=color)
        axis.axhline(0, color="0.6", lw=.8)
        axis.set(xticks=x, xticklabels=NOLL, xlabel="Noll index", ylabel="TARTS − Danish [µm, CCS]")
        axis.legend(frameon=False)
        path = out / ("residual_errorbars" if errorbars else "residual_lines")
        fig.savefig(path.with_suffix(".png"), dpi=180)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), action="append")
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--zoom-quantile", type=float, default=.002)
    args = parser.parse_args()
    if not 0 <= args.zoom_quantile < .5:
        parser.error("zoom-quantile must be in [0, .5)")
    cfg = settings(args.config)
    root = args.predictions_root or cfg["output_root"] / "predictions"
    tables = {name: prediction_table(root / name) for name in args.variant or ["supervised", "dare"]}
    keys = next(iter(tables.values())).index
    for table in tables.values():
        keys = keys.intersection(table.index)
    keys = keys.sort_values()
    if len(keys) == 0:
        raise ValueError("No common visit/CCD predictions")
    reference = danish_table(cfg["packages_dir"], keys)
    pred = {name: table.loc[keys].to_numpy(float) for name, table in tables.items()}
    target = reference.to_numpy(float)
    finite = np.isfinite(target).all(axis=1)
    for values in pred.values():
        finite &= np.isfinite(values).all(axis=1)
    keys, target = keys[finite], target[finite]
    pred = {name: values[finite] for name, values in pred.items()}
    if not len(keys):
        raise ValueError("No finite common comparisons")
    out = args.output_dir or cfg["output_root"] / "danish_comparison" / "zernikes"
    out.mkdir(parents=True, exist_ok=False)
    bounds = []
    for k in range(len(NOLL)):
        values = np.concatenate([target[:, k], *[p[:, k] for p in pred.values()]])
        lo, hi = np.quantile(values, [args.zoom_quantile, 1 - args.zoom_quantile])
        pad = max((hi - lo) * .07, 1e-4)
        bounds.append((float(lo - pad), float(hi + pad)))
    stats = {}
    joined = keys.to_frame(index=False)
    for k, j in enumerate(NOLL):
        joined[f"danish_Z{j}"] = target[:, k]
    for name, values in pred.items():
        stats[name] = mode_metrics(values, target)
        pd.DataFrame(stats[name]).to_csv(out / f"{name}_metrics.csv", index=False)
        parity(values, target, bounds, out / f"{name}_vs_danish")
        for k, j in enumerate(NOLL):
            joined[f"{name}_Z{j}"] = values[:, k]
    joined.to_parquet(out / "ccd_comparisons.parquet", index=False)
    residual_plots(stats, out)
    outside = {name: [int((~((target[:, k] >= lo) & (target[:, k] <= hi)
                            & (values[:, k] >= lo) & (values[:, k] <= hi))).sum())
                      for k, (lo, hi) in enumerate(bounds)] for name, values in pred.items()}
    summary = dict(frame="CCS", unit="micron", noll=NOLL,
              sample="one visit/CCD", samples=len(keys), available={k: len(v) for k, v in tables.items()},
              visits=len(set(keys.get_level_values("visit"))), errorbar="sample standard deviation",
              plot_limits=bounds, points_outside_axes=outside,
              coefficient_rmse_um={k: float(np.sqrt(np.mean((v-target)**2))) for k, v in pred.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
