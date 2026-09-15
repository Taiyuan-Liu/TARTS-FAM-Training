#!/usr/bin/env python3
"""Compare saved MIW block maps with each estimator's own reference."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from step04_analysis.config import ROOT, settings
sys.path.insert(0, str(ROOT / "external_repo/rubin-work/aos/code"))
from recompute_coadd_metrics import coarsen, coarsen_centers, block_metrics

TERMS = [5, 6, 7, 8]
COLORS = {"danish": "0.25", "tarts": "tab:orange"}


def metrics(grids, reference, edges):
    """Use Aaron's 3x3 rebin, spatial correlations and residual RMS."""
    centers = coarsen_centers((edges[1:] + edges[:-1]) / 2, 3)
    radius = np.hypot(centers[:, None], centers[None, :])
    return block_metrics(
        {j: coarsen(grids[k], 3) for k, j in enumerate(TERMS)},
        {j: coarsen(reference[k], 3) for k, j in enumerate(TERMS)},
        (radius >= 1.59) & (radius <= 1.725))


def collect(grids_dir, observations):
    records = []
    observing = pd.read_csv(observations).set_index("visit")
    for path in sorted((Path(grids_dir) / "blocks").glob("block_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if (str(data["frame"]), str(data["unit"])) != ("OCS", "micron"):
                raise ValueError("Expected OCS maps in microns")
            if not np.array_equal(data["terms"], TERMS):
                raise ValueError("Expected Z5--Z8 map order")
            times = observing.loc[data["visits"], "mjd"].to_numpy(float)
            for branch in ("danish", "tarts"):
                records.append(dict(ordinal=int(data["ordinal"]), day_obs=int(data["day_obs"]),
                    build_used=bool(data["build_used"]), branch=branch,
                    visits=len(data["visits"]), pairs=int(data[branch + "_pairs"]),
                    first_mjd=times.min(), last_mjd=times.max(), mean_mjd=times.mean(),
                    **metrics(data[branch], data[branch + "_reference"], data["edges"])))
    if not records:
        raise FileNotFoundError("No block maps")
    table = pd.DataFrame(records).sort_values(["day_obs", "mean_mjd", "ordinal", "branch"])
    order = table.drop_duplicates("ordinal").ordinal.tolist()
    table["block_index"] = table.ordinal.map({value: i for i, value in enumerate(order)})
    return table


def legends(label):
    return [Line2D([], [], color=COLORS[name], marker="o", ms=4, lw=1.5, label=text)
            for name, text in (("danish", "Danish"), ("tarts", label))] + [
            Patch(facecolor="tab:blue", alpha=.10, label="Reference-build blocks")]


def overview(table, output, label):
    fig, axis = plt.subplots(figsize=(16, 4.8), layout="constrained")
    for index in table.loc[table.build_used, "block_index"].unique():
        axis.axvspan(index - .5, index + .5, color="tab:blue", alpha=.10, linewidth=0)
    for branch, group in table.groupby("branch"):
        group = group.sort_values("block_index")
        axis.plot(group.block_index, group.corr_combined, "o-", color=COLORS[branch], lw=1.5, ms=3.4)
    lower = min(0.0, float(table.corr_combined.min()) - .04)
    axis.set(xlim=(-.6, table.block_index.max() + .6), ylim=(lower, 1.025),
             xlabel="Chronological i-band block index", ylabel="Mean spatial correlation r",
             title="Self-reference comparison · Z5–Z8 · OCS")
    axis.grid(axis="y", alpha=.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(handles=legends(label), loc="lower right", ncol=3, frameon=False)
    fig.savefig(output / "tarts_vs_danish_self_mean.png", dpi=180)
    fig.savefig(output / "tarts_vs_danish_self_mean.pdf")
    plt.close(fig)


def dated_layout(table):
    """Separate the reference month; preserve intra-night MJD spacing."""
    blocks = table.drop_duplicates("ordinal").copy()
    dates = pd.to_datetime(blocks.day_obs.astype(str), format="%Y%m%d")
    reference_dates = dates[blocks.build_used]
    if len(reference_dates):
        start = reference_dates.min().to_period("M").start_time
        end = start + pd.offsets.MonthBegin(1)
        periods = [("Before " + start.strftime("%B %Y"), dates < start),
                   (start.strftime("%B %Y"), (dates >= start) & (dates < end)),
                   ("From " + end.strftime("%B %Y"), dates >= end)]
    else:
        periods = [("Observing nights", np.ones(len(blocks), bool))]
    nights, previous_day, panel = [], None, 0
    for title, selected in periods:
        if not np.any(selected):
            continue
        cursor = 0.0
        for day, group in blocks[selected].groupby("day_obs", sort=True):
            first, last = group.first_mjd.min(), group.last_mjd.max()
            span = (last - first) * 24
            width = max(2.1, span)
            positions = cursor + (width - span) / 2 + (group.mean_mjd - first) * 24
            blocks.loc[group.index, "display_x"] = positions
            blocks.loc[group.index, "panel"] = panel
            date = pd.to_datetime(str(day), format="%Y%m%d")
            nights.append(dict(panel=panel, title=title, day_obs=int(day), left=cursor,
                right=cursor + width, label_x=cursor + width / 2,
                date_label=date.strftime("%m-%d"), reference=bool(group.build_used.all()),
                days_from_previous=(date - previous_day).days if previous_day is not None else 0))
            previous_day = date
            cursor += width + 1.45
        panel += 1
    return blocks, pd.DataFrame(nights)


def index_labels(axis, blocks):
    renderer = axis.figure.canvas.get_renderer()
    row_ends = []
    for row in blocks.sort_values("display_x").itertuples():
        text = axis.text(row.display_x, 1.035, str(row.block_index),
                         transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                         fontsize=8, color="0.3", clip_on=False)
        bounds = text.get_window_extent(renderer)
        level = next((i for i, end in enumerate(row_ends) if bounds.x0 > end + 3), len(row_ends))
        if level == len(row_ends):
            row_ends.append(bounds.x1)
        else:
            row_ends[level] = bounds.x1
        y = 1.035 + level * .075
        text.set_y(y)
        axis.plot([row.display_x, row.display_x], [1.003, y - .009],
                  transform=axis.get_xaxis_transform(), color="0.65", lw=.5, clip_on=False)
    axis.text(-.012, 1.035, "Index", transform=axis.transAxes, ha="right", fontsize=8, color="0.3")


def dated(table, output, label):
    blocks, nights = dated_layout(table)
    panels = nights.panel.nunique()
    fig, axes = plt.subplots(panels, 1, figsize=(17, max(4.5, 3.73 * panels)), squeeze=False)
    fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.10, hspace=.90)
    fig.suptitle("Danish / TARTS · self-reference · Z5–Z8 · OCS", y=.985, fontsize=15)
    fig.legend(handles=legends(label), ncol=3, loc="upper center", bbox_to_anchor=(.5, .95), frameon=False)
    joined = table.merge(blocks[["ordinal", "display_x", "panel"]], on="ordinal", validate="many_to_one")
    lower = min(0., float(table.corr_combined.min()) - .04)
    for panel, axis in enumerate(axes.flat):
        selected = nights[nights.panel == panel]
        for k, night in enumerate(selected.itertuples()):
            if night.reference or k % 2 == 0:
                axis.axvspan(night.left, night.right, color="tab:blue" if night.reference else "black",
                             alpha=.10 if night.reference else .022, lw=0)
            if night.days_from_previous >= 30:
                axis.text(night.left, 1.30, f"+{night.days_from_previous} d",
                          transform=axis.get_xaxis_transform(), fontsize=8, color="0.45")
        axis.set(xlim=(selected.left.min() - .7, selected.right.max() + .7),
                 ylim=(lower, 1.025), ylabel="Mean spatial r")
        axis.set_xticks(selected.label_x, selected.date_label, fontsize=9, rotation=35, ha="right")
        axis.set_title(selected.title.iloc[0], loc="left", fontsize=11, pad=46)
        axis.tick_params(axis="x", length=0, pad=5)
        axis.grid(axis="y", alpha=.19)
        axis.spines[["top", "right", "bottom"]].set_visible(False)
        for (branch, day), group in joined[joined.panel == panel].groupby(["branch", "day_obs"]):
            group = group.sort_values("display_x")
            axis.plot(group.display_x, group.corr_combined, "o-", color=COLORS[branch], lw=1.6, ms=4.1)
    fig.canvas.draw()
    for panel, axis in enumerate(axes.flat):
        index_labels(axis, blocks[blocks.panel == panel])
    fig.text(.5, .02, "Dates: observing nights. Within-night spacing follows MJD; between-night gaps are compressed.",
             ha="center", fontsize=9, color="0.4")
    fig.savefig(output / "tarts_vs_danish_self_mean_dated.png", dpi=180)
    fig.savefig(output / "tarts_vs_danish_self_mean_dated.pdf")
    plt.close(fig)
    blocks.to_csv(output / "block_dates.csv", index=False)


def compare(table, output, label="TARTS"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "time_metrics.csv", index=False)
    table.loc[~table.build_used].groupby("branch").agg(
        blocks=("ordinal", "count"), mean_correlation=("corr_combined", "mean"),
        mean_residual_rms_um=("resid_rms_combined", "mean")).to_csv(output / "nonreference_summary.csv")
    overview(table, output, label)
    dated(table, output, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--grids-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = settings(args.config)
    root = args.grids_dir or cfg["output_root"] / "miw_analysis" / args.variant / "time_consistency"
    compare(collect(root, cfg["miw"]["observations_file"]), args.output_dir or root / "figures",
            "TARTS DARE" if args.variant == "dare" else "TARTS")


if __name__ == "__main__":
    main()
