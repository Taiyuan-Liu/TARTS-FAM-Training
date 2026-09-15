#!/usr/bin/env python3
"""Compare Danish/TARTS native polar MIW maps in the presentation layout."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from astropy.table import Table
import numpy as np
import pandas as pd

from step04_analysis.config import NOLL, settings
from step04_analysis.danish_comparison.compare_zernikes import mode_metrics


def read_native(directory, frame="OCS", radial_range=(0.1, 1.6)):
    """Read the author's polar O/C fields and O-telescope RMS directly."""
    directory = Path(directory)
    decomp = Table.read(directory / "intrinsic_split_decomp.parquet")
    shape = (int(decomp.meta["n_r"]), int(decomp.meta["n_az"]))
    x = np.asarray(decomp.meta["X"]).reshape(shape)
    y = np.asarray(decomp.meta["Y"]).reshape(shape)
    radius = np.hypot(x[:, 0], y[:, 0])
    keep = (radius >= radial_range[0]) & (radius <= radial_range[1])
    component = "O" if frame == "OCS" else "C"
    fields = {int(row["j"]): np.asarray(row[f"{component}_{'re' if int(row['part']) == 0 else 'im'}"])
              .reshape(shape) for row in decomp}
    if frame == "OCS":
        rms = {int(row["j"]): float(row["O_tel"]) for row in
               Table.read(directory / "intrinsic_split_rms.parquet")}
    else:
        # C_cam in the author's diagnostic table can describe a separate full split.
        rms = {j: float(np.sqrt(np.nanmean(field[keep] ** 2))) for j, field in fields.items()}
    return dict(x=x, y=y, fields=fields, rms=rms, keep=keep)


def color_limit(j, branches):
    values = np.concatenate([branch["fields"][j][branch["keep"]].ravel() for branch in branches])
    value = max(float(np.nanpercentile(np.abs(values), 98)), 1e-8)
    step = 10.0 ** (np.floor(np.log10(value)) - 1)
    return float(np.ceil(value / step) * step)


def render(danish, tarts, output, frame="OCS", label="TARTS"):
    """One shared signed colorbar per mode, native polar fields and RMS."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (np.array_equal(danish["x"], tarts["x"]) and np.array_equal(danish["y"], tarts["y"])):
        raise ValueError("The two decompositions must use the same polar grid")
    if not np.array_equal(danish["keep"], tarts["keep"]):
        raise ValueError("The two maps must use the same radial window")
    arrays = [np.column_stack([branch["fields"][j][branch["keep"]].ravel() for j in NOLL])
              for branch in (danish, tarts)]
    records = mode_metrics(arrays[1], arrays[0])
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    left, right, top, bottom = .060, .989, .987, .008
    band_height = (top - bottom) / 3
    header_height, rms_height, band_gap, pair_gap = .028, .022, .017, .005
    map_height = (band_height - header_height - 2 * rms_height - band_gap - pair_gap) / 2
    map_width = map_height * 7.5 / 16
    column_width = (right - left) / 7
    triangulations = {}
    for index, j in enumerate(NOLL):
        band, col = divmod(index, 7)
        center = left + (col + .5) * column_width - .018
        band_top = top - band * band_height
        limit = color_limit(j, (danish, tarts))
        fig.text(center, band_top - header_height / 2, f"Z{j} · {frame} [µm]",
                 ha="center", va="center", fontsize=9.2)
        records[index].update(danish_polar_rms_um=danish["rms"][j],
                              tarts_polar_rms_um=tarts["rms"][j], color_limit_um=limit)
        for offset, (name, branch) in enumerate((("Danish", danish), (label, tarts))):
            map_top = band_top - header_height - offset * (map_height + rms_height + pair_gap)
            map_bottom = map_top - map_height
            axis = fig.add_axes([center - map_width / 2, map_bottom, map_width, map_height])
            values = branch["fields"][j].ravel()
            finite = np.isfinite(values)
            key = finite.tobytes()
            if key not in triangulations:
                triangulations[key] = Triangulation(branch["x"].ravel()[finite], branch["y"].ravel()[finite])
            axis.set_rasterization_zorder(0)
            artist = axis.tricontourf(triangulations[key], values[finite],
                levels=np.linspace(-limit, limit, 21), cmap="RdBu_r", extend="both", zorder=-1)
            axis.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), aspect="equal")
            axis.set_axis_off()
            fig.text(center, map_bottom - rms_height / 2, f"RMS={branch['rms'][j]:.4f}",
                     ha="center", va="center", fontsize=8.8)
            if col == 0:
                fig.text(.021, map_bottom + map_height / 2, name, rotation=90,
                         ha="center", va="center", fontsize=11.5, weight="bold")
        cax = fig.add_axes([center + map_width / 2 + .006, map_bottom + .003,
                           .0048, band_top - header_height - map_bottom - .006])
        ticks = np.asarray([-1, -.5, 0, .5, 1]) * limit
        bar = fig.colorbar(artist, cax=cax, ticks=ticks, extendfrac=.026)
        bar.set_ticklabels([f"{v:+.4g}" if v else "0" for v in ticks])
        bar.ax.tick_params(labelsize=7.3, length=2, width=.5, pad=2)
        bar.outline.set_linewidth(.5)
    for band in range(1, 3):
        y = top - band * band_height + band_gap / 2
        fig.add_artist(plt.Line2D([.047, .989], [y, y], transform=fig.transFigure,
                                  color="#d7dfe7", lw=.8))
    path = output / f"miw_{frame.lower()}_danish_vs_tarts_21modes"
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / f"map_metrics_{frame.lower()}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame", choices=("OCS", "CCS"), default="OCS")
    args = parser.parse_args()
    cfg = settings(args.config)
    root = args.input_dir or cfg["output_root"] / "miw_analysis" / args.variant / "field_maps"
    suffix = "pathA_50_34_i_5rot"
    branches = [read_native(root / branch / suffix, args.frame) for branch in ("danish", "tarts")]
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 3}):
        render(*branches, args.output_dir or root / "figures", args.frame,
               "TARTS DARE" if args.variant == "dare" else "TARTS")


if __name__ == "__main__":
    main()
