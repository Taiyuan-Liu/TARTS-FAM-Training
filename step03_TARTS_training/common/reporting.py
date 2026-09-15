"""A single portable HTML report per stage; figures are embedded, not sidecar files."""

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def figure_html(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    value = base64.b64encode(stream.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{value}" alt="Training or test plot">'


def write_report(out, kind, history, best_epoch, result=None, prediction=None, truth=None, zoom_quantile=0.002):
    body = [f"<h1>{'WaveNet' if kind == 'wavenet' else 'Aggregator'}</h1>",
            f"<p>Selected model: epoch {best_epoch}. Coordinates: CCS. Z4–Z28, µm.</p>"]
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
        epoch = [int(row["epoch"]) for row in history]
        for axis, key, label in zip(axes, ("train_supervised", "val_mrsse_arcsec", "val_dare"),
                                    ("Train supervised loss", "Validation mRSSE [arcsec]", "Validation DARE-GRAM")):
            axis.plot(epoch, [float(row[key]) for row in history])
            axis.axvline(best_epoch, color="0.6", linestyle=":")
            axis.set(xlabel="Epoch", ylabel=label)
        body.append(figure_html(fig))
    if result is not None:
        body.append(f"<h2>Simulation test</h2><p>One point = {result['point']}. "
                    f"N = {result['samples']:,}. RMSE = {result['coefficient_rmse_um']:.5f} µm. "
                    "The test split is not used for model selection.</p>")
        fig, axes = plt.subplots(4, 7, figsize=(20, 11.25), layout="constrained")
        limits = []
        for i, axis in enumerate(axes.flat):
            if i >= 25:
                axis.set_visible(False)
                continue
            lo, hi = np.quantile(np.concatenate([truth[:, i], prediction[:, i]]),
                                 [zoom_quantile, 1 - zoom_quantile])
            margin = max((hi - lo) * 0.07, 1e-4)
            lo, hi = float(lo - margin), float(hi + margin)
            axis.scatter(truth[:, i], prediction[:, i], s=3, alpha=0.25, rasterized=True)
            axis.plot([lo, hi], [lo, hi], color="0.4", linewidth=0.7)
            axis.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
            corr = result["per_mode_corr"][i]
            corr = f"{corr:.2f}" if corr is not None else "n/a"
            axis.set_title(f"Z{i + 4}  r={corr}\nRMSE={result['per_mode_rmse_um'][i]:.3f}", fontsize=9)
            axis.tick_params(labelsize=7)
            visible = ((truth[:, i] >= lo) & (truth[:, i] <= hi)
                       & (prediction[:, i] >= lo) & (prediction[:, i] <= hi))
            limits.append(dict(noll=i + 4, lower_um=lo, upper_um=hi, points_outside_axes=int((~visible).sum())))
        fig.supxlabel("Truth [µm, CCS]")
        fig.supylabel("Prediction [µm, CCS]")
        body.append(figure_html(fig))
        result["plot_limits"] = limits
        body.append(f"<p>Axes use the {zoom_quantile:.1%}–{1 - zoom_quantile:.1%} joint quantile range "
                    "plus a margin. Metrics include all points; omitted-point counts are below.</p>")
        encoded = json.dumps(result, indent=2, allow_nan=False)
        body.append('<details><summary>Metrics and plot limits</summary><pre>' + html.escape(encoded) + '</pre></details>')
        body.append('<script type="application/json" id="test-metrics">' + encoded + '</script>')
    else:
        body.append("<p>Test evaluation has not been completed for this report.</p>")
    config = (Path(out) / "config.yaml").read_text()
    body.append('<details><summary>Configuration and provenance</summary><pre>' + html.escape(config) + '</pre></details>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>TARTS training</title>'
                '<style>body{font:16px Arial,sans-serif;margin:24px;color:#222}img{width:100%;height:auto}'
                'pre{white-space:pre-wrap}details{margin:1em 0}</style><body>' + '\n'.join(body) + '</body></html>')
    partial = Path(out) / "report.html.partial"
    partial.write_text(document)
    partial.replace(Path(out) / "report.html")
