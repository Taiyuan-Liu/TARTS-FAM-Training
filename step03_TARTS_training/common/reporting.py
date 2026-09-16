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


def write_report(out, kind, history, best_epoch, result=None, prediction=None, truth=None,
                 zoom_quantile=0.002, plot_bounds=None):
    body = [f"<h1>{'WaveNet' if kind == 'wavenet' else 'Aggregator'}</h1>",
            f"<p>Selected model: epoch {best_epoch}. Coordinates: CCS. Z4–Z28, µm.</p>"]
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
        epoch = [int(row["epoch"]) for row in history]
        for axis, key, label in zip(axes, ("train_supervised", "val_mrsse_arcsec", "val_dare"),
                                    ("Train supervised loss", "Validation CCD-mean mRSSE [arcsec]"
                                     if kind == "wavenet" else "Validation mRSSE [arcsec]", "Validation DARE-GRAM")):
            axis.plot(epoch, [float(row[key]) for row in history])
            axis.axvline(best_epoch, color="0.6", linestyle=":")
            axis.set(xlabel="Epoch", ylabel=label)
        body.append(figure_html(fig))
    if result is not None:
        from .evaluation import METHOD_NAMES
        body.append(f"<h2>Simulation test — {html.escape(result['variant'])}</h2>"
                    f"<p>{result['ccd_groups']:,} state–CCD groups; {result['stamps']:,} stamps. "
                    "Every group contains both intra and extra. All methods use the same stamps and labels.</p>"
                    "<p>CCD equal: each group has equal weight. Stamp equal: each stamp has equal weight; "
                    "group predictions are weighted by their stamp counts. Single-stamp CCD-equal RMSE "
                    "averages squared errors within each group, then across groups and modes, before taking "
                    "the square root. mRSSE averages the arcsec-weighted residual norm with the same weights.</p>")
        body.append("<table><thead><tr><th>Method</th><th>CCD-equal RMSE [µm]</th>"
                    "<th>Stamp-equal RMSE [µm]</th><th>CCD-equal mRSSE [arcsec]</th>"
                    "<th>Stamp-equal mRSSE [arcsec]</th></tr></thead><tbody>")
        for name, method in result["methods"].items():
            if kind == "wavenet" and name == "aggregator":
                continue
            ccd, stamp = method["ccd_equal"], method["stamp_equal"]
            body.append(f"<tr><td>{METHOD_NAMES[name]}</td><td>{ccd['coefficient_rmse_um']:.6f}</td>"
                        f"<td>{stamp['coefficient_rmse_um']:.6f}</td><td>{ccd['mrsse_arcsec']:.6f}</td>"
                        f"<td>{stamp['mrsse_arcsec']:.6f}</td></tr>")
        body.append("</tbody></table>")
        plot_method = "wavenet_mean" if kind == "wavenet" else "aggregator"
        ccd, stamp = (result["methods"][plot_method][weight] for weight in ("ccd_equal", "stamp_equal"))
        body.append(f"<h2>{METHOD_NAMES[plot_method]}</h2><p>One point = one state–CCD group. "
                    "Panel RMSE values are CCD equal / stamp equal, in µm; correlation is CCD equal. "
                    "WaveNet and Aggregator use the same axes.</p>")
        fig, axes = plt.subplots(4, 7, figsize=(20, 11.25), layout="constrained")
        limits = []
        for i, axis in enumerate(axes.flat):
            if i >= 25:
                axis.set_visible(False)
                continue
            lo, hi = map(float, plot_bounds[i])
            axis.scatter(truth[:, i], prediction[:, i], s=3, alpha=0.25, rasterized=True)
            axis.plot([lo, hi], [lo, hi], color="0.4", linewidth=0.7)
            axis.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
            corr = ccd["per_mode_corr"][i]
            corr = f"{corr:.2f}" if corr is not None else "n/a"
            axis.set_title(f"Z{i + 4}  r={corr}\nRMSE={ccd['per_mode_rmse_um'][i]:.3f} / "
                           f"{stamp['per_mode_rmse_um'][i]:.3f}", fontsize=9)
            axis.tick_params(labelsize=7)
            visible = ((truth[:, i] >= lo) & (truth[:, i] <= hi)
                       & (prediction[:, i] >= lo) & (prediction[:, i] <= hi))
            limits.append(dict(noll=i + 4, lower_um=lo, upper_um=hi, points_outside_axes=int((~visible).sum())))
        fig.supxlabel("Truth [µm, CCS]")
        fig.supylabel("Prediction [µm, CCS]")
        body.append(figure_html(fig))
        body.append(f"<p>Axes use the {zoom_quantile:.1%}–{1 - zoom_quantile:.1%} joint quantiles of truth, "
                    "WaveNet CCD means and Aggregator predictions, plus a margin. Metrics include all points.</p>")
        body.append("<details><summary>Per-mode errors</summary><table><thead><tr><th>Mode</th>"
                    "<th>CCD-equal RMSE [µm]</th><th>Stamp-equal RMSE [µm]</th>"
                    "<th>CCD-equal bias [µm]</th><th>Stamp-equal bias [µm]</th></tr></thead><tbody>")
        for i in range(25):
            body.append(f"<tr><td>Z{i + 4}</td><td>{ccd['per_mode_rmse_um'][i]:.6f}</td>"
                        f"<td>{stamp['per_mode_rmse_um'][i]:.6f}</td><td>{ccd['per_mode_bias_um'][i]:.6f}</td>"
                        f"<td>{stamp['per_mode_bias_um'][i]:.6f}</td></tr>")
        body.append("</tbody></table></details>")
        encoded = json.dumps(dict(result, plot_method=plot_method, plot_limits=limits), indent=2, allow_nan=False)
        body.append('<details><summary>Metrics and plot limits</summary><pre>' + html.escape(encoded) + '</pre></details>')
        body.append('<script type="application/json" id="test-metrics">' + encoded + '</script>')
    config = (Path(out) / "config.yaml").read_text()
    body.append('<details><summary>Configuration and provenance</summary><pre>' + html.escape(config) + '</pre></details>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>TARTS training</title>'
                '<style>body{font:16px Arial,sans-serif;margin:24px;color:#222}img{width:100%;height:auto}'
                'table{border-collapse:collapse;margin:1em 0}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}'
                'th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap}details{margin:1em 0}'
                '</style><body>' + '\n'.join(body) + '</body></html>')
    partial = Path(out) / "report.html.partial"
    partial.write_text(document)
    partial.replace(Path(out) / "report.html")
