# Step04 results

Figures and metrics for Danish comparisons, spatial MIW maps and i-band
time consistency, using the supervised and DARE Step03 models.

## Danish comparison

Compare TARTS predictions with Danish CCD averages on 123 visits
(21,965 visit/CCD points), in physical CCS and microns.

| Figure | PNG | PDF |
| --- | --- | --- |
| Supervised vs Danish | [PNG](danish_comparison/zernikes/supervised_vs_danish.png) | [PDF](danish_comparison/zernikes/supervised_vs_danish.pdf) |
| DARE vs Danish | [PNG](danish_comparison/zernikes/dare_vs_danish.png) | [PDF](danish_comparison/zernikes/dare_vs_danish.pdf) |
| Per-mode residuals: mean and sample SD | [PNG](danish_comparison/zernikes/residual_errorbars.png) | [PDF](danish_comparison/zernikes/residual_errorbars.pdf) |
| Per-mode mean residuals | [PNG](danish_comparison/zernikes/residual_lines.png) | [PDF](danish_comparison/zernikes/residual_lines.pdf) |
| Simulated and real donut examples | [PNG](danish_comparison/donuts/sim_real_grayscale.png) | [PDF](danish_comparison/donuts/sim_real_grayscale.pdf) |
| Danish-preprocessed donut examples | [PNG](danish_comparison/danish_inputs/sim_real_grayscale.png) | [PDF](danish_comparison/danish_inputs/sim_real_grayscale.pdf) |

Metrics: [supervised](danish_comparison/zernikes/supervised_metrics.csv),
[DARE](danish_comparison/zernikes/dare_metrics.csv).

## Spatial MIW

Fit spatial wavefront maps from the 123-visit sample and compare their
OCS and CCS components. The configuration uses five rotator bins,
50 DOFs and 34 singular vectors, with the OCS/CCS split enabled for Z4.

| Figure | PNG | PDF |
| --- | --- | --- |
| Supervised / Danish OCS | [PNG](miw_analysis/supervised/field_maps/figures/miw_ocs_danish_vs_tarts_21modes.png) | [PDF](miw_analysis/supervised/field_maps/figures/miw_ocs_danish_vs_tarts_21modes.pdf) |
| DARE / Danish OCS | [PNG](miw_analysis/dare/field_maps/figures/miw_ocs_danish_vs_tarts_21modes.png) | [PDF](miw_analysis/dare/field_maps/figures/miw_ocs_danish_vs_tarts_21modes.pdf) |
| Supervised / Danish CCS | [PNG](miw_analysis/supervised/field_maps/figures/miw_ccs_danish_vs_tarts_21modes.png) | [PDF](miw_analysis/supervised/field_maps/figures/miw_ccs_danish_vs_tarts_21modes.pdf) |
| DARE / Danish CCS | [PNG](miw_analysis/dare/field_maps/figures/miw_ccs_danish_vs_tarts_21modes.png) | [PDF](miw_analysis/dare/field_maps/figures/miw_ccs_danish_vs_tarts_21modes.pdf) |

Native-layout PDFs: [Danish](miw_analysis/danish/field_maps/danish_intrinsic_split.pdf),
[supervised](miw_analysis/supervised/field_maps/tarts_intrinsic_split.pdf),
[DARE](miw_analysis/dare/field_maps/tarts_intrinsic_split.pdf).

| Metrics | OCS | CCS |
| --- | --- | --- |
| Supervised / Danish | [CSV](miw_analysis/supervised/field_maps/figures/map_metrics_ocs.csv) | [CSV](miw_analysis/supervised/field_maps/figures/map_metrics_ccs.csv) |
| DARE / Danish | [CSV](miw_analysis/dare/field_maps/figures/map_metrics_ocs.csv) | [CSV](miw_analysis/dare/field_maps/figures/map_metrics_ccs.csv) |

## I-band time consistency

DARE inference covers 1,494 visits. The time study compares 127 blocks
against each estimator's fixed 193-visit reference, using OCS Z5–Z8
and 3×3 spatial rebinning.

| Figure | PNG | PDF |
| --- | --- | --- |
| Block-index overview | [PNG](miw_analysis/full_iband_dare/time_consistency/figures/tarts_vs_danish_self_mean.png) | [PDF](miw_analysis/full_iband_dare/time_consistency/figures/tarts_vs_danish_self_mean.pdf) |
| Dated view | [PNG](miw_analysis/full_iband_dare/time_consistency/figures/tarts_vs_danish_self_mean_dated.png) | [PDF](miw_analysis/full_iband_dare/time_consistency/figures/tarts_vs_danish_self_mean_dated.pdf) |

Tables: [per-block metrics](miw_analysis/full_iband_dare/time_consistency/figures/time_metrics.csv),
[non-reference-block summary](miw_analysis/full_iband_dare/time_consistency/figures/nonreference_summary.csv),
[block dates](miw_analysis/full_iband_dare/time_consistency/figures/block_dates.csv).
