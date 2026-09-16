# Step04 — Danish comparison and MIW analysis

Step03 supplies the trained models. Step04 predicts real FAM wavefronts and
produces two groups of results:

| Entry | Result | Deck |
| --- | --- | --- |
| `danish_comparison/compare_zernikes.py` | Per-mode prediction–Danish scatter; before/after DARE residuals | 68, 71 |
| `danish_comparison/compare_donuts.py` | Simulated and real donuts on shared grayscale scales | 71 |
| `miw_analysis/field_maps/fit.py`, `plot.py` | Native MIW OCS/CCS maps, RMS and Danish–TARTS comparison | 67 |
| `miw_analysis/time_consistency/fit.py`, `compare.py` | Block-to-reference correlations and dated night panels | 72–73 |

The simulation/training setup and simulation-test scatter on pages 69–70
belong to Step01–03.

See [Results](output/RESULTS.md) for the final figures and metric tables.

## Layout

```text
step04_analysis/
  README.md
  config.yaml
  config.py                    Configuration reader and shared mode order
  requirements.txt
  .gitignore
  inference/
    predict.py                 Step03 models → real CCD predictions
  danish_comparison/           Predictions and images → comparison figures
    donut_example.npz
  miw_analysis/
    calibration/              OFC normalization and CCD-height map
    observations.csv          MIW observing metadata
    prepare_inputs.py          Predictions + source AOS tables → MIW inputs
    aos_tables.py              Broadcasting, clipping, averaging and rotation
    field_maps/                Fit maps; plot the saved native decomposition
    time_consistency/          Fit reference/blocks; compare saved maps
      visits.csv              Full i-band time-study selection
  external_repo/               Pinned upstream Git submodules
  output/                      Generated predictions, maps and figures
```

Commands below run from `step04_analysis/`. Paths inside `config.yaml`
are relative to that file; CLI paths are relative to the working directory.

## Models and data

The model paths are relative to `step03_TARTS_training/output/`:

| Variant | WaveNet | Aggregator |
| --- | --- | --- |
| `supervised` | `01_wavenet/model.pt` | `02_aggregator/model.pt` |
| `dare` | `03_dare_gram/wavenet/model.pt` | `03_dare_gram/aggregator/model.pt` |

Real WEP packages and the default 123-visit list are read through Step03's
`inputs/real_fam/`. Each trusted `visit_<id>.pkl` contains paired 200×200
CCS stamps, identities, field angles in degrees, SNR and pair FWHM.
WaveNet takes the central 160×160 crop with Step03's normalization.
Aggregator takes intra predictions followed by extra predictions, up to
200 stamps per CCD. Danish coefficients and its `used` flag do not enter
either network.

## Data by analysis

Each analysis keeps its own reference data and calibration:

| File | Purpose |
| --- | --- |
| `danish_comparison/donut_example.npz` | One saved simulated/real comparison: original 200×200 images, Danish-preprocessed 99×99 images and background masks |
| `miw_analysis/observations.csv` | Visit, observing night, program, rotator angle, band, altitude, azimuth and MJD for MIW preparation and dated plots |
| `miw_analysis/time_consistency/visits.csv` | The 1,494 visit IDs selected for the full i-band time study; passed to inference and MIW preparation with `--visits-file` |
| `miw_analysis/calibration/ofc_normalization.yaml` | Normalize the 50 telescope DOFs when constructing the OFC/SVD basis |
| `miw_analysis/calibration/ccd_height_map.fits.gz` | CCD-height calibration for the Z4 height contribution in the native MIW build and per-pair intrinsic output |

`config.py` reads the root YAML, resolves its relative paths and defines
the shared 21-mode comparison order Z4–Z19/Z22–Z26.

Rotator angles in `observations.csv` are in degrees; source AOS alt/az
are in radians. OFC normalization applies to the telescope DOFs, not to
image pixel values.

The default inference selection remains Step03's 123-visit
`heldout_visits.csv`; the 1,494-visit time-study file is selected explicitly.
The large real WEP image packages remain at
`/sdf/data/rubin/shared/aidonut/liuty/fam_refitwcs_pairs/visits/`,
accessed through Step03's `inputs/real_fam/visits` link.

The MIW adapter also reads the full source `aggregateAOSVisitTableRaw/Avg`
tables from Butler. Alternatively, `miw_analysis/aos_source/` can hold
local `visit_<id>/{raw,avg}.parquet` files, selected by `aos_source_dir`
or `--source-dir`. They supply pair geometry, intrinsic terms and rotation
metadata.

Generated predictions and figures go to `output/`. Large data products are
Git-ignored; figures, small metric tables and configuration records can be
included in the repository.

## Upstream dependencies

These are complete, unmodified repositories linked at fixed commits:

| Submodule | Upstream | Commit |
| --- | --- | --- |
| `ts_ofc` | [lsst-ts/ts_ofc](https://github.com/lsst-ts/ts_ofc) | `5245ded9c985c7707232f53f130a8cb6a402f7e1` |
| `ts_intrinsic_wavefront` | [lsst-ts/ts_intrinsic_wavefront](https://github.com/lsst-ts/ts_intrinsic_wavefront) | `7ce5729f5025c2e29afcbe57018734f74128650e` |
| `rubin-work` | [aaronroodman/rubin-work](https://github.com/aaronroodman/rubin-work) | `eedad8e9b7fe2559ceaa9f035aef4b085c2a050d` |

Clone the parent repository with `git clone --recurse-submodules`, or run
`git submodule update --init --recursive` after cloning.
The parent `.gitmodules` records the upstream URLs; its Git index records
the commits.

Inference uses Step03's PyTorch environment and `requirements.txt`.
For AOS conversion and MIW fitting on USDF:

```bash
source /sdf/home/l/liuty/rubin-user/.venv/setup.sh
setup -k -r external_repo/ts_ofc
setup -k -r external_repo/ts_intrinsic_wavefront
scons -Q -C external_repo/ts_intrinsic_wavefront python/lsst/ts/intrinsic/wavefront/version.py
```

The MIW wrapper calls the original `ts_intrinsic_wavefront/bin.src/`
entries. Time statistics import Aaron's original functions from
`rubin-work/aos/code/recompute_coadd_metrics.py`.

## 1. Predict real wavefronts

```bash
python -B inference/predict.py --variant supervised
python -B inference/predict.py --variant dare
```

For a one-CCD smoke, add `--visit 2026040100385 --max-detectors 1
--device cpu --output-dir <smoke-directory>`.
Use `--wavenet PATH --aggregator PATH` to override model paths.

Each visit NPZ contains `prediction_ccs_um`: one physical-CCS vector per
CCD, in µm, ordered Z4–Z28. It also stores CCD identities, sequence lengths,
per-stamp WaveNet outputs, mean/median summaries and the source pair mapping.
`pair_fwhm` is the packaged blur proxy used in AOS selection, not a TARTS
prediction. `model.json` records the loaded paths and coefficient convention.

## 2. Danish comparison

```bash
python -B danish_comparison/compare_zernikes.py
python -B danish_comparison/compare_donuts.py
python -B danish_comparison/compare_donuts.py \
  --sim-key simulation_danish_input_99 --real-key real_danish_input_99 \
  --output-dir output/danish_comparison/danish_inputs
```

The coefficient comparison uses the common visit×CCD sample and the 21
modes Z4–Z19, Z22–Z26, in physical CCS and µm. One scatter point is one CCD
prediction versus one Danish CCD average. Before/after DARE share axis
limits. Metrics include all selected points, including points outside the
zoomed plot limits. Residual plots show both mean ± sample standard
deviation and an alternative pair of mean curves.

The image command displays the saved arrays; it does not run Danish fitting.

## 3. MIW spatial maps

```bash
python -B miw_analysis/prepare_inputs.py --variant dare --from-butler
python -B miw_analysis/field_maps/fit.py --variant dare
python -B miw_analysis/field_maps/plot.py --variant dare
```

For local source AOS tables, replace `--from-butler` with
`--source-dir PATH`. The adapter broadcasts each CCD prediction to its
pairs and applies the native clipping, averaging and CCS→OCS conversion:
`Z_OCS = Z_CCS @ R(-rotTelPos)` for row vectors. MIW extraction does not
rotate the coefficients again.
It preserves both Raw/Avg tables and the MIW `donuts/visits.parquet` inputs.
A full-visit export needs full-visit predictions. Danish and TARTS retain
their own selected pairs; `--selection matched_danish` selects an explicitly
matched-pair comparison.

`field_maps/config.yaml` sets the MIW fit: i band, five rotator bins,
50 DOFs / 34 singular vectors, five iterations and a 73×73 field grid.
The author decomposition splits Z4 between telescope and camera; the other
modes use its OCS-only branch. The native polar decomposition is 80×180.
The plot reads those saved polar fields, uses the author's OCS RMS over
0.1–1.6 degrees, and shares one signed color range per mode between methods.
Use `--frame CCS` to plot the camera component.

Outputs include the author's `intrinsic_split.pdf` for each method,
compact 21-mode comparison figures and per-mode map metrics.

## 4. MIW time consistency

The time study uses the full i-band list, a fixed 193-visit reference, and
127 fixed blocks. Membership is in `time_consistency/reference_visits.json`
and `blocks.json`.

```bash
python -B inference/predict.py --variant dare \
  --visits-file miw_analysis/time_consistency/visits.csv \
  --output-dir output/predictions/full_iband_dare
python -B miw_analysis/time_consistency/prepare.py \
  --author-dir <Danish-MIW-extraction-directory> \
  --rotation-dir <raw-AOS-rotation-records-directory> \
  --predictions-dir output/predictions/full_iband_dare \
  --output-dir output/miw_analysis/full_iband_dare/inputs
python -B miw_analysis/time_consistency/fit.py reference --variant dare \
  --input-dir output/miw_analysis/full_iband_dare/inputs \
  --output-dir output/miw_analysis/full_iband_dare/time_consistency
python -B miw_analysis/time_consistency/fit.py blocks --variant dare \
  --input-dir output/miw_analysis/full_iband_dare/inputs \
  --output-dir output/miw_analysis/full_iband_dare/time_consistency
python -B miw_analysis/time_consistency/compare.py --variant dare \
  --grids-dir output/miw_analysis/full_iband_dare/time_consistency
```

The time-study preparation reads the Danish MIW extraction's
`donuts.parquet` and per-visit `rotation_<visit>.json` records containing
the raw AOS `rotTelPos` in radians. The sample spans several WEP collections.
It matches Danish's selected pair identities to the current predictions,
broadcasts each TARTS CCD vector to those pairs, and verifies the CCS→OCS
rotation against the source Danish coefficients. Both methods start with
identical pair positions and intrinsic terms. `pair_alignment.csv` records
the counts and rotation check per visit. The subsequent MIW fit applies its
own fit-quality selection to each method.

Time fits use the 50/34 basis and three iterations; all 21 modes enter the
fit, and Z5–Z8 are retained for comparison.
Add `--block 0` to fit a single block. Each method is compared with its
own fixed reference, using Aaron's 3×3 rebin and correlation functions.
The overview and dated panels show the mean spatial correlation of Z5–Z8;
reference-building blocks are marked. CSVs retain per-mode metrics,
block membership, MJD timing and summary statistics.

## Outputs

```text
output/
  RESULTS.md                 Figure and metric index
  predictions/{supervised,dare,full_iband_dare}/
  danish_comparison/{zernikes,donuts}/
  miw_analysis/
    danish/field_maps/        Shared Danish reference PDF
    {supervised,dare}/
      inputs/                Raw/Avg AOS tables and MIW Parquet inputs
      field_maps/            Native PDFs, OCS/CCS figures and metrics
    full_iband_dare/
      inputs/                Matched time-study pairs
      time_consistency/      Reference, block grids and time figures
```

Fitting and plotting are separate: replotting saved products does not repeat
the fits. Test scripts, scheduler wrappers and run logs are kept outside
this source directory.
