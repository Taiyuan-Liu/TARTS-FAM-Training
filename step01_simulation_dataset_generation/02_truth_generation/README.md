# 1.2 CCD-center optical truth

[truth.yaml](truth.yaml) defines the telescope, bandpass and OPD calculation.
`generate_detector_truth.py` fills its state parameters from one row of
`states.csv`: the 50 DOFs, pointing, band, time, zenith angle, rotator angle
and seeing. Detector IDs default to that row's ordered CCD list.

The truth telescope has **no added FAM defocus**. The author's camera piston
and other aberrations remain part of the selected 50-DOF state.

## Calculation

Resolve each CCD center through the nominal camera/telescope geometry, then
calculate its OPD using the selected perturbed telescope. Fit annular Z1–Z28
to the finite OPD pixels and retain Z4–Z28, with obscuration 0.61. OPD pixels
are in nm; saved coefficients are float32 microns in OCS.

These are monochromatic CCD-center optical labels. OCS-to-CCS rotation is
performed later, when attaching labels to stamps. The OPD path does not add
detector-specific focal-plane-height offsets to the perturbed telescope.

`opd_to_zernikes.py` implements the pixel fit. The imSim `AZ_*` headers
are not substituted for it: in the pinned implementation those diagnostics
use an unrotated field, whereas OPD pixels use the rotated field.

## Run: state 100

Run from `step01_simulation_dataset_generation/`, after loading the
[USDF environment](../README.md#environment-on-usdf).
Optical asset paths are set in [truth.yaml](truth.yaml).

```bash
python 02_truth_generation/generate_detector_truth.py \
  --state-index 100
```

Defaults: `--config` is `truth.yaml` beside the script; `--states` is
`output/selection/states.csv`; output is `output/states/state_100/truth/`.
Use `--states` for another selection or repeated `--detector-id` options
for a CCD subset. FEA and bending-mode asset paths are in `truth.yaml`;
`--fea-dir` and `--bend-dir` can override their locations.

To fit an existing OPD file directly:

```bash
python 02_truth_generation/opd_to_zernikes.py \
  --opd output/states/state_100/truth/opd_base_021_detectors_center.fits \
  --output-npy output/states/state_100/zk_from_opd.npy
```

## Outputs

- `opd_base_*_center.fits`: one 255 × 255 OPD image per CCD.
- `zk_true.npy`: shape `(N_CCD, 25)`, Z4–Z28, OCS, microns.
- `detector_fields.csv`: detector order, names and field coordinates.
- `manifest.json`: state parameters, CCD ordering, model configuration and
  label units/coordinates. Its model ID associates truth with both image sides.

`example_state100/` contains the published label manifest and detector-field
table for state 100's 21 CCDs.
