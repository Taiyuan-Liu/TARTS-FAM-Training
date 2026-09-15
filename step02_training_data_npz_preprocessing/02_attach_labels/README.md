# 2.2 Attach labels

**Input:** a stamp directory and Step01's corresponding `output/states/state_NNN/truth/`.

**Method:** match the detector to its truth row, rotate that vector into
CCS, and repeat it over the stamp rows.

For row-vector GalSim coefficients, pad Z4--Z28 to indices 0--28, then:

```text
Z_CCS = Z_OCS @ zernikeRotMatrix(28, +rotTelPos_radians)
```

Truth is Z4--Z28 in microns, annular obscuration 0.61, without the added
FAM imaging defocus. There is no focal-side sign flip. All stamps of one
state/CCD share its center label, even though their actual fields differ.

```bash
python 02_attach_labels/attach_labels.py \
  --stage1-dir output/stamps/state_100/intra/det000_R01_S00
```

State and side are obtained from the stamp metadata. The truth defaults to
`../step01_simulation_dataset_generation/output/states/state_100/truth/`, and the
image association to `output/states/state_100/intra/image_pass.yaml.manifest.json`
under Step01. Use `--label-dir` and `--image-manifest` for inputs elsewhere.
The association check compares the base telescope state, detector, exposure
and focus.

## Output

```text
output/labels/state_100/intra/det000_R01_S00/
  stamp_truth_ccs_um.npy    # (N,25), training targets
  stamp_truth_ocs_um.npy    # (N,25), original-frame targets
  stamp_label_index.ecsv   # source metadata plus truth row, rotTelPos and reference field
  manifest.json            # source association and coordinate/label definitions
```

`stage1/donut_stamps.fits[i].wep_im.image` corresponds to row i of both
target arrays and the label index. Roundtrip, norm and axisymmetric-mode
checks verify the rotation algebra.

The manifest records `stage1.directory`, `stage1.count` and source checksums;
`label_source.directory` and `source_detector_row` identify the CCD truth.
`image_identity` contains exposure/detector/side fields, and
`one_to_one_contract.row_count` matches the arrays and index.
`physics` records `noll_indices`, `stamp_image_frame`,
`training_target_frame`, `published_truth_frame`, `annular_obscuration`,
`rtp_deg`, `units` and `fam_diversity_in_truth`.
