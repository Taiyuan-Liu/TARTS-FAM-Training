# Training inputs

Simulation data are read directly from the sibling Step02
`output/dataset`, with unchanged CCS Z4–Z28 labels and fixed splits.
There are no initialization checkpoints: supervised training starts randomly.

`real_fam/` supplies unlabelled targets for DARE-GRAM:

- `visit_manifest.csv`: fixed 256 train / 32 validation visits;
  `source_pickle` paths are relative to this directory.
- `heldout_visits.csv`: 123 visits reserved for downstream comparison.
- `shards/train`, `shards/val`: normalized 160×160 float16 images and
  physical metadata, without Zernike labels.
- `visits/visit_<visit>.pkl`: trusted full-CCD WEP pair packages;
  the reader keeps image, position, SNR and acquisition columns only.

`shards/` and `visits/` link to existing shared data and are never modified.
Recreate this layout or materialize the links when transferring the project.
The CSVs are regular local files. Danish labels and its `used` flag are
not adaptation inputs.
