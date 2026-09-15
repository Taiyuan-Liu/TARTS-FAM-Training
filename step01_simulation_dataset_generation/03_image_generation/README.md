# 1.3 Full-catalog FAM CCD images

Render the selected telescope state twice: intra at -1.5 mm and extra at
+1.5 mm. These offsets are added to the same base optical state used by the
truth pass. Both exposures share its no-FAM CCD-center Zernike labels.

## Files and calculation

- [image.yaml](image.yaml): telescope/FEA, atmosphere, sky, photon tracing,
  diffraction, silicon response and amplifier readout.
- [sky_catalog.yaml](sky_catalog.yaml): Gaia catalog access and source model.
- [generate_images.py](generate_images.py): bind a state and exposure to the
  image configuration, then run GalSim with the imSim plugin.

The script reads the 50 DOFs, pointing, band, time, rotator angle and seeing
from one row of `states.csv`. CCD IDs and order come from the supplied truth
directory. It checks that the base optical configuration matches the truth,
then sets the side's focus, sequence number, detector list and output paths.
There is no OPD output in this pass.

The sequence number sets the image random seed. Intra and extra have distinct
sequence numbers and share `GROUPID`. The telescope focus is in metres;
the output `FOCUSZ` header is in millimetres.

## Run: state 100

Run from `step01_simulation_dataset_generation/`, after loading the
[USDF environment](../README.md#environment-on-usdf).
Optical asset paths are set in [image.yaml](image.yaml).

```bash
python 03_image_generation/generate_images.py \
  --state-index 100 --fam-side intra --nproc 21

python 03_image_generation/generate_images.py \
  --state-index 100 --fam-side extra --nproc 21
```

Run long renders as background batch jobs on compute nodes. Each command
prepares its configuration and renders all selected CCDs. `--nproc` controls
CCD-level parallelism; the atmosphere's worker count is in `image.yaml`.
State 100 uses sequence 201/202 and 15 s exposures. The truth input defaults
to `output/states/state_100/truth/`; images go to `output/states/state_100/intra/` and
`output/states/state_100/extra/`.

Add `--prepare-only` to inspect the filled YAML without rendering. Use separate
output directories for different states and sides. Optional
`--checkpoint-dir output/states/state_100/checkpoints/intra` enables imSim's per-CCD
checkpoint files; `--nbatch-per-checkpoint` defaults to 8. Reuse checkpoints
only for the same exposure and unchanged simulation settings.

Defaults: `--config` is the adjacent `image.yaml`; `--states` is
`output/selection/states.csv`.
For a CCD subset, generate truth for that subset and pass its `--label-dir`.
Asset locations are in the YAMLs. `--sky-config`, `--tree-rings`, `--fea-dir`
and `--bend-dir` override them. Optical asset paths must match the truth pass.

The render calls the GalSim CLI with `-v 2 -x -n 1 -j 1`: normal progress
logging, abort on error, one job subdivision. This does not limit the CCD
count. `--verbosity` changes logging. The entry point disables IERS
auto-download and uses the IERS data supplied by the configured environment.

## Outputs

Each side's output directory contains:

```text
image_pass.yaml                 # Actual imSim input for this exposure
image_pass.yaml.manifest.json   # Direct image/truth association
raw_*.fits.fz                   # Assembled electron images (eimages)
amp_*.fits.fz                   # Segmented amplifier raw images
```

The small association record contains state/CCD identity, exposure/side/focus,
the paired group, and the corresponding truth/model identity. It sits beside
the images so label attachment can find it directly.

Downstream stamp extraction uses **amp -> simulation-matched ISR -> stamps**.
The `raw_` prefix here names the eimage, not the amplifier raw.
