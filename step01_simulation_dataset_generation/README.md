# Simulation dataset generation

Generate FAM CCD images and their CCD-center optical truth from the author's
existing telescope states.

| Stage | Main files | Result |
|---|---|---|
| [1.1 Telescope state sampling](01_telescope_state_sampling/README.md) | Author DOF table, `select_states.py`, sampled state/CCD tables | A selected telescope/observation state and its 21 science detectors |
| [1.2 Optical truth](02_truth_generation/README.md) | `truth.yaml`, `generate_detector_truth.py`, `opd_to_zernikes.py` | Base CCD-center OPD and Z4–Z28, OCS, microns |
| [1.3 CCD images](03_image_generation/README.md) | `image.yaml`, `sky_catalog.yaml`, `generate_images.py` | Separate full-catalog intra/extra CCD images |

```text
step01_simulation_dataset_generation/
  01_telescope_state_sampling/ # Select states and assign CCDs
  02_truth_generation/        # truth.yaml and no-FAM CCD-center Zernikes
  03_image_generation/        # image.yaml, sky catalog configuration and CCD rendering
```

The common example is **state 100**, author SEQID **25**, i-band. Its rendered
sides are sequence 201 (intra, -1.5 mm) and 202 (extra, +1.5 mm), both 15 s.
The scripts write to this step's `output/` and read the preceding outputs
there by default. These defaults are anchored to the step directory, not
the shell's working directory.

```text
output/
  selection/                 # states.csv and state_detectors.csv
  states/
    state_000/
      truth/                 # no-FAM OPD and zk_true.npy
      intra/                 # image_pass.yaml, raw_*, amp_*
      extra/                 # image_pass.yaml, raw_*, amp_*
    ...
    state_999/
```

For a new run, use an empty output location. Existing output directories,
including symlinks, are never overwritten. `--output-dir` changes an output
location; `--states` and `--label-dir` select inputs elsewhere.

## Existing data on USDF

The local `output/states/` links all currently published products from:

```text
/sdf/data/rubin/shared/aidonut/liuty/production_1000i
```

As of September 14, 2026: 1,000 states with truth, 919 published intra sides,
914 published extra sides, and 909 states with both sides. Unpublished sides
have no link. Each `state_NNN/` is a directory containing these symlinks:

| Local path | Published source within `state_NNN/` |
|---|---|
| `truth` | `labels/base_ccd_center` |
| `intra` | `intra_seqXXXXXX/output/full_catalog` (sequence = 2 × state + 1) |
| `extra` | `extra_seqXXXXXX/output/full_catalog` (sequence = 2 × state + 2) |

The selected state and CCD tables are also included in
`01_telescope_state_sampling/sampled_1000_states_and_ccds/`.
Use `select_states.py` for a new selection, or link this included directory
as `output/selection` to reuse the published selection.

The two passes read their own [truth](02_truth_generation/truth.yaml) and
[image](03_image_generation/image.yaml) configurations. Both take telescope
parameters from the selected state-table row. The image pass also uses the
[sky catalog configuration](03_image_generation/sky_catalog.yaml).

The truth pass retains the selected telescope state and calculates optical
labels without added FAM defocus. The image pass uses that same state with
camera offsets of -1.5 mm and +1.5 mm. Both sides share the no-FAM truth.

Each stage README explains its inputs, outputs and commands. Optical asset
paths are set in the truth and image YAMLs; the USDF environment is below.

## Environment on USDF

Use the designated Rubin environment and the installed imSim/catalog sources:

```bash
source /sdf/home/l/liuty/rubin-user/.venv/setup.sh
export TARTS_ASSETS=/sdf/data/rubin/user/liuty/TARTS/ai_exploring
export IMSIM_DIR="$TARTS_ASSETS/repositories/imSim_v2.1.1_author_candidate"
export SKYCATALOGS_DIR="$TARTS_ASSETS/repositories/skyCatalogs"
export PYTHONPATH="$IMSIM_DIR:$SKYCATALOGS_DIR:$TARTS_ASSETS/work/imsim/python:$TARTS_ASSETS/work/imsim/astropy_iers_candidates/0.2026.2.2.0.48.1${PYTHONPATH:+:$PYTHONPATH}"
export BATOID_RUBIN_DATA_DIR="$TARTS_ASSETS/work/imsim/batoid_rubin_data"
export RUBIN_SIM_DATA_DIR="$TARTS_ASSETS/work/imsim/rubin_sim_data"
export SIMS_SED_LIBRARY_DIR="$RUBIN_SIM_DATA_DIR/sims_sed_library"
export DAF_BUTLER_REPOSITORY_INDEX=/sdf/group/rubin/shared/data-repos.yaml
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
```

Gaia access uses the user's existing USDF Butler/database credentials.
Run simulation on compute nodes with an explicit Rubin account, partition
and QOS.
