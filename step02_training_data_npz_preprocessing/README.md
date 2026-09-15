# Simulation images to training data

Three steps produce one-image / one-label training samples:

```text
amp raw -> ISR + WEP -> 200x200 stamps
                              |
CCD-center OCS truth -> OCS-to-CCS rotation
                              |
                   effective stamps + CCS targets
                              |
              central 160x160 crop + fixed state split
                              |
                 train / val / test NPZ shards
```

| Step | Entrypoint | Output |
| --- | --- | --- |
| [2.1 Cut stamps](01_ccd_to_stamps/README.md) | `ccd_to_stamps.py` | WEP stamps and source metadata |
| [2.2 Attach labels](02_attach_labels/README.md) | `attach_labels.py` | A CCS and OCS Z4--Z28 row for each stamp |
| [2.3 Build dataset](03_make_npz/README.md) | `build_dataset.py` | Train/val/test NPZ shards and fixed splits |

A WaveNet data point is one donut image, its field position / focal side /
band, and 25 CCS coefficients in microns. Both sides use the same base
CCD-center truth. FAM imaging defocus is not added to the target; intra
and extra are separate samples, not matched star pairs.

## Environment

On USDF, initialize Rubin CVMFS w_2026_27 and the personal environment:

```bash
source /sdf/home/l/liuty/rubin-user/.venv/setup.sh
cd /sdf/home/l/liuty/rubin-user/TARTS/TARTS-FAM-Training/step02_training_data_npz_preprocessing
```

The scripts use Rubin ISR/camera/WEP, NumPy, Astropy and GalSim. Run actual
CCD processing and large builds in background compute sessions; use
`--workers` for parallel NPZ writing.

## Run through the three steps

From this directory, process one state-100 CCD and focal side:

```bash
python 01_ccd_to_stamps/ccd_to_stamps.py \
  --amp ../step01_simulation_dataset_generation/output/states/state_100/intra/amp_IM_P_20280313_000201-0-i-R01_S00-det000.fits.fz \
  --state-index 100 --side intra

python 02_attach_labels/attach_labels.py \
  --stage1-dir output/stamps/state_100/intra/det000_R01_S00

python 03_make_npz/build_dataset.py --workers 4
```

The first command reads Step01's `output/`. Label attachment reads the
matching truth and image association there automatically. Packing collects
the labelled CCD sides under this step's `output/labels/`.

```text
output/
  stamps/state_100/intra/det000_R01_S00/   # 200x200 WEP images and metadata
  labels/state_100/intra/det000_R01_S00/   # one label row per image
  dataset/
    train/part-0000.npz ...
    val/part-0000.npz ...
    test/part-0000.npz ...
    splits.csv
```

Defaults are anchored to this step directory, regardless of the shell's
working directory. For a new run, outputs must not already exist.
`--output-dir` selects another output; each substep also accepts explicit
input paths.

## Published dataset

The published dataset is available directly on USDF at
`/sdf/data/rubin/shared/aidonut/liuty/TARTS_training_data`.
It contains 524,206 samples in 1,026 NPZ shards.

The local `output/dataset/{train,val,test}` links to the corresponding
`shards/{train,val,test}` directories there. `output/dataset/splits.csv`
links to `03_make_npz/splits.csv`. `output/stamps` and
`output/labels` expose all 38,493 processed CCD sides in its source snapshot:
618,913 stamps, of which 524,206 pass the training selection.

The complete input list is:

```text
/sdf/data/rubin/user/liuty/TARTS/ai_exploring/data_preprocessing/published_20260901T193057Z/snapshot.csv
```

It combines the `01_ccd_to_stamps/` and `02_attach_labels/` products in:

```text
/sdf/data/rubin/user/liuty/TARTS/ai_exploring/data_preprocessing/published_20260821T061853Z
/sdf/data/rubin/shared/aidonut/liuty/data_preprocessing/published_20260824T054500Z
/sdf/data/rubin/user/liuty/TARTS/ai_exploring/data_preprocessing/published_20260901T193057Z
```

Read these products through `output/stamps/state_NNN/SIDE/detNNN_NAME/` and
`output/labels/state_NNN/SIDE/detNNN_NAME/`. Use a new output location to rerun
processing; the published link targets are not overwritten.

## Split policy

[The fixed split table](03_make_npz/splits.csv) preserves the 1000-state
design's 800/100/100 assignment and keeps pointing twins together. Dataset
building reads this table; it never recomputes splits from available images.

Earlier 100-to-1000 expansion moved seven previously trained states into
test (1, 12, 18, 22, 28, 30, 62). Evaluation of warm-started checkpoints must
exclude ancestral training overlap; a fixed table does not remove that
earlier model exposure.
