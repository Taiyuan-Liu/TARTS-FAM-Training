# TARTS for Full Array Mode

An end-to-end Rubin Full Array Mode (FAM) workflow based on
[PetchMa/TARTS](https://github.com/PetchMa/TARTS): simulate defocused CCDs,
prepare donut images and Zernike labels, train WaveNet and Aggregator, and
compare real-data predictions with Danish and measured intrinsic wavefronts
(MIW).

## Workflow

| Step | What it does |
| --- | --- |
| [01 — Simulation](step01_simulation_dataset_generation/README.md) | Select telescope states and CCDs; calculate no-FAM optical truth; render intra/extra CCD images with imSim. |
| [02 — Training data](step02_training_data_npz_preprocessing/README.md) | Amplifier raw → ISR/WEP → 200×200 stamps → CCS labels → 160×160 train/val/test NPZ shards. |
| [03 — Training](step03_TARTS_training/README.md) | Train a single-donut WaveNet, then a CCD-level Aggregator; adapt both to unlabelled real images with DARE-GRAM. |
| [04 — Analysis](step04_analysis/README.md) | Predict real FAM wavefronts, compare with Danish, and fit MIW spatial maps and time consistency. |

Training targets are CCD-center Z4–Z28 in physical CCS, in microns. The
additional ±1.5 mm FAM imaging defocus is not included in the target.
Step04 converts predictions to OCS for MIW analysis.

## Setup

Initialize the pinned external repositories after cloning:

```bash
git submodule update --init --recursive
```

On USDF, Rubin/ISR/WEP/Butler commands use:

```bash
source /sdf/home/l/liuty/rubin-user/.venv/setup.sh
```

Training and inference also need the PyTorch environment described in
[Step03](step03_TARTS_training/README.md#train). imSim optical assets and Gaia
access are described in [Step01](step01_simulation_dataset_generation/README.md#environment-on-usdf).
MIW's upstream setup/build commands are in
[Step04](step04_analysis/README.md#upstream-dependencies).
Submit compute jobs with an explicit Rubin account, partition and QOS.

## Checkpoints

[Download the four checkpoints from Google Drive](https://drive.google.com/drive/folders/1zehpesCujCeXMM4irpptTeNQoBTll3ev).
The folder and files are shared read-only with anyone who has the link.
This is the September 15, 2026 model set, approximately 330.5 MiB in total.

Download each file and place it at the following path relative to
`step03_TARTS_training/output/`, renaming it to `model.pt`:

| Model | Download | Destination | Selected epoch |
| --- | --- | --- | ---: |
| Supervised WaveNet | [wavenet.pt](https://drive.google.com/file/d/1HMWwXNDn_a84hG0QxuqMIf38ZoWOeDsV/view) | `01_wavenet/model.pt` | 10 |
| Supervised Aggregator | [aggregator.pt](https://drive.google.com/file/d/1SYWrwIITUppUd3w-PAkqFBhtq_Usd8_X/view) | `02_aggregator/model.pt` | 114 |
| DARE WaveNet | [dare_wavenet.pt](https://drive.google.com/file/d/1VHctjXzJv9d4GW3pvV4GUgDK7HrhaBhZ/view) | `03_dare_gram/wavenet/model.pt` | 12 |
| DARE Aggregator | [dare_aggregator.pt](https://drive.google.com/file/d/1BqD6pRNgypMESOlnsPI9gR8QESkyINuT/view) | `03_dare_gram/aggregator/model.pt` | 87 |

Use the supervised pair together, or the DARE pair together. These are
inference checkpoints, not optimizer/resume checkpoints. Their paths are
already configured in [Step04](step04_analysis/config.yaml).

## Training results

Each stage's `output/` contains its resolved `config.yaml`, `history.csv`,
`train.log` and self-contained HTML report. Download/open the HTML locally
to view its plots.

Simulation test: **1,938 state–CCD groups, 53,888 stamps**, including outer
detectors. Every group contains both intra and extra; all rows below use
the same stamps and CCD-center truth, CCS Z4–Z28 in µm.

CCD equal gives each state–CCD group equal weight. Stamp equal gives each
stamp equal weight; group predictions receive weight equal to their stamp
counts. Coefficient RMSE pools squared errors over samples and all 25 modes
before taking the square root. See [evaluation definitions](step03_TARTS_training/README.md#evaluate-and-predict).

### Supervised

| Method | CCD-equal RMSE [µm] | Stamp-equal RMSE [µm] |
| --- | ---: | ---: |
| WaveNet single stamp | 0.183293 | 0.120678 |
| WaveNet CCD mean | 0.172807 | 0.105394 |
| WaveNet CCD median | 0.174158 | 0.107498 |
| Aggregator | 0.169464 | 0.100597 |

Reports: [WaveNet](step03_TARTS_training/output/01_wavenet/report.html) ·
[Aggregator](step03_TARTS_training/output/02_aggregator/report.html).

### DARE-GRAM

| Method | CCD-equal RMSE [µm] | Stamp-equal RMSE [µm] |
| --- | ---: | ---: |
| WaveNet single stamp | 0.178564 | 0.116668 |
| WaveNet CCD mean | 0.171297 | 0.105861 |
| WaveNet CCD median | 0.171791 | 0.106751 |
| Aggregator | 0.168714 | 0.101778 |

Reports: [WaveNet](step03_TARTS_training/output/03_dare_gram/wavenet/report.html) ·
[Aggregator](step03_TARTS_training/output/03_dare_gram/aggregator/report.html).

Reports include mRSSE, per-mode errors and matched-axis test scatter plots.
Each scatter point represents one state–CCD group: a WaveNet CCD mean or an
Aggregator prediction. These tables evaluate simulation truth, not real-data accuracy.

## Real-data analysis

[Step04 results](step04_analysis/output/RESULTS.md) collect the TARTS–Danish
coefficient comparisons, OCS/CCS spatial MIW maps and i-band time-consistency
plots, with PNG/PDF figures and CSV metrics.

## Existing data on USDF

Datasets remain on USDF; they are not uploaded to GitHub or Google Drive.
The existing checkout exposes them through these output directories:

| Data | USDF path |
| --- | --- |
| Selected states and CCDs | `/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step01_simulation_dataset_generation/output/selection` |
| Simulated CCD images and no-FAM truth | `/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step01_simulation_dataset_generation/output/states` |
| 200×200 WEP stamps and metadata | `/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step02_training_data_npz_preprocessing/output/stamps` |
| Per-stamp labels | `/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step02_training_data_npz_preprocessing/output/labels` |
| Train/val/test NPZ shards and splits | `/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step02_training_data_npz_preprocessing/output/dataset` |

The training dataset contains 524,206 stamps in 1,026 NPZ shards. The split
is fixed by [splits.csv](step02_training_data_npz_preprocessing/03_make_npz/splits.csv).
Most data entries above are links to the published products in shared storage.
They are local access paths, not directories included in a Git clone.

Real FAM inputs used by Step03 and Step04 are available at:

```text
/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step03_TARTS_training/inputs/real_fam/shards
/sdf/data/rubin/user/liuty/TARTS/TARTS-FAM-Training/step03_TARTS_training/inputs/real_fam/visits
```

The per-step READMEs explain how to reuse these locations in another USDF
checkout. Access requires the corresponding USDF filesystem permissions;
the model downloads do not grant access to the data or Butler collection.

## Sources

- Network cores and DARE-GRAM are adapted from
  [PetchMa/TARTS](https://github.com/PetchMa/TARTS/tree/9ed6d0351100863bef8057a7a980ea875d17d51f).
  Its MIT copyright/license is retained in
  [Step03/LICENSE](step03_TARTS_training/LICENSE).
- MIW/OFC and Aaron Roodman's time-analysis code are referenced through
  [three pinned upstream submodules](.gitmodules). Their source and license
  information remain in their upstream repositories; `rubin-work` does not
  contain an explicit license at the pinned revision.
