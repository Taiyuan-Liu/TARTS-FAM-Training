# TARTS training for FAM

Train WaveNet and Aggregator from random initialization, then adapt both
networks to real images with DARE-GRAM.

## Inputs

- Simulation: `../step02_training_data_npz_preprocessing/output/dataset`.
  Read the NPZ shards and fixed state/pointing splits directly.
- Real images: `inputs/real_fam/`; see [input layout](inputs/README.md).
- Images are 160×160. Labels are CCD-center Z4–Z28 in physical CCS, µm.
- WaveNet receives the image, field position, defocus side and band.
  Images use per-image minmax then population z-score; fields use
  `deg_to_rad / 0.021`, side uses `2*intra-1`, and band uses
  `(effective_wavelength_um-0.710)/0.174`.

Configuration paths are relative to their YAML file. CLI paths are relative
to the working directory.

## Train

Install [requirements.txt](requirements.txt) in a CUDA/PyTorch environment.
Use one GPU, eight CPU cores and 96 GB host RAM for the full dataset.
Run each command in a background GPU job, from this directory:

```bash
python -B 01_wavenet/train.py
python -B 02_aggregator/train.py --wavenet output/01_wavenet/model.pt

python -B 03_dare/adapt.py --stage wavenet \
    --checkpoint output/01_wavenet/model.pt
python -B 03_dare/adapt.py --stage aggregator \
    --checkpoint output/02_aggregator/model.pt \
    --wavenet output/03_dare_gram/wavenet/model.pt
```

WaveNet uses ResNet101 and a 256/205/69 predictor. Supervised training
traverses all stamps each epoch. Model selection uses validation mRSSE of
the per-CCD mean predictions.

Aggregator freezes the supplied WaveNet and groups its predictions by
state/CCD, requiring both defocus sides. Tokens contain 25 predictions,
two field coordinates in degrees and relative SNR. Sequences are constructed
in memory, ordered intra then extra and padded to 200 tokens. Training and
validation use mRSSE.

DARE-GRAM combines simulation regression with feature alignment to real
images. WaveNet adaptation samples eight stamps per CCD per epoch;
Aggregator adaptation uses all simulation CCD groups. Target forwards
retain gradients while keeping BatchNorm statistics fixed.

The DARE weight decreases exponentially between the initial simulation
validation error and 1.05 times that error. Within this bound, checkpoint
selection minimizes `E/E0 + D/D0`: simulation error and feature alignment
loss relative to their initial values. Settings are in
[03_dare/config.yaml](03_dare/config.yaml).

Use `--config path.yaml` for another configuration and `--resume path/resume.pt`
to continue training. `epochs` is the maximum epoch number.

## Outputs

```text
output/
├── 01_wavenet/
├── 02_aggregator/
└── 03_dare_gram/
    ├── wavenet/
    └── aggregator/
```

Each training directory contains:

| File | Contents |
| --- | --- |
| `config.yaml` | Resolved settings, data identity and environment |
| `train.log` | Training log |
| `history.csv` | Per-epoch losses, validation metrics, LR and timing |
| `model.pt` | Selected inference model and input conventions |
| `resume.pt` | Latest model, optimizer, scheduler and RNG state |
| `report.html` | Training curves and evaluation results |

Aggregator's `cache/` contains per-stamp predictions and IDs, with metadata
for real CCD groups. Cached predictions are tied to the input data and
WaveNet checkpoint. Model files, caches and large input links are excluded
from Git. The published `config.yaml`, `history.csv`, `train.log` and
`report.html` are regular files in the stage's `output/` directory.
Checkpoint download links and their destination paths are in the
[root README](../README.md#checkpoints).

## Evaluate and predict

Evaluate each model pair on the simulation test split:

```bash
python -B evaluate.py --variant supervised
python -B evaluate.py --variant dare
```

Each command updates the WaveNet and Aggregator `report.html` files. Use
`--wavenet` and `--aggregator` to select explicit checkpoints. Cached WaveNet
predictions are reused; `--device cpu` runs the remaining Aggregator inference
on CPU when that cache is available.

All comparisons use the same state/CCD groups with both defocus sides and
all stamps in those groups, including the outer detectors. Reported methods
are single-stamp WaveNet, its CCD mean and median, and Aggregator. Both
coefficient RMSE [µm] and mRSSE [arcsec] use two weightings:

- **CCD equal:** each state/CCD group contributes equally. For single-stamp
  WaveNet, each stamp receives weight `1 / (stamps in its group)`; predictions
  are not averaged before computing its errors.
- **Stamp equal:** each stamp contributes equally. A group-level prediction
  receives weight equal to the number of stamps in its group.

Coefficient RMSE pools squared errors over the population and all 25 modes
before taking the square root. mRSSE averages the arcsec-weighted residual
norm. Training losses and checkpoint selection remain as described above.

Scatter plots use one point per state/CCD: the WaveNet CCD mean or Aggregator
prediction. Both reports share axes, using the joint 0.2–99.8% quantiles of
truth and both predictions. Metrics include all points; `--zoom-quantile 0`
displays the full range. Tables and per-mode statistics for both weightings
are embedded in the HTML, including a `test-metrics` JSON block.

`common.models.load_model(path, kind, device)` loads an inference checkpoint.
Use the WaveNet recorded by the Aggregator's checkpoint. Outputs are CCS
Z4–Z28 in µm; conversion to OCS belongs to downstream analysis.

## Source

Network cores and DARE-GRAM are adapted from
[PetchMa/TARTS](https://github.com/PetchMa/TARTS), commit
`9ed6d0351100863bef8057a7a980ea875d17d51f`.
FAM loading, grouping and training configuration are implemented here.
The author's license is in [LICENSE](LICENSE).
