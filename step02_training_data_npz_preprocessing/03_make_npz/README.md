# 2.3 Build the dataset

**Input:** labelled CCD sides under `output/labels/` and the fixed `splits.csv`.

```bash
python 03_make_npz/build_dataset.py --workers 4
```

Output goes to `output/dataset/`. No hand-written input list is needed for
the default layout.

For labels stored elsewhere, `--inputs` can select another label root or
a CSV with one row per CCD side. Paths may be absolute or relative to the CSV:

```csv
stage2_dir
state082_det005/intra/stage2
state082_det005/extra/stage2
```

Existing published label directories can be listed directly. The source
stamp path is read from `manifest.json` at `stage1.directory`; relative
paths there are resolved against the label directory. If stamps have been
relocated, add a `stage1_dir` column to `inputs.csv` with their current
location. The recorded source checksums must still match.

Then run one command:

```bash
python 03_make_npz/build_dataset.py \
  --inputs /path/to/run/inputs.csv \
  --output-dir /path/to/new_dataset \
  --workers 4
```

## Method

1. Keep stamps with WEP `effective == 1`.
2. Read the CCS view `.wep_im.image` and crop `[20:180,20:180]`.
3. Look up the state's split in `splits.csv`; all its CCDs and sides stay together.
4. Write at most 512 samples per uncompressed NPZ.

`--samples-per-shard` sets shard capacity. `--splits` selects another fixed
registry; unknown states fail instead of triggering a new random split.
The supplied 1000-state registry retains the published fixed assignment.
Its `split_group_id` identifies indivisible pointing twins.

A larger input subset uses the same split registry and a new output
directory. Sample IDs and shard boundaries are local to each build.

## Output

```text
dataset/
  train/part-0000.npz ...
  val/part-0000.npz ...
  test/part-0000.npz ...
  splits.csv
```

Each shard contains parallel arrays, where N is at most 512:

| Field | Shape | Meaning |
| --- | --- | --- |
| `image` | (N,160,160) | float32 CCS image, not yet normalized |
| `zk_true_ccs_um` | (N,25) | float32 Z4--Z28 training target |
| `zk_true_ocs_um` | (N,25) | original-frame copy |
| `field_x_deg`, `field_y_deg` | (N,) | actual donut field, CCS degrees |
| `intra`, `band` | (N,) | intra=1 / extra=0; ugrizy=0..5 |
| `state_index`, `detector_id`, `seq_num` | (N,) | image identity |
| `sample_id`, `source_stamp_index` | (N,) | dataset row ID and source FITS row |
| `snr`, `rtp_deg` | (N,) | quality and camera rotation |

`splits.csv` contains the fixed `state_index,split_group_id,split` assignment,
including states not yet present in the NPZ files. Step03 reads each sample's
metadata directly from its NPZ and groups Aggregator inputs by state/CCD.

To find a source image, use the NPZ's state, detector, focal side and
`source_stamp_index`: look under `output/stamps/state_NNN/SIDE/detNNN_*/`
and select that row of `donut_stamps.fits`. Its label is the same row under
`output/labels/`. No source-directory paths are needed during training.

Training performs normalization when loading: image min/peak followed by
mean/population-standard-deviation scaling; fields in radians / 0.021;
side as `(intra-0.5)/0.5`; band through the author's wavelength lookup.
No normalization is baked into the NPZ.
