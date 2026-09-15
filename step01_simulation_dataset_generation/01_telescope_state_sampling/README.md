# 1.1 Telescope state sampling

Select observing states from the author's complete 11,999-row table and assign
21 science CCDs to each state. The selected 50-DOF vectors and observing
parameters are used unchanged; this step samples existing states, without
generating new perturbations or fitting Zernikes to DOFs.

## Files

| File | Purpose |
|---|---|
| `production_visits_ldofs_11999.csv` | Full author observing-state and 50-DOF table, including all bands |
| `select_states.py` | Select states for one band and assign balanced CCD coverage |
| `sampled_1000_states_and_ccds/states.csv` | Published 1,000-state selection used by the following stages |
| `sampled_1000_states_and_ccds/state_detectors.csv` | Published per-state detector names, IDs and focal-plane positions |

## Selection method

First filter the author table by band. Split numerical DOF norm into five
quantile bins and altitude into four global quantile bins. Within each of the
20 DOF-norm/altitude cells, split rotator angle (`RTP`) into five quantile bins.
These axes give 100 strata. For a requested size `N`, sample `N / 100` states
without replacement from every stratum. `N` must be a positive multiple of 100
and each stratum must contain enough candidates.

The DOF norm is the Euclidean norm of the 50 stored numerical values. Its
components have mixed physical units, so it describes perturbation size for
sampling, not wavefront RMS. Altitude and rotator angle are in degrees.

CCD assignment is independent of state stratification. Each state gets one
of the nine science CCDs in each of 21 rafts. Balanced assignments cover all
189 science CCDs; for 1,000 states each CCD appears 111 or 112 times.

## Make a new selection

Run from `step01_simulation_dataset_generation/`:

```bash
source /sdf/home/l/liuty/rubin-user/.venv/setup.sh
python 01_telescope_state_sampling/select_states.py \
  --band i --n-states 1000 --seed 20260822
```

`--author-table` optionally selects another source CSV; the default is the
11,999-row CSV beside the script. The default output is `output/selection/`.
The default author configuration root is
`/sdf/data/rubin/shared/aos_group/final_imsim_simulation`; use
`--author-config-root` to change it. That root contains
`perturbation_<SEQID>/<SEQID>.yaml`; their paths and hashes record the source
of each selected state. The simulation passes read their own `truth.yaml`
and `image.yaml`, with state parameters supplied by `states.csv`.

The output directory receives `states.csv`, `state_detectors.csv` and
`selection.json`, a compact record of the selection settings. Use
`sampled_1000_states_and_ccds/` for the published sample, or pass the tables from a
new selection to the following stages. Both simulation scripts read
`output/selection/states.csv` by default.

## Hand off a state to the two simulation passes

Read the row for `state_index = k` from `states.csv` with a CSV parser.
`author_config` and `author_config_sha256` identify the author YAML and its
contents. `dof_json` stores the unchanged 50-DOF vector; `dof_norm` and the
three stratum columns describe its selection. Pointing, time, band, altitude,
rotator angle, airmass and seeing remain attached to that state.

Use the semicolon-separated `detector_ids` in their recorded order for both
passes. `state_detectors.csv` supplies the matching detector names and
focal-plane coordinates in raft order. For state `k`, the intra sequence is
`2*k + 1` with focus `-1.5 mm`; the extra sequence is `2*k + 2` with focus
`+1.5 mm`. The two exposures share `group_id`. Carry the author DOFs into
the exposure metadata so downstream Zernike labels describe the same state.

The following stages use published state 100: author SEQID 25, intra sequence
201 and extra sequence 202. Its ordered detector IDs are:

```text
0,14,25,32,43,46,56,64,73,88,92,104,113,124,131,140,144,161,169,178,184
```

For another state, pass its `--state-index` to the truth and image entry points.
Both use that row's telescope parameters, detector list and pair identifiers.
