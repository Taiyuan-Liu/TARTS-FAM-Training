# 2.1 Cut stamps

**Input:** one published imSim `amp_*.fits.fz`, state index and focal side.

**Method:** amplifier raw -> simulation-matched ISR -> WEP direct detection
-> recentered 200x200 stamps.

The ISR uses overscan, nominal camera gains and camera-model crosstalk.
Real bias/dark/flat/defect, deferred-charge, linearity and brighter-fatter
calibrations are disabled. WEP uses a source limit of 40 and cutout padding
of 40 pixels. No AlignNet or intra/extra source matching is involved.

```bash
python 01_ccd_to_stamps/ccd_to_stamps.py \
  --amp ../step01_simulation_dataset_generation/output/states/state_100/intra/amp_IM_P_20280313_000201-0-i-R01_S00-det000.fits.fz \
  --state-index 100 --side intra
```

The production sequence convention is checked: state k uses intra
`2*k+1` and extra `2*k+2`. The detector is read from the amplifier header.
Change `--source-limit` to change the maximum source count.

## Output

```text
output/stamps/state_100/intra/det000_R01_S00/
  donut_stamps.fits       # WEP DonutStamps
  stamp_metadata.ecsv    # one row per stamp, including field, SNR and effective flag
  manifest.json          # image identity, ISR/WEP settings and product hashes
```

`stamp_index` is the FITS row. WEP stores native stamp pixels in DVCS;
`DonutStamp.wep_im.image` supplies the CCS image used for training.
The metadata keeps both coordinate views, source and recentered positions,
quality flags, and exposure identity.

The manifest stores the number of stamps at `counts.stamps`, the source
exposure at `input`, and each file's size and checksum at
`products[filename].bytes` and `products[filename].sha256`.

This input is **amp**, not the assembled `raw_*` eimage. A CCD with no
accepted sources produces a valid empty stamp table.
