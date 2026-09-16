#!/usr/bin/env python3
"""Align current predictions with the Danish MIW sample on identical pair support."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import galsim
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table
from step04_analysis.config import MODE_INDEX, NOLL, settings

IDENTITY = ['detector', 'intra_donut_id', 'extra_donut_id']
COLUMNS = ['day_obs', 'seq_num', *IDENTITY, 'thx_OCS', 'thy_OCS',
           'zk_CCS', 'zk_OCS', 'zk_intrinsic_OCS']


def align_visit(source, prediction, theta):
    """Broadcast one CCD vector, then apply the standard CCS-to-OCS rotation."""
    assert str(prediction['frame']) == 'CCS' and str(prediction['unit']) == 'micron'
    assert np.array_equal(prediction['noll'], np.arange(4, 29))
    names = dict(zip(prediction['detector_id'], prediction['detector_name']))
    pairs = pd.MultiIndex.from_arrays([
        [str(names[d]) for d in prediction['pair_detector_id']],
        prediction['intra_donut_id'].astype(str), prediction['extra_donut_id'].astype(str)])
    keys = pd.MultiIndex.from_frame(source[IDENTITY].astype(str))
    if not pairs.is_unique or not keys.is_unique:
        raise ValueError('Duplicate pair identity')
    matched = pairs.get_indexer(keys) >= 0
    danish = source.loc[matched].reset_index(drop=True)
    if len(danish) < 100:
        raise ValueError('Insufficient matched Danish/TARTS pair support')
    rotation = galsim.zernike.zernikeRotMatrix(26, -theta)[4:, 4:]
    padded = np.zeros((len(danish), 23))
    padded[:, MODE_INDEX] = np.stack(danish.zk_CCS)
    gate = float(np.max(np.abs((padded @ rotation)[:, MODE_INDEX] - np.stack(danish.zk_OCS))))
    if not np.isfinite(gate) or gate > 1e-6:
        raise ValueError(f'Danish CCS/OCS source rotation disagrees by {gate} microns')
    lookup = pd.DataFrame(prediction['prediction_ccs_um'][:, MODE_INDEX],
                          index=prediction['detector_name'].astype(str))
    ccs = lookup.loc[danish.detector.astype(str)].to_numpy(np.float32)
    ccs = (ccs * np.float32(1000)).astype(np.float32) * np.float32(.001)
    if not np.isfinite(ccs).all():
        raise ValueError('Non-finite TARTS prediction')
    padded[:, MODE_INDEX] = ccs
    tarts = danish.copy()
    tarts['zk_CCS'] = list(ccs)
    tarts['zk_OCS'] = list((padded @ rotation)[:, MODE_INDEX])
    audit = dict(source_pairs=len(source), common_pairs=len(danish),
                 missing_source_pairs=int((~matched).sum()), predicted_pairs=len(pairs),
                 rotation_error_um=gate, rotTelPos_rad=theta)
    return danish, tarts, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--author-dir', type=Path, required=True,
                        help='Danish MIW extraction containing donuts.parquet')
    parser.add_argument('--rotation-dir', type=Path, required=True,
                        help='Raw-AOS rotation_<visit>.json records containing rotTelPos in radians')
    parser.add_argument('--predictions-dir', type=Path)
    parser.add_argument('--visits-file', type=Path, default=Path(__file__).with_name('visits.csv'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cfg = settings(args.config)
    visits = pd.read_csv(args.visits_file).visit.astype(int).tolist()
    if len(visits) != len(set(visits)):
        raise ValueError('Duplicate visits')
    predictions = args.predictions_dir or cfg['output_root'] / 'predictions/full_iband_dare'
    out = args.output_dir or cfg['output_root'] / 'miw_analysis/full_iband_dare/inputs'
    for visit in visits:
        if not (predictions / f'visit_{visit}.npz').is_file():
            raise FileNotFoundError(f'Missing prediction: {visit}')
    observing = pd.read_csv(cfg['miw']['observations_file']).set_index('visit', drop=False).loc[visits]
    out.mkdir(parents=True, exist_ok=False)
    audits, writers = [], {}
    with ExitStack() as stack:
        for branch in ('danish', 'tarts'):
            (out / branch).mkdir()
        for visit in visits:
            source = pq.read_table(args.author_dir / 'donuts.parquet', columns=COLUMNS,
                                  filters=[('day_obs', '=', visit // 100000),
                                           ('seq_num', '=', visit % 100000)]).to_pandas()
            rotation = json.loads((args.rotation_dir / f'rotation_{visit}.json').read_text())
            assert int(rotation['visit']) == visit
            with np.load(predictions / f'visit_{visit}.npz', allow_pickle=False) as prediction:
                assert int(prediction['visit']) == visit
                danish, tarts, audit = align_visit(source, prediction, float(rotation['rotTelPos']))
            for branch, frame in [('danish', danish), ('tarts', tarts)]:
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if branch not in writers:
                    writers[branch] = stack.enter_context(pq.ParquetWriter(out / branch / 'donuts.parquet', table.schema))
                writers[branch].write_table(table, row_group_size=len(frame))
            audits.append(dict(visit=visit, **audit))
            print(f'visit {visit}: {len(danish)} common pairs', flush=True)
    for branch in ('danish', 'tarts'):
        meta = Table.from_pandas(observing.reset_index(drop=True))
        meta.meta.update(nollIndices=NOLL, selection='matched Danish native-used pairs',
                         frame='OCS', unit='micron')
        meta.write(out / branch / 'visits.parquet', format='parquet')
    pd.DataFrame(audits).to_csv(out / 'pair_alignment.csv', index=False)
    record = dict(author_dir=str(args.author_dir), rotation_dir=str(args.rotation_dir),
                  predictions_dir=str(predictions), visits=len(visits),
                  pairs=sum(r['common_pairs'] for r in audits),
                  dropped_source_pairs=sum(r['missing_source_pairs'] for r in audits),
                  max_rotation_error_um=max(r['rotation_error_um'] for r in audits),
                  selection='Danish native-used pair IDs intersected with current prediction pair IDs',
                  frame='OCS', unit='micron', noll=NOLL)
    (out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
