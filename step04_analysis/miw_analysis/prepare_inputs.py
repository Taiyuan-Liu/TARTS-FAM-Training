#!/usr/bin/env python3
"""Convert TARTS predictions and source AOS tables into MIW input Parquet."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table
from step04_analysis.config import NOLL, settings
from step04_analysis.miw_analysis.aos_tables import broadcast_prediction, build_visit_tables


def extract_visit(raw, visit, observing):
    """Equivalent MIW mktable selection; no new wavefront fit or rotation."""
    if int(raw.meta["visit"]) != visit or list(raw.meta["nollIndices"]) != NOLL:
        raise ValueError("AOS visit or Noll ordering mismatch")
    used = np.asarray(raw["used"], bool)
    data = raw[used]
    frame = pd.DataFrame({name: list(data[name]) if np.asarray(data[name]).ndim > 1
                         else np.asarray(data[name]) for name in data.colnames})
    frame["day_obs"], frame["seq_num"] = visit // 100000, visit % 100000
    blur = np.asarray(raw.meta["estimatorInfo"]["fwhm"], float)
    frame["blur"] = blur[used] if len(blur) == len(raw) else blur
    for axis in ("x", "y"):
        frame[f"intra_extra_offset_{axis}_arcsec"] = np.abs(
            frame[f"th{axis}_OCS_intra"] - frame[f"th{axis}_OCS_extra"]) * 206265
    frame["matched_intra_extra"] = True  # Original MIW extraction disables this cut.
    counts = frame.groupby("detector").size()
    meta = {key: raw.meta[key] for key in ("visit", "ra", "dec", "az", "alt", "band", "mjd")}
    meta.update(day_obs=visit // 100000, seq_num=visit % 100000,
                skyAngle=raw.meta["rotAngle"], nollIndices=NOLL,
                n_donuts=len(frame), n_detectors=len(counts),
                n_detectors_with_min_donuts=int((counts >= 3).sum()),
                median_blur_arcsec=float(np.nanmedian(frame.blur)) if len(frame) else np.nan,
                rotator_angle=float(observing["rotator_angle"]),
                science_program=str(observing["science_program"]))
    return frame, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, help="Source visit_<id>/{raw,avg}.parquet directory")
    parser.add_argument("--from-butler", action="store_true")
    parser.add_argument("--selection", choices=("native", "matched_danish"), default="native")
    parser.add_argument("--visits-file", type=Path)
    parser.add_argument("--visit", type=int, action="append")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = settings(args.config)
    if args.visit:
        visits = args.visit
    else:
        visits = pd.read_csv(args.visits_file or cfg["visits_file"])["visit"].astype("int64").tolist()
        if not visits or len(set(visits)) != len(visits):
            raise ValueError("Expected a nonempty, unique visit list")
    observing = pd.read_csv(cfg["miw"]["observations_file"]).set_index("visit")
    source = args.source_dir or cfg["aos_source_dir"]
    predictions = args.predictions_dir or cfg["output_root"] / "predictions" / args.variant
    out = args.output_dir or cfg["output_root"] / "miw_analysis" / args.variant / "inputs"
    if args.from_butler:
        from lsst.daf.butler import Butler
        butler = Butler(cfg["butler"]["repo"], collections=[cfg["butler"]["source_collection"]],
                        instrument="LSSTCam", writeable=False)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {branch: [] for branch in ("danish", "tarts")}
    writers = {}
    for branch in ("danish", "tarts"):
        (out / branch).mkdir()
    with ExitStack() as stack:
        for visit in visits:
            if args.from_butler:
                raw, avg = [butler.get(f"aggregateAOSVisitTable{kind}", visit=visit,
                                      instrument="LSSTCam") for kind in ("Raw", "Avg")]
            else:
                raw, avg = [Table.read(source / f"visit_{visit}" / f"{name}.parquet")
                            for name in ("raw", "avg")]
            with np.load(predictions / f"visit_{visit}.npz", allow_pickle=False) as prediction:
                coefficients, blur = broadcast_prediction(raw, prediction)
            tarts = build_visit_tables(raw, avg, coefficients, blur, args.selection)
            for branch, tables in (("danish", (raw, avg)), ("tarts", tarts)):
                folder = out / "aos" / branch / f"visit_{visit}"
                folder.mkdir(parents=True)
                for name, table in zip(("raw", "avg"), tables):
                    table.write(folder / f"{name}.parquet", format="parquet")
                frame, meta = extract_visit(tables[0], visit, observing.loc[visit])
                if frame.empty:
                    raise ValueError(f"No used {branch} rows for visit {visit}")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if branch not in writers:
                    writers[branch] = stack.enter_context(
                        pq.ParquetWriter(out / branch / "donuts.parquet", table.schema))
                writers[branch].write_table(table, row_group_size=len(frame))
                metadata[branch].append(meta)
            print(f"visit {visit}: Danish {metadata['danish'][-1]['n_donuts']} pairs; "
                  f"TARTS {metadata['tarts'][-1]['n_donuts']} pairs", flush=True)
    for branch in ("danish", "tarts"):
        table = Table(rows=metadata[branch])
        table.meta.update(min_donuts_per_detector=3, matched_threshold_arcsec=None)
        table.write(out / branch / "visits.parquet", format="parquet")


if __name__ == "__main__":
    main()
