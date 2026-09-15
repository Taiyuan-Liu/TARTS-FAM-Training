#!/usr/bin/env python
"""Select observing states and balanced science CCDs for a FAM simulation."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.time import Time, TimeDelta


AUTHOR_TABLE = Path(__file__).with_name("production_visits_ldofs_11999.csv")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"
STRATA = (5, 4, 5)


def quantile_bins(values, count):
    edges = np.quantile(values, np.linspace(0, 1, count + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


def select_states(candidates, n_states, rng):
    """Draw equally from DOF-norm, altitude and within-cell rotator strata."""
    n_cells = int(np.prod(STRATA))
    if n_states <= 0 or n_states % n_cells:
        raise ValueError(f"n-states must be a positive multiple of {n_cells}")
    dofs = np.asarray([json.loads(row["DOF"]) for row in candidates], dtype=float)
    altitude = np.asarray([float(row["ALTITUDE"]) for row in candidates])
    rotation = np.asarray([float(row["RTP"]) for row in candidates])
    if dofs.shape != (len(candidates), 50) or not np.isfinite(dofs).all():
        raise ValueError("Each candidate must contain 50 finite DOFs")
    if not np.isfinite(altitude).all() or not np.isfinite(rotation).all():
        raise ValueError("Altitude and rotator angle must be finite")
    if len({int(row["SEQID"]) for row in candidates}) != len(candidates):
        raise ValueError("Candidate SEQIDs must be unique")

    norms = np.linalg.norm(dofs, axis=1)
    dof_bins = quantile_bins(norms, STRATA[0])
    altitude_bins = quantile_bins(altitude, STRATA[1])
    selected = []
    for d in range(STRATA[0]):
        for a in range(STRATA[1]):
            parent = np.flatnonzero((dof_bins == d) & (altitude_bins == a))
            if len(parent) < STRATA[2]:
                raise ValueError(f"Too few candidates in DOF/altitude cell {(d, a)}")
            rotation_bins = quantile_bins(rotation[parent], STRATA[2])
            for r in range(STRATA[2]):
                pool = parent[rotation_bins == r]
                count = n_states // n_cells
                if len(pool) < count:
                    raise ValueError(f"Cell {(d, a, r)} has {len(pool)} states; needs {count}")
                for index in rng.choice(pool, size=count, replace=False):
                    selected.append(dict(
                        candidates[index], dof_norm=float(norms[index]),
                        dof_stratum=d, alt_stratum=a, rtp_stratum=r,
                    ))
    return sorted(selected, key=lambda row: int(row["SEQID"]))


def science_rafts():
    """Read science CCD identities and focal-plane positions from LsstCamSim."""
    from lsst.afw.cameraGeom import FOCAL_PLANE, DetectorType
    from lsst.obs.lsst import LsstCamSim

    rafts = {}
    for detector in LsstCamSim.getCamera():
        if detector.getType() != DetectorType.SCIENCE:
            continue
        raft, sensor = detector.getName().split("_")
        x, y = detector.getCenter(FOCAL_PLANE)
        rafts.setdefault(raft, {})[sensor] = {
            "raft": raft, "sensor": sensor,
            "detector_id": int(detector.getId()),
            "detector_name": detector.getName(),
            "focal_x_mm": round(float(x), 4),
            "focal_y_mm": round(float(y), 4),
            "focal_r_mm": round(float(np.hypot(x, y)), 4),
        }
    return rafts


def assign_detectors(rafts, n_states, rng):
    """Choose one CCD per raft per state, balancing each raft's sensor usage."""
    assignments = [[] for _ in range(n_states)]
    for raft in sorted(rafts):
        sensors = sorted(rafts[raft])
        repeats, remainder = divmod(n_states, len(sensors))
        sequence = sensors * repeats + list(rng.choice(sensors, remainder, replace=False))
        rng.shuffle(sequence)
        for state_index, sensor in enumerate(sequence):
            assignments[state_index].append(rafts[raft][sensor])
    return assignments


def unique_group_ids(states):
    """Give each intra/extra pair a unique TAI time-shaped GROUPID."""
    used = set()
    groups = []
    for state in states:
        time = Time(float(state["MJD"]), format="mjd", scale="tai", precision=3)
        while time.isot in used:
            time += TimeDelta(0.001, format="sec")
        groups.append(time.isot)
        used.add(time.isot)
    return groups


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tables(states, assignments, config_root):
    """Translate the selection into the truth/image passes' input fields."""
    configs = [config_root / f"perturbation_{row['SEQID']}" / f"{row['SEQID']}.yaml"
               for row in states]
    # Config files live on shared storage; hash them concurrently after sampling.
    with ThreadPoolExecutor(max_workers=16) as pool:
        hashes = list(pool.map(sha256, configs))
    groups = unique_group_ids(states)
    state_rows, detector_rows = [], []
    for k, (state, detectors) in enumerate(zip(states, assignments)):
        seqid = int(state["SEQID"])
        state_rows.append({
            "state_index": k, "seqid": seqid, "band": state["BAND"],
            "intra_seqnum": 2 * k + 1, "extra_seqnum": 2 * k + 2,
            "intra_focus_mm": -1.5, "extra_focus_mm": 1.5,
            "group_id": groups[k],
            "rtp_deg": float(state["RTP"]),
            "ra_deg": float(state["RA"]), "dec_deg": float(state["DEC"]),
            "mjd": float(state["MJD"]),
            "altitude_deg": float(state["ALTITUDE"]),
            "azimuth_deg": float(state["AZIMUTH"]),
            "zenith_deg": float(state["ZENITH"]),
            "airmass": float(state["AIRMASS"]), "seeing": float(state["SEEING"]),
            "dof_norm": round(state["dof_norm"], 6),
            "dof_stratum": state["dof_stratum"],
            "alt_stratum": state["alt_stratum"],
            "rtp_stratum": state["rtp_stratum"],
            "n_detectors": len(detectors),
            "detector_ids": ";".join(str(d["detector_id"]) for d in detectors),
            "detector_names": ";".join(d["detector_name"] for d in detectors),
            "author_config": str(configs[k]), "author_config_sha256": hashes[k],
            "dof_json": state["DOF"],
        })
        detector_rows.extend(dict(state_index=k, seqid=seqid, **d) for d in detectors)
    return state_rows, detector_rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-table", type=Path, default=AUTHOR_TABLE)
    parser.add_argument("--author-config-root", type=Path,
                        default=Path("/sdf/data/rubin/shared/aos_group/final_imsim_simulation"))
    parser.add_argument("--band", default="i", choices=list("ugrizy"))
    parser.add_argument("--n-states", type=int, default=1000,
                        help="Number of states, a positive multiple of 100")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "selection")
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("output-dir already exists; choose a new selection directory")

    with args.author_table.open() as stream:
        candidates = [row for row in csv.DictReader(stream) if row["BAND"] == args.band]
    rng = np.random.default_rng(args.seed)
    states = select_states(candidates, args.n_states, rng)
    assignments = assign_detectors(science_rafts(), len(states), rng)
    state_rows, detector_rows = build_tables(
        states, assignments, args.author_config_root.resolve(),
    )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "states.csv", state_rows)
    write_csv(args.output_dir / "state_detectors.csv", detector_rows)
    settings = {
        "band": args.band, "n_states": args.n_states, "seed": args.seed,
        "quantile_strata": dict(zip(("dof_norm", "altitude", "rotation_within_cell"), STRATA)),
        "author_table_sha256": sha256(args.author_table),
        "camera": "LsstCamSim", "rafts_per_state": len(assignments[0]),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(settings, indent=2) + "\n")
    cells = Counter((r["dof_stratum"], r["alt_stratum"], r["rtp_stratum"]) for r in states)
    print(f"Selected {len(states)} {args.band}-band states: {len(cells)} cells, "
          f"{args.n_states // len(cells)} states per cell")
    print(f"Assigned {len(detector_rows)} state/CCD combinations; "
          f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
