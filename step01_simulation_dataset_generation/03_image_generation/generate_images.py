#!/usr/bin/env python
"""Render one FAM exposure from a selected state and its CCD-center truth."""

from __future__ import annotations

import argparse
import ast
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


def canonical_sha256(payload) -> str:
    """Identify the optical configuration shared with the truth pass."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_config_for_state(config_path: Path, states_path: Path, state_index: int):
    with states_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["state_index"]) == state_index]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for state {state_index}, found {len(rows)}")
    state = rows[0]
    config = yaml.safe_load(config_path.read_text())
    config["eval_variables"].update(
        cboresight={"type": "RADec", "ra": f"{state['ra_deg']} deg",
                    "dec": f"{state['dec_deg']} deg"},
        sband=state["band"], azenith=f"{state['zenith_deg']} deg",
        artp=f"{state['rtp_deg']} deg", fmjd=float(state["mjd"]),
        frawSeeing=float(state["seeing"]), iseqnum=int(state["seqid"]),
        ldofs=json.loads(state["dof_json"]),
    )
    config["input"]["telescope"]["file_name"] = f"LSST_{state['band']}.yaml"
    return config, state


def expression_tree(value):
    """Compare GalSim expressions without treating whitespace as physics."""
    if isinstance(value, dict):
        return {key: expression_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expression_tree(item) for item in value]
    if isinstance(value, str) and value.startswith("$"):
        return ast.dump(ast.parse(value[1:], mode="eval"))
    return value


def read_truth_model(truth: dict) -> dict:
    """Read the stored optical inputs, or the author's hash-verified YAML."""
    if "model_config" in truth:
        model = truth["model_config"]
        identity = {
            "state_parameters_sha256": canonical_sha256({
                "eval_variables": model["eval_variables"], "bandpass": model["bandpass"],
            }),
            "telescope_config_without_focusZ": model["telescope"],
        }
    else:
        source = truth["author_config"]
        content = Path(source["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("The author's telescope configuration has changed")
        author = yaml.safe_load(content)
        telescope = deepcopy(author["input"]["telescope"])
        telescope.pop("focusZ", None)
        for key, asset in (("fea_dir", "fea"), ("bend_dir", "bend")):
            telescope["fea"][key] = truth["inputs"][asset]["path"]
        model = {"eval_variables": author["eval_variables"],
                 "bandpass": author["image"]["bandpass"], "telescope": telescope}
        identity = {
            "author_config_sha256": source["sha256"],
            "telescope_config_without_focusZ": telescope,
            "fea_manifest_sha256": truth["inputs"]["fea"]["manifest_sha256"],
            "bend_manifest_sha256": truth["inputs"]["bend"]["manifest_sha256"],
            "imsim_commit": truth["software"]["imsim"]["commit"],
        }
    if canonical_sha256(identity) != truth["physics"]["base_telescope_state_sha256"]:
        raise ValueError("Truth optical inputs do not match its recorded model identity")
    return model


def prepare_exposure(config: dict, state: dict, *, label_manifest: Path,
                     fam_side: str, output_dir: Path, nproc: int,
                     checkpoint_dir: Path | None = None,
                     nbatch_per_checkpoint: int = 8) -> tuple[dict, dict]:
    """Bind the shared optical state, then add only the selected FAM exposure."""
    from imsim.camera import get_camera

    config = deepcopy(config)
    label_bytes = label_manifest.read_bytes()
    truth = json.loads(label_bytes)
    selection, physics = truth["selection"], truth["physics"]
    detector_ids = [int(value) for value in selection["detector_ids"]]
    camera_name = config["output"]["camera"]
    camera = get_camera(camera_name)
    detector_names = [camera[value].getName() for value in detector_ids]
    if not detector_ids or len(set(detector_ids)) != len(detector_ids):
        raise ValueError("Truth must specify a nonempty list of unique CCDs")
    if camera_name != selection["camera"] or detector_names != [
        record["detector_name"] for record in selection["detectors"]
    ]:
        raise ValueError("Image and truth camera/detector geometry differ")
    if int(state["state_index"]) != truth.get("state_index", int(state["state_index"])):
        raise ValueError("Image and truth state indices differ")
    if physics["label_focus_m"] != 0.0 or not physics["fam_excluded_from_label"]:
        raise ValueError("Truth must exclude the added FAM defocus")
    if len(config["eval_variables"]["ldofs"]) != 50:
        raise ValueError("Expected 50 telescope DOFs")

    # Compare the base state before changing sequence number or adding FAM focus.
    telescope = deepcopy(config["input"]["telescope"])
    telescope.pop("focusZ", None)
    image_model = {"eval_variables": config["eval_variables"],
                   "bandpass": config["image"]["bandpass"], "telescope": telescope}
    if expression_tree(image_model) != expression_tree(read_truth_model(truth)):
        raise ValueError("Image and truth use different base telescope configurations")
    model_sha = physics["base_telescope_state_sha256"]

    focus_mm = float(state[f"{fam_side}_focus_mm"])
    seqnum = int(state[f"{fam_side}_seqnum"])
    if not (focus_mm < 0 if fam_side == "intra" else focus_mm > 0):
        raise ValueError("FAM intra requires negative focus; extra requires positive focus")
    if not 1 <= nproc <= len(detector_ids) or seqnum < 1 or nbatch_per_checkpoint < 1:
        raise ValueError("Invalid parallelism, exposure sequence or checkpoint interval")

    config["eval_variables"]["iseqnum"] = seqnum
    config["input"]["telescope"]["focusZ"] = focus_mm / 1000.0
    config["image"]["nbatch_per_checkpoint"] = nbatch_per_checkpoint
    if checkpoint_dir is not None:
        config["input"]["checkpoint"] = {
            "dir": str(checkpoint_dir.expanduser().resolve()),
            "file_name": {
                "type": "FormattedStr",
                "format": f"checkpoint_seq{seqnum:06d}_%s.hdf",
                "items": ["$det_name"],
            },
        }

    output = config["output"]
    if "opd" in output:
        raise ValueError("Use truth.yaml for OPD and image.yaml for CCD rendering")
    output.update(det_num={"type": "List", "items": detector_ids},
                  nfiles=len(detector_ids), nproc=nproc, dir=str(output_dir))
    output["header"].update({
        "focusZ": focus_mm, "reason": f"fam_{fam_side}_simulation",
        "altitude": "$altitude.deg", "azimuth": "$azimuth.deg",
        "GROUPID": state["group_id"],
    })
    output["readout"].setdefault("added_keywords", {})["GROUPID"] = state["group_id"]

    # This small record is consumed directly by the stamp-label attachment step.
    association = {
        "state_index": int(state["state_index"]),
        "camera": camera_name,
        "detector_ids": detector_ids,
        "detector_names": detector_names,
        "seqnum": seqnum,
        "fam_side": fam_side,
        "focus_m": focus_mm / 1000.0,
        "focus_header_mm": focus_mm,
        "group_id": state["group_id"],
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "base_telescope_state_sha256": model_sha,
    }
    return config, association


def render(config_path: Path, verbosity: int = 2) -> None:
    """Run the GalSim CLI in-process, using the installed frozen IERS table."""
    from astropy.utils import iers
    from galsim.main import main as galsim_main

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    galsim_main(["-v", str(verbosity), "-x", "-n", "1", "-j", "1", str(config_path)])


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "image.yaml")
    parser.add_argument("--states", type=Path, default=here.parent / "output/selection/states.csv")
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--label-dir", type=Path,
                        help="Default: output/states/state_NNN/truth under this step")
    parser.add_argument("--fam-side", choices=("intra", "extra"), required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: output/states/state_NNN/intra or extra under this step")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--sky-config", type=Path)
    parser.add_argument("--tree-rings", type=Path)
    parser.add_argument("--fea-dir", type=Path)
    parser.add_argument("--bend-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--nbatch-per-checkpoint", type=int, default=8)
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write the exposure YAML and label association without rendering")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_output = Path(__file__).resolve().parents[1] / "output/states" / f"state_{args.state_index:03d}"
    args.label_dir = args.label_dir or state_output / "truth"
    args.output_dir = args.output_dir or state_output / args.fam_side
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    config_path = args.config.expanduser().resolve()
    config, state = load_config_for_state(
        config_path, args.states.expanduser().resolve(), args.state_index,
    )
    sky = args.sky_config or config_path.parent / config["input"]["sky_catalog"]["file_name"]
    config["input"]["sky_catalog"]["file_name"] = str(sky.expanduser().resolve())
    if args.tree_rings is not None:
        config["input"]["tree_rings"]["file_name"] = str(args.tree_rings.expanduser().resolve())
    fea = config["input"]["telescope"]["fea"]
    for key in ("fea_dir", "bend_dir"):
        path = getattr(args, key) or Path(fea[key])
        fea[key] = str(path.expanduser().resolve())

    output_dir = args.output_dir.expanduser().resolve()
    config, association = prepare_exposure(
        config, state, label_manifest=args.label_dir.expanduser().resolve() / "manifest.json",
        fam_side=args.fam_side, output_dir=output_dir, nproc=args.nproc,
        checkpoint_dir=args.checkpoint_dir, nbatch_per_checkpoint=args.nbatch_per_checkpoint,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    generated = output_dir / "image_pass.yaml"
    generated.write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "image_pass.yaml.manifest.json").write_text(
        json.dumps(association, indent=2) + "\n",
    )
    print(f"State {args.state_index}, {args.fam_side}, {len(association['detector_ids'])} CCDs: {generated}", flush=True)
    if not args.prepare_only:
        render(generated, args.verbosity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
