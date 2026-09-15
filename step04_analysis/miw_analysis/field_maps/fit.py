#!/usr/bin/env python3
"""Run the existing MIW build, camera/telescope split and sidecar commands."""

import argparse
from copy import deepcopy
from pathlib import Path
import runpy
import sys
import shutil

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import yaml
from step04_analysis.config import ROOT, settings


def commands(output, config_path, windows, branches, name, height_map_dir):
    binary = ROOT / "external_repo/ts_intrinsic_wavefront/bin.src"
    for branch in branches:
        common = ["--param-set", branch, "--mi-name", name, "--config", str(config_path),
                  "--output-root", str(output)]
        for lo, hi in windows:
            yield binary / "run_build_intrinsic.py", [*common, "--rot-min", str(lo), "--rot-max", str(hi)]
        yield binary / "run_intrinsic_split.py", common
        yield binary / "run_make_intrinsic_sidecar.py", [*common, "--height-map-dir", str(height_map_dir)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--miw-config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--branch", choices=("danish", "tarts"), action="append")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    cfg = settings(args.config)
    root = cfg["output_root"] / "miw_analysis" / args.variant
    source = args.input_dir or root / "inputs"
    output = args.output_dir or root / "field_maps"
    branches = args.branch or ["danish", "tarts"]
    model = yaml.safe_load(args.miw_config.read_text())
    model = deepcopy(model)
    model["defaults"]["build"]["batoid_rubin_height_map_dir"] = str(cfg["miw"]["height_map_dir"])
    model["defaults"]["build"]["ofc_normalization_yaml"] = str(cfg["miw"]["normalization_file"])
    resolved = output / "config.yaml"
    if not args.print_commands:
        output.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(model, sort_keys=False))
        for branch in branches:
            folder = output / branch
            folder.mkdir(exist_ok=True)
            for name in ("donuts.parquet", "visits.parquet"):
                target = folder / name
                if not target.exists():
                    target.symlink_to((source / branch / name).resolve())
    for script, arguments in commands(output, resolved,
                                     model["defaults"]["rotator_bins"], branches,
                                     "pathA_50_34_i_5rot", cfg["miw"]["height_map_dir"]):
        print("python", script, *arguments, flush=True)
        if not args.print_commands:
            sys.argv = [str(script), *arguments]
            runpy.run_path(str(script), run_name="__main__")
    if not args.print_commands:
        suffix = "pathA_50_34_i_5rot/intrinsic_split.pdf"
        for branch in branches:
            shutil.copy2(output / branch / suffix, output / f"{branch}_intrinsic_split.pdf")


if __name__ == "__main__":
    main()
