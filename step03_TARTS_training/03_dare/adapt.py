#!/usr/bin/env python3
"""Adapt one network with DARE-GRAM using simulation and real images."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.training import dare_configs, default_config, run_training
from real_targets import aggregator_targets, real_identity, visit_plan, wave_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config("dare"))
    parser.add_argument("--stage", choices=("wavenet", "aggregator"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="Starting network checkpoint")
    parser.add_argument("--wavenet", type=Path, help="Frozen WaveNet for Aggregator inputs")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config, models = dare_configs(args.config)
    model = models[args.stage]
    if args.checkpoint:
        model["initial_checkpoint"] = str(args.checkpoint.resolve())
    if args.wavenet:
        if args.stage != "aggregator":
            parser.error("--wavenet is an Aggregator input")
        model["wave_checkpoint"] = str(args.wavenet.resolve())
    rows = visit_plan(config["real_target_root"], config["heldout_manifest"])
    identity = real_identity(Path(config["real_target_root"]), rows, config["heldout_manifest"])
    model["target_identity"] = identity
    if args.stage == "wavenet":
        run_training("wavenet", model,
                     lambda: wave_targets(config["real_target_root"], rows), args.resume)
    else:
        aggregate = model
        run_training("aggregator", aggregate,
                     lambda: aggregator_targets(rows, aggregate["wave_checkpoint"],
                                                Path(aggregate["output_dir"]) / "cache",
                                                torch.device(aggregate["device"]),
                                                aggregate["wave_precision"], aggregate["wave_batch_size"]),
                     args.resume)


if __name__ == "__main__":
    main()
