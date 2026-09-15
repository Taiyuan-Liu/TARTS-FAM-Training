#!/usr/bin/env python3
"""Refresh test plots and metrics in a stage's report.html."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from common.training import evaluate_checkpoint, model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("wavenet", "aggregator"), required=True)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--config", type=Path, help="Model config, or DARE config for --variant dare")
    parser.add_argument("--checkpoint", type=Path, help="Model to evaluate")
    parser.add_argument("--wavenet", type=Path, help="Frozen WaveNet for Aggregator inputs")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--zoom-quantile", type=float, default=0.002)
    args = parser.parse_args()
    if not 0 <= args.zoom_quantile < 0.5:
        parser.error("zoom-quantile must be in [0,0.5)")
    config = model_config(args.kind, args.variant, args.config)
    if args.device:
        config["device"] = args.device
    if args.wavenet:
        config["wave_checkpoint"] = str(args.wavenet.resolve())
    evaluate_checkpoint(args.kind, config, precision=args.precision, zoom_quantile=args.zoom_quantile,
                        checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
