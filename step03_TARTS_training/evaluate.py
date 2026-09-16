#!/usr/bin/env python3
"""Compare a WaveNet/Aggregator pair on identical test groups and stamps."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from common.evaluation import evaluate_pair
from common.training import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("supervised", "dare"), default="supervised")
    parser.add_argument("--config", type=Path, help="Aggregator config, or DARE config for --variant dare")
    parser.add_argument("--wavenet", type=Path, help="Frozen WaveNet checkpoint")
    parser.add_argument("--aggregator", type=Path, help="Matching Aggregator checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--zoom-quantile", type=float, default=0.002)
    args = parser.parse_args()
    if not 0 <= args.zoom_quantile < 0.5:
        parser.error("zoom-quantile must be in [0,0.5)")
    configs = {kind: model_config(kind, args.variant,
               args.config if kind == "aggregator" or args.variant == "dare" else None)
               for kind in ("wavenet", "aggregator")}
    evaluate_pair(configs, args.variant, args.wavenet, args.aggregator,
                  args.device, args.zoom_quantile)


if __name__ == "__main__":
    main()
