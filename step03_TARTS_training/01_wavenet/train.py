#!/usr/bin/env python3
"""Train a randomly initialized WaveNet on every labelled FAM stamp."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.training import training_cli

if __name__ == "__main__":
    training_cli("wavenet")
