"""Shared analysis settings and coefficient ordering."""

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
NOLL = list(range(4, 20)) + list(range(22, 27))
MODE_INDEX = np.asarray(NOLL) - 4


def settings(path=None):
    path = Path(path or ROOT / "config.yaml").resolve()
    config = yaml.safe_load(path.read_text())
    for section in (config, config["miw"], *config["models"].values()):
        for key, value in list(section.items()):
            if isinstance(value, str) and (key.endswith(("_dir", "_root", "_file")) or key in ("wavenet", "aggregator")):
                p = Path(value).expanduser()
                section[key] = p if p.is_absolute() else (path.parent / p).absolute()
    return config
