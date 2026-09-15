#!/usr/bin/env python
"""Fit annular Zernike coefficients from imSim OPD pixels.

Fit Z1--Z28 jointly, then store Z4--Z28 in micrometres.
The fit uses OPD pixels, not the AZ_* FITS header coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import galsim
import numpy as np
from astropy.io import fits


def fit_opd_array(
    opd_nm: np.ndarray,
    *,
    noll_fit_max: int = 28,
    noll_store_min: int = 4,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Return one ``(1, 25)`` float32 Z4--Z28 vector in micrometres."""
    opd_nm = np.asarray(opd_nm)
    if opd_nm.ndim != 2 or opd_nm.shape[0] != opd_nm.shape[1]:
        raise ValueError(f"OPD must be a square 2-D array, got {opd_nm.shape}")
    if not np.issubdtype(opd_nm.dtype, np.number):
        raise TypeError(f"OPD must be numeric, got {opd_nm.dtype}")
    if noll_fit_max < noll_store_min or noll_store_min < 1:
        raise ValueError("Invalid Noll range")
    if not 0.0 <= obscuration < 1.0:
        raise ValueError("obscuration must be in [0, 1)")

    grid = np.linspace(-1.0, 1.0, opd_nm.shape[0])
    opd_x, opd_y = np.meshgrid(grid, grid)
    valid = ~np.isnan(opd_nm)
    if not np.any(valid) or not np.all(np.isfinite(opd_nm[valid])):
        raise ValueError("OPD contains no finite, unmasked fit pixels")

    basis = galsim.zernike.zernikeBasis(
        int(noll_fit_max),
        opd_x[valid],
        opd_y[valid],
        R_inner=float(obscuration),
    )[1:]
    coefficients_nm, *_ = np.linalg.lstsq(
        basis.T, opd_nm[valid], rcond=-1
    )
    selected_nm = coefficients_nm[noll_store_min - 1 : noll_fit_max]
    expected = noll_fit_max - noll_store_min + 1
    if selected_nm.shape != (expected,) or not np.all(np.isfinite(selected_nm)):
        raise ValueError("OPD fit returned invalid coefficients")
    return (selected_nm / 1000.0).astype(np.float32).reshape(1, expected)


def fit_opd_fits(
    opd_path: Path,
    *,
    noll_fit_max: int = 28,
    noll_store_min: int = 4,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Fit all image HDUs in order and return ``(n_hdu, n_zernike)``."""
    opd_path = Path(opd_path).expanduser().resolve()
    if not opd_path.is_file():
        raise FileNotFoundError(opd_path)
    fitted: list[np.ndarray] = []
    with fits.open(opd_path, memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            if hdu.data is None:
                raise ValueError(f"OPD HDU {index} contains no image")
            fitted.append(
                fit_opd_array(
                    hdu.data,
                    noll_fit_max=noll_fit_max,
                    noll_store_min=noll_store_min,
                    obscuration=obscuration,
                )[0]
            )
    if not fitted:
        raise ValueError(f"No OPD image HDUs found in {opd_path}")
    return np.asarray(fitted, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opd", required=True, type=Path)
    parser.add_argument("--output-npy", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fitted = fit_opd_fits(args.opd)
    output = args.output_npy.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, fitted, allow_pickle=False)
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "shape": list(fitted.shape),
        "dtype": str(fitted.dtype),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
