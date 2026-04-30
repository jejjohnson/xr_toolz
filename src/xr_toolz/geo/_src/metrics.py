"""Deprecated — moved to :mod:`xr_toolz.metrics`.

This module re-exports the pixel and spectral metrics from their new
home for one release, with a :class:`DeprecationWarning` on import.
Schedule removal in the next minor release.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "xr_toolz.geo._src.metrics is deprecated; import from "
    "xr_toolz.metrics (pixel + spectral submodules) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from xr_toolz.metrics._src.pixel import (
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    r2_score,
    rmse,
)
from xr_toolz.metrics._src.spectral import (
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
)


__all__ = [
    "bias",
    "correlation",
    "find_intercept_1D",
    "mae",
    "mse",
    "nrmse",
    "psd_error",
    "psd_score",
    "r2_score",
    "resolved_scale",
    "rmse",
]
