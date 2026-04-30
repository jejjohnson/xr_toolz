"""Tier A — array kernels for pointwise (pixel-level) evaluation metrics.

Pure-array entry points used by Tier B (xarray) wrappers. Signatures
follow D11: ``(prediction, reference, *, axis, **kwargs) -> ndarray``.

Backend: numpy. JAX / CuPy variants are out of scope for the pilot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


Axis = int | tuple[int, ...]


def mse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean squared error along ``axis``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.mean((pred - ref) ** 2, axis=axis)


def rmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Root mean squared error along ``axis``."""
    return np.sqrt(mse(prediction, reference, axis=axis))


def mae(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean absolute error along ``axis``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.mean(np.abs(pred - ref), axis=axis)


def bias(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Mean bias ``<pred - ref>`` along ``axis``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    return np.mean(pred - ref, axis=axis)


def nrmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)`` along ``axis``."""
    err = rmse(prediction, reference, axis=axis)
    ref = np.asarray(reference)
    scale = np.sqrt(np.mean(ref**2, axis=axis))
    return 1.0 - err / scale


def correlation(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Pearson correlation between prediction and reference along ``axis``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    pred_mean = np.mean(pred, axis=axis, keepdims=True)
    ref_mean = np.mean(ref, axis=axis, keepdims=True)
    pred_anom = pred - pred_mean
    ref_anom = ref - ref_mean
    num = np.mean(pred_anom * ref_anom, axis=axis)
    denom = np.sqrt(np.mean(pred_anom**2, axis=axis) * np.mean(ref_anom**2, axis=axis))
    return num / denom


def r2_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    axis: Axis = -1,
) -> NDArray[np.floating]:
    """Coefficient of determination: ``1 - SS_res / SS_tot`` along ``axis``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    ss_res = np.sum((ref - pred) ** 2, axis=axis)
    ref_mean = np.mean(ref, axis=axis, keepdims=True)
    ss_tot = np.sum((ref - ref_mean) ** 2, axis=axis)
    return 1.0 - ss_res / ss_tot


__all__ = [
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "r2_score",
    "rmse",
]
