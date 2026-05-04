"""Tier A — array kernels for value-preserving smoothers (D11, D12).

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point. These kernels operate on raw :class:`numpy.ndarray`
inputs with an explicit ``axis``, returning an array of the same shape
(no reduction).

Backend: numpy + scipy. JAX / CuPy variants are out of scope for the
F3.3 pilot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


def moving_average(
    arr: ArrayLike,
    *,
    axis: int = -1,
    window: int,
    center: bool = True,
    min_periods: int | None = None,
) -> NDArray[np.floating]:
    """Sliding-window mean along ``axis``. NaN-skipping.

    Parameters
    ----------
    arr
        Input array (any shape).
    axis
        Axis to smooth along.
    window
        Window length (number of samples).
    center
        If True, the window is centered on the output sample; otherwise
        trailing.
    min_periods
        Minimum number of non-NaN samples required inside the window
        for the output to be non-NaN. Defaults to ``window``.

    Returns
    -------
    NDArray
        Smoothed array, same shape as the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is None:
        min_periods = window

    a = np.asarray(arr, dtype=float)
    moved = np.moveaxis(a, axis, -1)
    if center:
        pad_left = (window - 1) // 2
        pad_right = window - 1 - pad_left
    else:
        pad_left = window - 1
        pad_right = 0

    pad_widths = [(0, 0)] * moved.ndim
    pad_widths[-1] = (pad_left, pad_right)
    padded = np.pad(moved, pad_widths, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1)

    valid = (~np.isnan(windows)).sum(axis=-1)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(windows, axis=-1)
    out = np.where(valid >= min_periods, means, np.nan)
    return np.moveaxis(out, -1, axis)


def gaussian_smooth(
    arr: ArrayLike,
    *,
    axis: int = -1,
    sigma: float,
    truncate: float = 4.0,
) -> NDArray[np.floating]:
    """Gaussian convolution along ``axis`` with standard deviation ``sigma``.

    Delegates to :func:`scipy.ndimage.gaussian_filter1d`. NaN inputs
    propagate.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    a = np.asarray(arr, dtype=float)
    return gaussian_filter1d(
        a, sigma=sigma, axis=axis, truncate=truncate, mode="reflect"
    )


def lowpass_filter(
    arr: ArrayLike,
    *,
    axis: int = -1,
    cutoff: float,
    order: int = 4,
    btype: str = "low",
) -> NDArray[np.floating]:
    """Zero-phase Butterworth filter along ``axis``.

    ``cutoff`` is the normalized critical frequency (fraction of the
    Nyquist rate); values must lie in ``(0, 1)``. ``btype`` is forwarded
    to :func:`scipy.signal.butter` (``"low"``, ``"high"``, ``"band"``,
    ``"bandstop"``).

    The filter is applied with :func:`scipy.signal.sosfiltfilt` for zero
    phase and SOS-form numerical stability.
    """
    a = np.asarray(arr, dtype=float)
    sos = butter(order, cutoff, btype=btype, output="sos")
    return sosfiltfilt(sos, a, axis=axis)


__all__ = [
    "gaussian_smooth",
    "lowpass_filter",
    "moving_average",
]
