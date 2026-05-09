"""Time-axis resampling primitives."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import xarray as xr


_ALLOWED_RESAMPLE_METHODS = frozenset(
    {"mean", "sum", "max", "min", "median", "std", "var", "first", "last", "count"}
)


def resample_time(
    ds: xr.Dataset | xr.DataArray,
    freq: str = "1D",
    method: str = "mean",
    *,
    time: str = "time",
    interp_method: Literal["linear", "nearest", "cubic"] = "linear",
) -> xr.Dataset | xr.DataArray:
    """Resample along the time axis via xarray's built-in resampler.

    Args:
        ds: Input.
        freq: Pandas-style frequency string (e.g. ``"1D"``, ``"6H"``,
            ``"1M"``).
        method: Zero-argument aggregation method; one of ``"mean"``,
            ``"sum"``, ``"max"``, ``"min"``, ``"median"``, ``"std"``,
            ``"var"``, ``"first"``, ``"last"``, ``"count"``. Methods
            that require additional arguments (e.g. ``reduce``, ``quantile``)
            are intentionally not exposed here. Use ``"interpolate"`` for
            upsampling via xarray's resampler interpolation.
        time: Name of the time dimension.
        interp_method: Interpolation method forwarded to xarray when
            ``method="interpolate"``.

    Returns:
        Resampled container.
    """
    resampler = ds.resample({time: freq})
    if method == "interpolate":
        if _target_is_coarser_than_source(ds, freq=freq, time=time):
            raise ValueError(
                "resample_time(method='interpolate') only supports upsampling; "
                f"got target frequency {freq!r} coarser than the input."
            )
        return resampler.interpolate(interp_method)

    if method not in _ALLOWED_RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown resample method {method!r}; expected one of "
            f"{sorted(_ALLOWED_RESAMPLE_METHODS | {'interpolate'})}."
        )
    return getattr(resampler, method)()


def _target_is_coarser_than_source(
    ds: xr.Dataset | xr.DataArray, *, freq: str, time: str
) -> bool:
    coord = ds[time]
    if coord.size < 2:
        return False

    try:
        values_ns = np.asarray(coord.values, dtype="datetime64[ns]").astype("int64")
        deltas = np.diff(values_ns)
        positive_deltas = deltas[deltas > 0]
        if positive_deltas.size == 0:
            return False
        source_delta_ns = int(np.median(positive_deltas))
        start = pd.Timestamp(coord.values[0])
        if pd.isna(start):
            return False
        target_delta_ns = _target_delta_ns(freq, cast(pd.Timestamp, start))
    except (TypeError, ValueError):
        return False

    return target_delta_ns > source_delta_ns


def _target_delta_ns(freq: str, start: pd.Timestamp) -> int:
    offset = pd.tseries.frequencies.to_offset(freq)
    try:
        return int(offset.nanos)
    except ValueError:
        return int((start + offset - start).value)
