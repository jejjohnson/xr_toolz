"""Time-axis resampling primitives."""

from __future__ import annotations

import xarray as xr


def resample_time(
    ds: xr.Dataset | xr.DataArray,
    freq: str = "1D",
    method: str = "mean",
    time: str = "time",
) -> xr.Dataset | xr.DataArray:
    """Resample along the time axis via xarray's built-in resampler.

    Args:
        ds: Input.
        freq: Pandas-style frequency string (e.g. ``"1D"``, ``"6H"``,
            ``"1M"``).
        method: Aggregation method; must be a string method on the
            Resample object (``"mean"``, ``"sum"``, ``"max"``, ...).
        time: Name of the time dimension.

    Returns:
        Resampled container.
    """
    resampler = ds.resample({time: freq})
    if not hasattr(resampler, method):
        raise ValueError(
            f"Unknown resample method {method!r}; expected one of "
            "'mean', 'sum', 'max', 'min', 'median', 'std', 'var', 'first', 'last'."
        )
    return getattr(resampler, method)()
