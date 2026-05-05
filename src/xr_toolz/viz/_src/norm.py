"""Shared (vmin, vmax) helper for matched colour scales across panels."""

from __future__ import annotations

import numpy as np
import xarray as xr


def shared_norm(
    *arrays: xr.DataArray | xr.Dataset,
    q: tuple[float, float] | None = (0.02, 0.98),
    symmetric: bool = False,
) -> tuple[float, float]:
    """Compute matched ``(vmin, vmax)`` across multiple inputs.

    Useful for multi-panel comparison grids where the eye should not
    pick up colour-scale stretch artefacts as if they were structural
    differences in the data.

    Args:
        *arrays: One or more :class:`xr.DataArray` or
            :class:`xr.Dataset`. A single-variable Dataset is auto
            unwrapped; a multi-variable Dataset raises.
        q: ``(low, high)`` quantile pair in ``[0, 1]``. Default
            ``(0.02, 0.98)`` strips outliers. Pass ``None`` for the
            full ``(min, max)`` range.
        symmetric: When ``True``, return symmetric limits ``(-M, +M)``
            with ``M = max(|low|, |high|)``. Useful for divergent
            (signed-error) fields.

    Returns:
        ``(vmin, vmax)`` as Python floats. NaNs are ignored. Returns
        ``(nan, nan)`` if every input is fully NaN.
    """
    if not arrays:
        raise ValueError("shared_norm requires at least one input array.")
    if q is not None and not (0.0 <= q[0] <= q[1] <= 1.0):
        raise ValueError(f"q must satisfy 0 <= q[0] <= q[1] <= 1, got {q!r}.")

    flats: list[np.ndarray] = []
    for a in arrays:
        if isinstance(a, xr.Dataset):
            if len(a.data_vars) != 1:
                raise ValueError(
                    "shared_norm: Dataset inputs must have exactly one "
                    f"data variable; got {list(a.data_vars)}."
                )
            (var,) = a.data_vars
            arr = a[var].values
        else:
            arr = a.values
        flats.append(np.asarray(arr).ravel())
    flat = np.concatenate(flats)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return (float("nan"), float("nan"))

    if q is None:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = (float(x) for x in np.quantile(finite, q))

    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    return (lo, hi)


__all__ = ["shared_norm"]
