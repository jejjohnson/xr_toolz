"""Shared helpers for V4.3 budget residuals."""

from __future__ import annotations

import numpy as np
import xarray as xr

from xr_toolz import calc


def _time_derivative(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    """``∂da/∂t`` in per-second units.

    For datetime-typed coordinates uses
    :meth:`xr.DataArray.differentiate` with ``datetime_unit="s"``;
    otherwise treats the coordinate as already in seconds.
    """
    if time_dim not in da.dims:
        return xr.zeros_like(da)
    coord = da[time_dim]
    if np.issubdtype(coord.dtype, np.datetime64):
        return da.differentiate(time_dim, datetime_unit="s")
    return da.differentiate(time_dim)


def _flux_divergence(
    tracer: xr.DataArray,
    *,
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray | None,
    lat: str,
    lon: str,
    depth: str | None,
) -> xr.DataArray:
    """``∇·(u φ)`` using spherical horizontal + rectilinear vertical."""
    flux_u = u * tracer
    flux_v = v * tracer
    div = calc.partial(
        flux_u, lon, geometry="spherical", lon=lon, lat=lat
    ) + calc.partial(flux_v, lat, geometry="spherical", lon=lon, lat=lat)
    if w is not None and depth is not None and depth in tracer.dims:
        flux_w = w * tracer
        div = div + calc.partial(flux_w, depth, geometry="rectilinear")
    return div


def _tracer_budget_residual(
    ds: xr.Dataset,
    *,
    tracer_var: str,
    u_var: str,
    v_var: str,
    w_var: str | None,
    surface_flux_var: str | None,
    time_dim: str,
    lat: str,
    lon: str,
    depth: str | None,
) -> xr.DataArray:
    """Return ``∂φ/∂t + ∇·(u φ) - F_surface`` as a per-cell residual.

    A closed budget has ≈ 0; deviations measure the imbalance the
    model leaves un-resolved.
    """
    phi = ds[tracer_var]
    u = ds[u_var]
    v = ds[v_var]
    w = ds[w_var] if w_var is not None else None
    tendency = _time_derivative(phi, time_dim)
    flux_div = _flux_divergence(phi, u=u, v=v, w=w, lat=lat, lon=lon, depth=depth)
    res = tendency + flux_div
    if surface_flux_var is not None:
        res = res - ds[surface_flux_var]
    return res.rename(f"{tracer_var}_budget_residual")
