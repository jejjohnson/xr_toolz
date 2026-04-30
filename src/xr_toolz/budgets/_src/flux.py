"""Boundary-flux primitive — V4.2.

Compute area-integrated advective fluxes through the four horizontal
faces of a control volume. The vertical face is included when the user
provides a vertical velocity variable.
"""

from __future__ import annotations

import xarray as xr


def boundary_flux(
    ds: xr.Dataset,
    *,
    variable: str | None,
    velocity_vars: dict[str, str],
    face_metrics: xr.Dataset,
    region: xr.DataArray | None = None,
) -> xr.Dataset:
    """Advective flux of ``variable`` through the boundary faces.

    Computes ``φ · u_n · A_face`` summed over each face direction.

    Args:
        ds: Dataset containing ``variable`` (or ``None`` for volume
            flux only) and the velocity components.
        variable: Tracer name or ``None``. If ``None`` the flux is the
            volume flux ``u_n · A_face`` (used by the volume budget).
        velocity_vars: Mapping ``{"u": "u_var", "v": "v_var", "w": "w_var"}``;
            ``"w"`` is optional. Names are looked up in ``ds``.
        face_metrics: Dataset carrying ``area_e`` (east face area, m²),
            ``area_n`` (north face area, m²), and optionally
            ``area_top``.
        region: Optional boolean mask delimiting the control volume.

    Returns:
        Dataset with one variable per face direction
        (``flux_x``, ``flux_y``, optionally ``flux_z``); each is a
        scalar (or ``time``-indexed) integrated flux.
    """
    out: dict[str, xr.DataArray] = {}
    tracer = ds[variable] if variable is not None else None

    for axis, area_key in (("u", "area_e"), ("v", "area_n"), ("w", "area_top")):
        if axis not in velocity_vars:
            continue
        if area_key not in face_metrics:
            continue
        u = ds[velocity_vars[axis]]
        a = face_metrics[area_key]
        flux = u if tracer is None else tracer * u
        flux = flux * a
        if region is not None:
            flux = flux.where(region)
        flux_dim = {"u": "x", "v": "y", "w": "z"}[axis]
        spatial_dims = [d for d in flux.dims if d != "time"]
        out[f"flux_{flux_dim}"] = flux.sum(dim=spatial_dims, skipna=True)

    return xr.Dataset(out)


__all__ = ["boundary_flux"]
