"""Coordinate validation and harmonization.

These helpers normalize coordinate names (``longitude`` → ``lon``,
``latitude`` → ``lat``), wrap values into the standard geographic
ranges, and attach CF-style ``units``, ``standard_name``, and
``long_name`` attributes.
"""

from __future__ import annotations

import xarray as xr

from xr_toolz.transforms._src.encoders.coord_space import (
    lat_180_to_90,
    lon_360_to_180,
)


_LONGITUDE_ALIASES = ("longitude",)
_LATITUDE_ALIASES = ("latitude",)

_LON_ATTRS = {
    "units": "degrees_east",
    "standard_name": "longitude",
    "long_name": "Longitude",
}
_LAT_ATTRS = {
    "units": "degrees_north",
    "standard_name": "latitude",
    "long_name": "Latitude",
}


def validate_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the longitude coordinate.

    Renames ``longitude`` to ``lon`` if present, wraps values into the
    ``[-180, 180)`` range, and assigns CF ``units``, ``standard_name``,
    and ``long_name`` attributes (preserving any pre-existing attrs).

    Args:
        ds: Input dataset.

    Returns:
        Dataset with a harmonized ``lon`` coordinate.
    """
    new_ds = _rename_first_match(ds, _LONGITUDE_ALIASES, "lon")
    if "lon" not in new_ds.coords and "lon" not in new_ds.variables:
        raise KeyError("No longitude coordinate found (expected 'lon' or 'longitude').")

    existing_attrs = dict(new_ds["lon"].attrs)
    new_ds["lon"] = lon_360_to_180(new_ds["lon"])
    new_ds["lon"] = new_ds["lon"].assign_attrs(**{**existing_attrs, **_LON_ATTRS})
    return new_ds


def validate_latitude(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the latitude coordinate.

    Renames ``latitude`` to ``lat`` if present, wraps values into the
    ``[-90, 90)`` range, and assigns CF ``units``, ``standard_name``,
    and ``long_name`` attributes (preserving any pre-existing attrs).

    Args:
        ds: Input dataset.

    Returns:
        Dataset with a harmonized ``lat`` coordinate.
    """
    new_ds = _rename_first_match(ds, _LATITUDE_ALIASES, "lat")
    if "lat" not in new_ds.coords and "lat" not in new_ds.variables:
        raise KeyError("No latitude coordinate found (expected 'lat' or 'latitude').")

    existing_attrs = dict(new_ds["lat"].attrs)
    new_ds["lat"] = lat_180_to_90(new_ds["lat"])
    new_ds["lat"] = new_ds["lat"].assign_attrs(**{**existing_attrs, **_LAT_ATTRS})
    return new_ds


def rename_coords(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Rename any coordinates or variables that match ``mapping``.

    Keys not present in ``ds`` are silently ignored, so this helper is
    safe to use as a first-pass harmonizer without pre-checking names.

    Args:
        ds: Input dataset.
        mapping: ``{old_name: new_name}``. Only names actually present
            are renamed.

    Returns:
        Dataset with matching names renamed.
    """
    present = {old: new for old, new in mapping.items() if old in ds.variables}
    if not present:
        return ds
    return ds.rename(present)


def _rename_first_match(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    target: str,
) -> xr.Dataset:
    """Rename the first matching alias to ``target`` if ``target`` is missing."""
    if target in ds.variables:
        return ds
    for name in candidates:
        if name in ds.variables:
            return ds.rename({name: target})
    return ds


# ---------- variable-attribute harmonization ------------------------------


def validate_ssh(ds: xr.Dataset, variable: str = "ssh") -> xr.Dataset:
    """Attach CF-style attrs to an SSH variable."""
    ds = ds.copy()
    ds[variable] = ds[variable].assign_attrs(
        units="m",
        standard_name="sea_surface_height",
        long_name="Sea Surface Height",
    )
    return ds


def validate_velocity(
    ds: xr.Dataset,
    u: str = "u",
    v: str = "v",
) -> xr.Dataset:
    """Attach CF-style attrs to zonal / meridional velocity variables."""
    ds = ds.copy()
    ds[u] = ds[u].assign_attrs(
        units="m s-1",
        standard_name="sea_water_x_velocity",
        long_name="Zonal Velocity",
    )
    ds[v] = ds[v].assign_attrs(
        units="m s-1",
        standard_name="sea_water_y_velocity",
        long_name="Meridional Velocity",
    )
    return ds
