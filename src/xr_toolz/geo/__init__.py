"""Generic xarray geoprocessing — domain-agnostic operators.

This submodule hosts everything that applies across Earth-science
domains: coordinate validation, spatial/temporal subsetting, detrending
(climatology, anomalies), pixel-level metrics, and coordinate-range
encoders.

The public API is re-exported from :mod:`xr_toolz.geo._src`. For direct
access to a specific Layer-0 module, import e.g.
``xr_toolz.geo._src.detrend``. Layer-1 ``Operator`` wrappers land in a
follow-up release; see ``docs/design/`` for the full scope and roadmap.
"""

from xr_toolz.geo._src.detrend import (
    add_climatology,
    calculate_anomaly,
    calculate_anomaly_smoothed,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    remove_climatology,
)
from xr_toolz.geo._src.encoders import (
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
)
from xr_toolz.geo._src.metrics import (
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    rmse,
)
from xr_toolz.geo._src.subset import (
    select_variables,
    subset_bbox,
    subset_time,
    subset_where,
)
from xr_toolz.geo._src.validation import (
    rename_coords,
    validate_latitude,
    validate_longitude,
)


__all__ = [
    "add_climatology",
    "bias",
    "calculate_anomaly",
    "calculate_anomaly_smoothed",
    "calculate_climatology",
    "calculate_climatology_season",
    "calculate_climatology_smoothed",
    "correlation",
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
    "mae",
    "mse",
    "nrmse",
    "remove_climatology",
    "rename_coords",
    "rmse",
    "select_variables",
    "subset_bbox",
    "subset_time",
    "subset_where",
    "validate_latitude",
    "validate_longitude",
]
