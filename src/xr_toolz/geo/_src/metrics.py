"""Pixel-level evaluation metrics.

All metrics take ``ds_pred`` (prediction) and ``ds_ref`` (reference)
Datasets, a variable name, and a list of reduction dimensions. They
return an :class:`xr.DataArray` with the remaining dimensions.

Convention: positive ``bias`` means the prediction is larger than the
reference. Correlation is Pearson's ``r``.
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr


Dims = str | Sequence[str]


def mse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean squared error reduced over ``dims``."""
    diff = ds_pred[variable] - ds_ref[variable]
    return (diff**2).mean(dim=dims)


def rmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Root mean squared error reduced over ``dims``."""
    return mse(ds_pred, ds_ref, variable, dims) ** 0.5


def nrmse(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)``.

    Returns a score in ``(-inf, 1]`` where 1 means a perfect match and 0
    means the prediction is as wrong as a zero prediction.
    """
    err = rmse(ds_pred, ds_ref, variable, dims)
    scale = (ds_ref[variable] ** 2).mean(dim=dims) ** 0.5
    return 1.0 - err / scale


def mae(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean absolute error reduced over ``dims``."""
    return abs(ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def bias(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dims``."""
    return (ds_pred[variable] - ds_ref[variable]).mean(dim=dims)


def correlation(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    dims: Dims,
) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dims``."""
    return xr.corr(ds_pred[variable], ds_ref[variable], dim=dims)
