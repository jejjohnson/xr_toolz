"""xarray accessors for the sklearn bridge."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import xarray as xr

from xr_toolz.utils._src.sklearn_wrap import NanPolicy, XarrayEstimator


class _SklearnAccessor:
    """Thin xarray accessor that delegates to :class:`XarrayEstimator`."""

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = xarray_obj

    def _wrap(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> XarrayEstimator:
        return XarrayEstimator(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        )

    def _wrap_fitted(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> XarrayEstimator:
        wrap = self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        )
        wrap.estimator_ = estimator
        return wrap

    def fit(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
        **kwargs: Any,
    ) -> XarrayEstimator:
        """Fit ``estimator`` on this xarray object via ``XarrayEstimator``."""
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).fit(self._obj, y=y, **kwargs)

    def fit_transform(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
        **kwargs: Any,
    ) -> xr.DataArray | Any:
        """Fit and transform this xarray object via ``XarrayEstimator``."""
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).fit_transform(self._obj, y=y, **kwargs)

    def transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> xr.DataArray | Any:
        """Transform this xarray object with a fitted sklearn estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).transform(self._obj)

    def inverse_transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> xr.DataArray | Any:
        """Inverse-transform this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).inverse_transform(self._obj)

    def predict(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> xr.DataArray | Any:
        """Predict from this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).predict(self._obj)

    def predict_proba(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> xr.DataArray | Any:
        """Predict class probabilities with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).predict_proba(self._obj)

    def score(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> float:
        """Score this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).score(self._obj, y=y)


@xr.register_dataarray_accessor("sklearn")
class SklearnDataArrayAccessor(_SklearnAccessor):
    """``DataArray.sklearn`` adapter for :class:`XarrayEstimator`."""


@xr.register_dataset_accessor("sklearn")
class SklearnDatasetAccessor(_SklearnAccessor):
    """``Dataset.sklearn`` adapter for :class:`XarrayEstimator`."""
