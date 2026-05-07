"""xarray accessors for the sklearn bridge.

Registers the ``da.sklearn`` and ``ds.sklearn`` accessors at import time
so users can run sklearn-style verbs directly on xarray objects without
constructing an :class:`XarrayEstimator` by hand. Each accessor method
constructs an ``XarrayEstimator`` under the hood and delegates — there
is no parallel marshalling path. Importing :mod:`xr_toolz.utils` is
enough to register the accessors.

Example:
    >>> import xarray as xr
    >>> from sklearn.preprocessing import StandardScaler
    >>> import xr_toolz.utils  # registers the .sklearn accessor  # noqa: F401
    >>> scaled = ssh.sklearn.fit_transform(StandardScaler(), sample_dim="time")
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import xarray as xr

from xr_toolz.utils._src.sklearn_wrap import NanPolicy, XarrayEstimator


class _SklearnAccessor:
    """Thin xarray accessor that delegates to :class:`XarrayEstimator`.

    Provides ``fit``, ``fit_transform``, ``transform``, ``inverse_transform``,
    ``predict``, ``predict_proba``, and ``score`` directly on
    :class:`xarray.DataArray` / :class:`xarray.Dataset` objects via the
    registered ``.sklearn`` namespace. The accessor never re-implements
    the stack→delegate→unstack marshalling — every method constructs an
    :class:`XarrayEstimator` and forwards.

    Methods that need a *fitted* estimator (``transform``,
    ``inverse_transform``, ``predict``, ``predict_proba``, ``score``)
    use a shortcut that sets ``estimator_`` on a fresh wrapper. That
    shortcut bypasses the wrapper's ``_fitted_meta_`` capture, so
    ``inverse_transform`` falls into the generic
    ``(sample_dim, component)`` layout rather than restoring the
    training feature grid. To get the original grid back, fit an
    :class:`XarrayEstimator` explicitly:

    >>> from xr_toolz.utils import XarrayEstimator
    >>> from sklearn.decomposition import PCA
    >>> wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
    >>> recon = wrap.inverse_transform(wrap.transform(da))   # → (time, lat, lon)

    Example:
        >>> # Fit-and-transform directly off a DataArray
        >>> scores = da.sklearn.fit_transform(
        ...     PCA(n_components=3),
        ...     sample_dim="time",
        ...     nan_policy="mask",
        ... )

        >>> # Score a fitted estimator against a held-out slice
        >>> r2 = da_test.sklearn.score(fitted_regressor, y_test, sample_dim="time")

        >>> # Dataset variant: column-concat data_vars before fitting
        >>> scaled = ds.sklearn.fit_transform(StandardScaler(), sample_dim="time")
    """

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
