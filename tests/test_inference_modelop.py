"""Behavioral tests for :class:`xr_toolz.inference.ModelOp`.

Covers the framework-agnostic core: xarray <-> array marshalling,
``method=`` dispatch, batched non-feature dims, raw-array inputs, and
JSON-safe :meth:`get_config`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.inference import ModelOp


class _DummyModel:
    """Minimal duck-typed model: ``predict`` doubles, ``transform`` halves."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x.sum(axis=-1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x


@pytest.fixture
def da_2d() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((6, 3)),
        dims=("sample", "feature"),
        coords={"sample": np.arange(6), "feature": ["a", "b", "c"]},
    )


@pytest.fixture
def da_batched() -> xr.DataArray:
    rng = np.random.default_rng(1)
    return xr.DataArray(
        rng.standard_normal((4, 5, 3)),
        dims=("time", "sample", "feature"),
        coords={"time": np.arange(4), "feature": ["a", "b", "c"]},
    )


def test_predict_default_method(da_2d: xr.DataArray) -> None:
    op = ModelOp(_DummyModel())
    out = op(da_2d)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("sample",)
    np.testing.assert_allclose(out.values, 2.0 * da_2d.values.sum(axis=-1))
    assert out.name == "prediction"


def test_method_dispatch_transform(da_2d: xr.DataArray) -> None:
    op = ModelOp(_DummyModel(), method="transform", output_name="halved")
    out = op(da_2d)
    assert out.dims == ("sample", "output")
    np.testing.assert_allclose(out.values, 0.5 * da_2d.values)
    assert out.name == "halved"


def test_batched_leading_dim(da_batched: xr.DataArray) -> None:
    op = ModelOp(_DummyModel())
    out = op(da_batched)
    assert out.dims == ("time", "sample")
    expected = 2.0 * da_batched.values.sum(axis=-1)
    np.testing.assert_allclose(out.values, expected)


def test_dataset_input_stacks_along_feature() -> None:
    rng = np.random.default_rng(2)
    ds = xr.Dataset(
        {
            "a": (("sample",), rng.standard_normal(4)),
            "b": (("sample",), rng.standard_normal(4)),
        },
        coords={"sample": np.arange(4)},
    )
    op = ModelOp(_DummyModel())
    out = op(ds)
    assert out.dims == ("sample",)
    expected = 2.0 * (ds["a"].values + ds["b"].values)
    np.testing.assert_allclose(out.values, expected)


def test_raw_numpy_input_2d() -> None:
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((5, 3))
    op = ModelOp(_DummyModel())
    out = op(arr)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_allclose(out.values, 2.0 * arr.sum(axis=-1))


def test_missing_feature_dim_raises() -> None:
    da = xr.DataArray(np.zeros((4, 3)), dims=("sample", "channel"))
    op = ModelOp(_DummyModel())
    with pytest.raises(ValueError, match="feature dim"):
        op(da)


def test_unknown_method_raises() -> None:
    op = ModelOp(_DummyModel(), method="nope")
    with pytest.raises(AttributeError, match="nope"):
        op(np.zeros((2, 2)))


def test_get_config_is_json_serializable() -> None:
    op = ModelOp(_DummyModel(), method="predict", feature_dim="band")
    cfg = op.get_config()
    assert cfg["model"] == "<model>"
    assert cfg["method"] == "predict"
    assert cfg["feature_dim"] == "band"
    assert json.loads(json.dumps(cfg)) == cfg


def test_repr_uses_config() -> None:
    op = ModelOp(_DummyModel(), feature_dim="band")
    r = repr(op)
    assert "ModelOp" in r and "feature_dim='band'" in r


def test_works_inside_graph(da_2d: xr.DataArray) -> None:
    from xr_toolz.core import Graph, Input

    inp = Input("x")
    out = ModelOp(_DummyModel())(inp)
    graph = Graph(inputs={"x": inp}, outputs={"y": out})
    result = graph(x=da_2d)
    np.testing.assert_allclose(result["y"].values, 2.0 * da_2d.values.sum(axis=-1))
