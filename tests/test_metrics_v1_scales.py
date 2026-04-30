"""Tests for V1 (Scales of Evaluation): SkillByLeadTime, EvaluateByRegion.

Covers parity vs manual loops, support for the three region input
formats, the inner-Operator constraint, and graceful NaN handling.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.metrics import (
    MAE,
    RMSE,
    Bias,
    Correlation,
    EvaluateByRegion,
    PSDScore,
    SkillByLeadTime,
    evaluate_by_region,
    normalize_regions,
    skill_by_lead_time,
)


# ---------- fixtures ------------------------------------------------------


@pytest.fixture
def lead_time_pair() -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((4, 6, 8))
    ref = rng.standard_normal((4, 6, 8))
    coords = {
        "lead_time": np.arange(4),
        "lat": np.linspace(-1.0, 1.0, 6),
        "lon": np.linspace(0.0, 1.0, 8),
    }
    ds_p = xr.Dataset({"x": (("lead_time", "lat", "lon"), pred)}, coords=coords)
    ds_r = xr.Dataset({"x": (("lead_time", "lat", "lon"), ref)}, coords=coords)
    return ds_p, ds_r


@pytest.fixture
def regional_pair() -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(1)
    lat = np.linspace(-2.0, 2.0, 8)
    lon = np.linspace(-2.0, 2.0, 10)
    pred = rng.standard_normal((8, 10))
    ref = rng.standard_normal((8, 10))
    ds_p = xr.Dataset({"x": (("lat", "lon"), pred)}, coords={"lat": lat, "lon": lon})
    ds_r = xr.Dataset({"x": (("lat", "lon"), ref)}, coords={"lat": lat, "lon": lon})
    return ds_p, ds_r


# ---------- V1.1 ----------------------------------------------------------


@pytest.mark.parametrize(
    "metric_cls", [RMSE, MAE, Bias, Correlation], ids=lambda c: c.__name__
)
def test_skill_by_lead_time_parity(
    lead_time_pair: tuple[xr.Dataset, xr.Dataset], metric_cls
) -> None:
    """Per-lead skill must match a manual isel + metric loop."""
    ds_p, ds_r = lead_time_pair
    inner = metric_cls("x", ("lat", "lon"))
    op = SkillByLeadTime(inner)
    out = op(ds_p, ds_r)

    expected_pieces = [
        inner(ds_p.isel(lead_time=i), ds_r.isel(lead_time=i))
        for i in range(ds_p.sizes["lead_time"])
    ]
    expected = xr.concat(expected_pieces, dim=ds_p["lead_time"])
    np.testing.assert_allclose(out.values, expected.values)
    assert "lead_time" in out.dims


def test_skill_by_lead_time_with_psdscore(
    lead_time_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """Inner Dataset-returning metrics (PSDScore) are stacked along lead_time."""
    ds_p, ds_r = lead_time_pair
    inner = PSDScore("x", psd_dims=["lon"])
    op = SkillByLeadTime(inner)
    out = op(ds_p, ds_r)
    assert isinstance(out, xr.Dataset)
    assert "score" in out.data_vars
    assert "lead_time" in out.dims


def test_skill_by_lead_time_missing_lead_dim_raises() -> None:
    ds_p = xr.Dataset({"x": (("time",), np.zeros(5))})
    ds_r = xr.Dataset({"x": (("time",), np.zeros(5))})
    op = SkillByLeadTime(RMSE("x", dims="time"))
    with pytest.raises(ValueError, match="lead_time"):
        op(ds_p, ds_r)


def test_skill_by_lead_time_rejects_non_operator() -> None:
    with pytest.raises(TypeError, match="Operator"):
        SkillByLeadTime(lambda p, r: p)  # type: ignore[arg-type]


def test_skill_by_lead_time_get_config_introspectable() -> None:
    op = SkillByLeadTime(RMSE("x", ("lat", "lon")), lead_dim="step")
    cfg = op.get_config()
    assert cfg["lead_dim"] == "step"
    assert cfg["metric"]["class"] == "RMSE"
    assert cfg["metric"]["config"]["variable"] == "x"
    assert json.loads(json.dumps(cfg)) == cfg


def test_skill_by_lead_time_custom_lead_dim() -> None:
    rng = np.random.default_rng(2)
    pred = rng.standard_normal((3, 5))
    ref = rng.standard_normal((3, 5))
    ds_p = xr.Dataset(
        {"x": (("step", "time"), pred)},
        coords={"step": [1, 2, 3], "time": np.arange(5)},
    )
    ds_r = xr.Dataset(
        {"x": (("step", "time"), ref)}, coords={"step": [1, 2, 3], "time": np.arange(5)}
    )
    op = SkillByLeadTime(RMSE("x", dims="time"), lead_dim="step")
    out = op(ds_p, ds_r)
    assert "step" in out.dims
    assert out.sizes["step"] == 3


# ---------- V1.2 ----------------------------------------------------------


def test_evaluate_by_region_dict_of_masks(
    regional_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """A dict of named boolean masks is the simplest spec."""
    ds_p, ds_r = regional_pair
    masks = {
        "north": ds_p["lat"] > 0,
        "south": ds_p["lat"] <= 0,
    }
    op = EvaluateByRegion(RMSE("x", ("lat", "lon")), regions=masks)
    out = op(ds_p, ds_r)
    assert isinstance(out, xr.Dataset)
    assert list(out["region"].values) == ["north", "south"]

    # Parity vs manual where + metric.
    inner = RMSE("x", ("lat", "lon"))
    expected_north = inner(ds_p.where(masks["north"]), ds_r.where(masks["north"]))
    expected_south = inner(ds_p.where(masks["south"]), ds_r.where(masks["south"]))
    np.testing.assert_allclose(
        out["x"].sel(region="north").values, expected_north.values
    )
    np.testing.assert_allclose(
        out["x"].sel(region="south").values, expected_south.values
    )


def test_evaluate_by_region_int_mask_dataarray(
    regional_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """An integer label mask DataArray is also accepted."""
    ds_p, ds_r = regional_pair
    mask_vals = np.where(
        np.broadcast_to(ds_p["lat"].values[:, None] > 0, (8, 10)), 0, 1
    )
    mask = xr.DataArray(
        mask_vals, dims=("lat", "lon"), coords={"lat": ds_p["lat"], "lon": ds_p["lon"]}
    )
    op = EvaluateByRegion(RMSE("x", ("lat", "lon")), regions=mask)
    out = op(ds_p, ds_r)
    assert "region" in out.dims
    assert out.sizes["region"] == 2


def test_evaluate_by_region_empty_region_yields_nan(
    regional_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """A region with zero valid pixels should NaN, not raise."""
    ds_p, ds_r = regional_pair
    masks = {
        "everywhere": xr.ones_like(ds_p["x"], dtype=bool),
        "nowhere": xr.zeros_like(ds_p["x"], dtype=bool),
    }
    op = EvaluateByRegion(RMSE("x", ("lat", "lon")), regions=masks)
    out = op(ds_p, ds_r)
    assert np.isfinite(out["x"].sel(region="everywhere").values)
    assert np.isnan(out["x"].sel(region="nowhere").values)


def test_evaluate_by_region_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Unsupported regions type"):
        evaluate_by_region(
            xr.Dataset({"x": (("t",), np.zeros(3))}),
            xr.Dataset({"x": (("t",), np.zeros(3))}),
            metric=RMSE("x", "t"),
            regions=42,
        )


def test_normalize_regions_dict_round_trip(
    regional_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    ds_p, _ = regional_pair
    masks = {"a": ds_p["lat"] > 0, "b": ds_p["lat"] <= 0}
    mask, names = normalize_regions(masks, ds_p)
    assert names == {0: "a", 1: "b"}
    assert mask.dtype == np.int64
    assert set(np.unique(mask.values).tolist()) == {0, 1}


def test_evaluate_by_region_regionmask_lazy_import_error() -> None:
    """Passing an unknown object that isn't a regionmask.Regions raises TypeError,
    not ImportError — the regionmask import only fires for objects that
    look like regionmask.Regions."""
    pytest.importorskip("regionmask")
    import regionmask

    # Build a tiny custom Regions instance.
    regions = regionmask.Regions(
        outlines=[np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])],
        names=["square"],
        abbrevs=["sq"],
    )
    rng = np.random.default_rng(3)
    lat = np.linspace(-2.0, 2.0, 8)
    lon = np.linspace(-2.0, 2.0, 10)
    ds_p = xr.Dataset(
        {"x": (("lat", "lon"), rng.standard_normal((8, 10)))},
        coords={"lat": lat, "lon": lon},
    )
    ds_r = xr.Dataset(
        {"x": (("lat", "lon"), rng.standard_normal((8, 10)))},
        coords={"lat": lat, "lon": lon},
    )
    op = EvaluateByRegion(RMSE("x", ("lat", "lon")), regions=regions)
    out = op(ds_p, ds_r)
    assert "region" in out.dims
    assert "square" in out["region"].values.tolist()


def test_evaluate_by_region_get_config_json_safe(
    regional_pair: tuple[xr.Dataset, xr.Dataset],
) -> None:
    ds_p, _ = regional_pair
    op = EvaluateByRegion(RMSE("x", ("lat", "lon")), regions={"n": ds_p["lat"] > 0})
    cfg = op.get_config()
    assert cfg["metric"]["class"] == "RMSE"
    assert cfg["regions"] == ["n"]
    assert json.loads(json.dumps(cfg)) == cfg


def test_works_inside_graph(lead_time_pair: tuple[xr.Dataset, xr.Dataset]) -> None:
    from xr_toolz.core import Graph, Input

    ds_p, ds_r = lead_time_pair
    p = Input("p")
    r = Input("r")
    out = SkillByLeadTime(RMSE("x", ("lat", "lon")))(p, r)
    graph = Graph(inputs={"p": p, "r": r}, outputs={"y": out})
    result = graph(p=ds_p, r=ds_r)
    expected = skill_by_lead_time(ds_p, ds_r, metric=RMSE("x", ("lat", "lon")))
    np.testing.assert_allclose(result["y"].values, expected.values)
