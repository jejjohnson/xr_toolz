"""V6 validation-panel tests — V6.1 base + V6.2/3/4 panels."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")  # headless — must run before pyplot import

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Graph, Input, Sequential
from xr_toolz.viz.validation import (
    EulerianLagrangianPanel,
    EventVerificationPanel,
    LeadTimeSkillPanel,
    ProcessBudgetPanel,
    PSDIsotropicPanel,
    PSDIsotropicScorePanel,
    PSDSpaceTimePanel,
    PSDSpaceTimeScorePanel,
    ScaleSkillPanel,
    SpectralSkillPanel,
)
from xr_toolz.viz.validation._src.base import _ValidationPanel


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- V6.1 base ----------------------------------------------------------


def test_validation_panel_returns_figure():
    class _Demo(_ValidationPanel):
        def _build(self, fig, axes, x):
            axes.plot(np.arange(len(x)), x)

    fig = _Demo()(np.array([1.0, 2.0, 3.0]))
    assert isinstance(fig, mpl_figure.Figure)


def test_validation_panel_get_config_round_trips_kwargs():
    class _Demo(_ValidationPanel):
        def _build(self, fig, axes, x):
            return None

    cfg = _Demo(figsize=(6, 4), title="hello", style=None).get_config()
    assert cfg["figsize"] == [6, 4]
    assert cfg["title"] == "hello"


def test_validation_panel_does_not_mutate_input_dataset():
    da = xr.DataArray(np.arange(5.0), dims=("lead_time",), name="rmse")
    ds = da.to_dataset()
    ds_before = ds.copy(deep=True)
    LeadTimeSkillPanel()(ds)
    xr.testing.assert_identical(ds, ds_before)


def test_validation_panel_in_sequential_at_last_step_returns_figure():
    da = xr.DataArray(np.arange(5.0), dims=("lead_time",))
    pipeline = Sequential([LeadTimeSkillPanel()])
    out = pipeline(da)
    assert isinstance(out, mpl_figure.Figure)


def test_validation_panel_non_terminal_in_sequential_fails_downstream():
    """A panel placed mid-pipeline must fail downstream — a Figure has no
    Dataset / DataArray methods, so the next op's natural error surfaces.

    Sequential itself is intentionally generic (it pipes any callable),
    so the contract is documented rather than runtime-enforced; this
    test pins the failure mode so a future Sequential rewrite that
    silently accepts a Figure does not regress D10.
    """

    class _NeedsDataArray:
        def __call__(self, x):
            return x.where(x > 0)  # Figure has no .where()

    da = xr.DataArray(np.arange(5.0), dims=("lead_time",))
    pipeline = Sequential([LeadTimeSkillPanel(), _NeedsDataArray()])
    with pytest.raises(AttributeError):
        pipeline(da)


def test_validation_panel_in_graph_emits_figure_alongside_score():
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    inp = Input(name="skill")
    fig_node = LeadTimeSkillPanel()(inp)
    g = Graph(inputs={"skill": inp}, outputs={"figure": fig_node, "raw": inp})
    out = g(skill=da)
    assert isinstance(out["figure"], mpl_figure.Figure)
    xr.testing.assert_identical(out["raw"], da)


# ---- V6.2 V1 panels -----------------------------------------------------


def test_lead_time_skill_panel_renders_dataarray():
    da = xr.DataArray(
        np.array([0.1, 0.2, 0.3, 0.4]),
        coords={"lead_time": np.arange(4)},
        dims=("lead_time",),
        name="rmse",
    )
    fig = LeadTimeSkillPanel()(da)
    assert isinstance(fig, mpl_figure.Figure)


def test_lead_time_skill_panel_renders_dataset_with_legend():
    da_a = xr.DataArray(
        np.arange(4.0),
        coords={"lead_time": np.arange(4)},
        dims=("lead_time",),
        name="rmse",
    )
    da_b = da_a.copy(data=np.arange(4.0) * 1.5).rename("mae")
    ds = xr.Dataset({"rmse": da_a, "mae": da_b})
    fig = LeadTimeSkillPanel()(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_scale_skill_panel_default_metric_picks_first():
    ds = xr.Dataset(
        {
            "rmse": ("region", [0.5, 0.7, 0.6]),
            "mae": ("region", [0.4, 0.5, 0.45]),
        },
        coords={"region": ["GS", "MD", "AC"]},
    )
    fig = ScaleSkillPanel()(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_scale_skill_panel_explicit_metric():
    ds = xr.Dataset(
        {"rmse": ("region", [0.5, 0.7]), "mae": ("region", [0.4, 0.5])},
        coords={"region": ["A", "B"]},
    )
    fig = ScaleSkillPanel(metric="mae")(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_spectral_skill_panel_uses_log_x_by_default():
    da = xr.DataArray(
        np.linspace(0.5, 1.0, 20),
        coords={"freq": np.logspace(-3, 0, 20)},
        dims=("freq",),
        name="psd_score",
    )
    fig = SpectralSkillPanel()(da)
    assert isinstance(fig, mpl_figure.Figure)


def test_spectral_skill_panel_log_y_toggle():
    da = xr.DataArray(
        np.linspace(0.5, 1.0, 5),
        coords={"freq": np.logspace(-2, 0, 5)},
        dims=("freq",),
    )
    fig = SpectralSkillPanel(log_y=True)(da)
    assert isinstance(fig, mpl_figure.Figure)


# ---- V6.3 V3 + V4 panels ------------------------------------------------


def _trajectory_fixture(n_traj=3, n_t=10):
    rng = np.random.default_rng(0)
    times = np.arange(n_t)
    lons = np.cumsum(rng.normal(0.0, 0.3, size=(n_traj, n_t)), axis=1) - 10.0
    lats = np.cumsum(rng.normal(0.0, 0.2, size=(n_traj, n_t)), axis=1) + 35.0
    return xr.Dataset(
        {
            "lon": (("trajectory", "time"), lons),
            "lat": (("trajectory", "time"), lats),
        },
        coords={"trajectory": np.arange(n_traj), "time": times},
    )


def _eulerian_fixture():
    lon = np.linspace(-15, -5, 21)
    lat = np.linspace(30, 40, 21)
    LON, LAT = np.meshgrid(lon, lat, indexing="xy")
    field = 0.1 * np.sin(np.deg2rad(LON)) * np.cos(np.deg2rad(LAT))
    return xr.Dataset(
        {"ssh": (("lat", "lon"), field)},
        coords={"lat": lat, "lon": lon},
    )


def test_eulerian_lagrangian_panel_renders():
    panel = EulerianLagrangianPanel(eulerian_var="ssh")
    fig = panel(_eulerian_fixture(), _trajectory_fixture())
    assert isinstance(fig, mpl_figure.Figure)


def test_eulerian_lagrangian_panel_handles_eulerian_with_time():
    eul = _eulerian_fixture()
    eul = eul.expand_dims(time=np.arange(2))
    panel = EulerianLagrangianPanel()
    fig = panel(eul, _trajectory_fixture())
    assert isinstance(fig, mpl_figure.Figure)


def test_process_budget_panel_residual_only():
    rng = np.random.default_rng(1)
    time = np.arange(20)
    res = xr.DataArray(
        rng.normal(0.0, 0.01, size=(20, 5, 5)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.arange(5), "lon": np.arange(5)},
    )
    fig = ProcessBudgetPanel()(res)
    assert isinstance(fig, mpl_figure.Figure)


def test_process_budget_panel_with_components():
    rng = np.random.default_rng(2)
    time = np.arange(15)
    base = rng.normal(0.0, 0.05, size=(15, 4, 4))
    coords = {"time": time, "lat": np.arange(4), "lon": np.arange(4)}
    dims = ("time", "lat", "lon")
    components = {
        "tendency": xr.DataArray(base, dims=dims, coords=coords),
        "advection": xr.DataArray(-base * 0.5, dims=dims, coords=coords),
        "source": xr.DataArray(np.zeros_like(base), dims=dims, coords=coords),
    }
    residual = xr.DataArray(base * 0.5, dims=dims, coords=coords)
    fig = ProcessBudgetPanel()(residual, components)
    assert isinstance(fig, mpl_figure.Figure)


# ---- V6.4 V5 panel ------------------------------------------------------


def _objects_fixture(label_grid):
    lon = np.linspace(-10.0, 0.0, label_grid.shape[1])
    lat = np.linspace(30.0, 40.0, label_grid.shape[0])
    return xr.Dataset(
        {"label": (("lat", "lon"), label_grid)},
        coords={"lat": lat, "lon": lon},
    )


def test_event_verification_panel_renders_basic():
    pred = np.zeros((10, 12), dtype=int)
    pred[2:5, 2:5] = 1
    pred[6:8, 7:10] = 2
    ref = np.zeros((10, 12), dtype=int)
    ref[2:5, 2:5] = 10
    ref[7:9, 1:3] = 20
    matches = {
        "hits": [(1, 10)],
        "false_alarms": [2],
        "misses": [20],
    }
    scores = {"POD": 0.5, "FAR": 0.5, "CSI": 0.33, "IoU": 0.4}
    fig = EventVerificationPanel(use_cartopy=False)(
        _objects_fixture(pred), _objects_fixture(ref), matches, scores
    )
    assert isinstance(fig, mpl_figure.Figure)


def test_event_verification_panel_handles_no_hits():
    pred = np.zeros((6, 6), dtype=int)
    pred[1:3, 1:3] = 1
    ref = np.zeros((6, 6), dtype=int)
    ref[3:5, 3:5] = 7
    matches = {"hits": [], "false_alarms": [1], "misses": [7]}
    scores = {"POD": 0.0, "FAR": 1.0, "CSI": 0.0, "IoU": 0.0}
    fig = EventVerificationPanel(use_cartopy=False)(
        _objects_fixture(pred), _objects_fixture(ref), matches, scores
    )
    assert isinstance(fig, mpl_figure.Figure)


# ---- viz package surface ------------------------------------------------


def test_viz_validation_top_level_imports():
    import xr_toolz.viz.validation as vv

    for name in (
        "LeadTimeSkillPanel",
        "ScaleSkillPanel",
        "SpectralSkillPanel",
        "EulerianLagrangianPanel",
        "ProcessBudgetPanel",
        "EventVerificationPanel",
        "PSDIsotropicPanel",
        "PSDIsotropicScorePanel",
        "PSDSpaceTimePanel",
        "PSDSpaceTimeScorePanel",
    ):
        assert hasattr(vv, name)


# ---- V1.5 PSD panels ----------------------------------------------------


@pytest.fixture
def _psd_fixtures():
    """Build small realistic PSD/score fixtures from the GLORYS cache.

    Skips if the cache is missing — these tests are meant to exercise
    the panels against real spectra, not synthetic input.
    """
    from pathlib import Path

    import scipy.ndimage as ndi

    from xr_toolz.metrics import psd_score
    from xr_toolz.transforms import power_spectrum

    cache = Path(__file__).resolve().parents[1] / ".cache"
    g_path = cache / "glorys12_north_atlantic_2023-06.nc"
    if not g_path.exists():
        pytest.skip("North Atlantic GLORYS cache missing — see notebook for rebuild.")

    g = xr.open_dataset(g_path)["zos"].rename({"latitude": "lat", "longitude": "lon"})
    # Subsample for test speed (every 4th cell in space, every 3rd day).
    g = g.isel(
        time=slice(None, None, 3), lat=slice(None, None, 4), lon=slice(None, None, 4)
    )

    t = g["time"].values
    days = (t - t[0]) / np.timedelta64(1, "D")
    g = g.assign_coords(time=days.astype("float64"))
    g_a = (g - g.mean(("lat", "lon"))).fillna(0.0)
    s = xr.apply_ufunc(
        lambda a: ndi.gaussian_filter(a, sigma=(0, 2.0, 2.0)), g_a
    ).rename("zos")

    iso = power_spectrum(g_a, dim=("lon", "lat"), isotropic=True).mean("time")
    iso_score = psd_score(
        s.to_dataset(name="zos"),
        g_a.to_dataset(name="zos"),
        "zos",
        psd_dims=("lon", "lat"),
        isotropic=True,
    ).mean("time")
    st = power_spectrum(g_a, dim=("lon", "time")).mean("lat")
    st_score = psd_score(
        s.to_dataset(name="zos"),
        g_a.to_dataset(name="zos"),
        "zos",
        psd_dims=("lon", "time"),
        avg_dims=("lat",),
        isotropic=False,
    )
    return {"iso": iso, "iso_score": iso_score, "st": st, "st_score": st_score}


def test_psd_isotropic_panel_renders(_psd_fixtures):
    fig = PSDIsotropicPanel(freq_dim="freq_r", space_scale=1.0 / 111.0)(
        _psd_fixtures["iso"]
    )
    assert isinstance(fig, mpl_figure.Figure)


def test_psd_isotropic_score_panel_resolves_scale(_psd_fixtures):
    panel = PSDIsotropicScorePanel(freq_dim="freq_r", space_scale=1.0 / 111.0)
    fig = panel(_psd_fixtures["iso_score"])
    # The smoothed twin must produce a resolved-scale crossing — a
    # legend entry containing "Resolved scale" is the panel's
    # observable signal that find_intercept_1D succeeded.
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("Resolved scale" in lab for lab in labels)


def test_psd_isotropic_score_panel_clips_negative_scores(_psd_fixtures):
    score = _psd_fixtures["iso_score"].copy()
    score["score"] = score["score"] * 0 - 0.5  # all-negative
    fig = PSDIsotropicScorePanel(freq_dim="freq_r", space_scale=1.0 / 111.0)(score)
    ymin, ymax = fig.axes[0].get_ylim()
    assert ymin == 0.0 and ymax == 1.0


def test_psd_space_time_panel_renders(_psd_fixtures):
    fig = PSDSpaceTimePanel(
        freq_space_dim="freq_lon",
        freq_time_dim="freq_time",
        space_scale=1.0 / 111.0,
        time_scale=1.0,
    )(_psd_fixtures["st"])
    assert isinstance(fig, mpl_figure.Figure)


def test_psd_space_time_score_panel_renders(_psd_fixtures):
    fig = PSDSpaceTimeScorePanel(
        freq_space_dim="freq_lon",
        freq_time_dim="freq_time",
        space_scale=1.0 / 111.0,
        time_scale=1.0,
    )(_psd_fixtures["st_score"])
    assert isinstance(fig, mpl_figure.Figure)


def test_psd_panels_get_config_keys():
    cfg = PSDIsotropicPanel(space_scale=2.0).get_config()
    assert cfg["space_scale"] == 2.0
    cfg = PSDIsotropicScorePanel(threshold=0.7, space_scale=1e-3).get_config()
    assert cfg["threshold"] == 0.7
    cfg = PSDSpaceTimePanel(time_scale=86400.0).get_config()
    assert cfg["time_scale"] == 86400.0
    cfg = PSDSpaceTimeScorePanel(threshold=0.4).get_config()
    assert cfg["threshold"] == 0.4


# ---- savefig / show args (panel-wide) -----------------------------------


def test_panel_savefig_writes_file(tmp_path):
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    out = tmp_path / "skill.png"
    fig = LeadTimeSkillPanel(savefig=out)(da)
    assert isinstance(fig, mpl_figure.Figure)
    assert out.exists() and out.stat().st_size > 0


def test_panel_savefig_kwargs_forwarded(tmp_path):
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    out_lo = tmp_path / "lo.png"
    out_hi = tmp_path / "hi.png"
    LeadTimeSkillPanel(savefig=out_lo, savefig_kwargs={"dpi": 50})(da)
    LeadTimeSkillPanel(savefig=out_hi, savefig_kwargs={"dpi": 200})(da)
    # Higher dpi → larger PNG.
    assert out_hi.stat().st_size > out_lo.stat().st_size


def test_panel_show_calls_pyplot_show(monkeypatch):
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    calls = {"n": 0}
    monkeypatch.setattr(plt, "show", lambda *a, **kw: calls.update(n=calls["n"] + 1))
    LeadTimeSkillPanel(show=True)(da)
    assert calls["n"] == 1


def test_panel_show_default_does_not_call_pyplot_show(monkeypatch):
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    calls = {"n": 0}
    monkeypatch.setattr(plt, "show", lambda *a, **kw: calls.update(n=calls["n"] + 1))
    LeadTimeSkillPanel()(da)
    assert calls["n"] == 0


def test_psd_panel_savefig_writes_file(tmp_path, _psd_fixtures):
    out = tmp_path / "iso.png"
    PSDIsotropicPanel(freq_dim="freq_r", space_scale=1.0 / 111.0, savefig=out)(
        _psd_fixtures["iso"]
    )
    assert out.exists() and out.stat().st_size > 0
