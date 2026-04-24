"""AemetSource tests with a fake HTTP client (no network I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from xr_toolz.data import AemetAuthError, AEMETCredentials, AemetSource
from xr_toolz.data._src.aemet.source import AemetRateLimitError
from xr_toolz.types import TimeRange


# ---- fake HTTP ----------------------------------------------------------


@dataclass
class _Resp:
    status_code: int
    body: Any = None
    headers: dict[str, str] | None = None
    text: str = ""

    def json(self):
        return self.body

    @property
    def content(self) -> bytes:
        import json

        if self.body is None:
            return b""
        return json.dumps(self.body).encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeClient:
    """In-memory router: path → list of responses (each get() pops one)."""

    def __init__(self, routes: dict[str, list[_Resp]]):
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, headers=None, timeout=None):
        self.calls.append((url, headers))
        for path, responses in self.routes.items():
            if url.endswith(path) or url == path:
                if not responses:
                    raise AssertionError(f"no more responses queued for {path}")
                return responses.pop(0)
        raise AssertionError(f"unexpected URL: {url}")


def _env_ok(datos: str, *, remaining: int = 100) -> _Resp:
    return _Resp(
        status_code=200,
        body={"estado": 200, "descripcion": "exito", "datos": datos, "metadatos": ""},
        headers={"Remaining-request-count": str(remaining)},
    )


# ---- fixtures -----------------------------------------------------------


@pytest.fixture
def source_and_fake():
    fake = FakeClient(routes={})
    src = AemetSource(
        credentials=AEMETCredentials(api_key="test"),
        client=fake,
        max_retries=2,
        max_workers=1,
    )
    return src, fake


# ---- auth ---------------------------------------------------------------


def test_requires_api_key(monkeypatch, tmp_path):
    # Isolate from the developer's real .env / env so the autoload
    # doesn't resolve a key from the enclosing shell.
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    src = AemetSource(credentials=None, client=FakeClient({}))
    with pytest.raises(AemetAuthError):
        src._require_key()


def test_401_raises_auth_error(source_and_fake):
    src, fake = source_and_fake
    fake.routes = {
        "/valores/climatologicos/inventarioestaciones/todasestaciones": [
            _Resp(status_code=401, text="bad key")
        ]
    }
    with pytest.raises(AemetAuthError):
        src.list_stations()


# ---- rate limit ----------------------------------------------------------


def test_429_retries_then_raises(source_and_fake):
    src, fake = source_and_fake
    path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    fake.routes = {
        path: [_Resp(status_code=429), _Resp(status_code=429), _Resp(status_code=429)]
    }
    # Zero backoff for the test so it doesn't sleep.
    with pytest.raises(AemetRateLimitError):
        # Monkeypatch sleep via zero-retry config.
        src.max_retries = 2
        src.list_stations()


# ---- stations -----------------------------------------------------------


def test_list_stations_parses_dms(source_and_fake):
    src, fake = source_and_fake
    envelope_path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    data_url = "https://fake.aemet/data1"
    fake.routes = {
        envelope_path: [_env_ok(data_url)],
        data_url: [
            _Resp(
                status_code=200,
                body=[
                    {
                        "indicativo": "3195",
                        "nombre": "MADRID, RETIRO",
                        "provincia": "MADRID",
                        "latitud": "402358N",
                        "longitud": "034041W",
                        "altitud": "667",
                        "indsinop": "08222",
                    }
                ],
            )
        ],
    }
    stations = src.list_stations()
    assert len(stations) == 1
    s = stations["3195"]
    assert s.name == "MADRID, RETIRO"
    assert 40.0 < s.lat < 40.5
    assert -4.0 < s.lon < -3.5
    assert s.altitude == 667.0
    assert s.wmo_id == "08222"
    assert s.source == "aemet"


# ---- daily --------------------------------------------------------------


def _daily_row(fecha: str, **kwargs: str) -> dict[str, str]:
    base = {
        "fecha": fecha,
        "indicativo": "3195",
        "tmed": "10,5",
        "tmin": "5,0",
        "tmax": "15,0",
        "prec": "0,3",
    }
    base.update(kwargs)
    return base


def test_get_daily_stitches_chunked_windows(source_and_fake):
    # The parent fixture is unused here; this test builds its own
    # path-matching client to handle the variable daily URL.
    del source_and_fake
    # Two 180-day chunks for a single station.
    # The mock returns the same two-row payload for both envelope URLs.
    env = _env_ok("https://fake.aemet/daily1")
    env2 = _env_ok("https://fake.aemet/daily2")

    # We don't pin exact URL paths because the date strings vary; match prefix.
    class PathMatcher(FakeClient):
        def get(self, url: str, headers=None, timeout=None):
            self.calls.append((url, headers))
            if "/valores/climatologicos/diarios/datos/" in url:
                return self.routes["envelope"].pop(0)
            return self.routes[url].pop(0)

    fake2 = PathMatcher(
        routes={
            "envelope": [env, env2],
            "https://fake.aemet/daily1": [
                _Resp(
                    status_code=200,
                    body=[
                        _daily_row("2024-01-01"),
                        _daily_row("2024-01-02", tmed="11,0"),
                    ],
                )
            ],
            "https://fake.aemet/daily2": [
                _Resp(
                    status_code=200,
                    body=[_daily_row("2024-07-10", tmed="22,5")],
                )
            ],
        }
    )
    src2 = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake2,
        max_workers=1,
    )
    tr = TimeRange.parse("2024-01-01", "2024-07-15")
    ds = src2.get_daily(["3195"], time=tr)
    assert ds.sizes["station"] == 1
    # Full daily index over the window
    assert ds.sizes["time"] >= 190
    tmed = ds["air_temperature_daily_mean"].sel(station="3195")
    # First two days observed, middle is NaN, then the July value is observed.
    values = tmed.values
    assert not np.isnan(values[0])
    assert not np.isnan(values[1])
    assert np.isnan(values[100])  # a gap somewhere
    assert ds.attrs["source"] == "aemet"
    assert ds.attrs["featureType"] == "timeSeries"
    assert ds.attrs["endpoint"] == "daily"
    # CF attrs wired through
    attrs = ds["air_temperature_daily_mean"].attrs
    assert attrs.get("standard_name") == "air_temperature"


def test_get_daily_requires_time(source_and_fake):
    src, _ = source_and_fake
    with pytest.raises(ValueError, match="TimeRange"):
        src.get_daily(["3195"])


def test_get_daily_rejects_empty_stations(source_and_fake):
    src, _ = source_and_fake
    tr = TimeRange.parse("2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="at least one station"):
        src.get_daily([], time=tr)


# ---- dataset-level dispatch ---------------------------------------------


def test_open_stations_preset_returns_inventory(source_and_fake):
    src, fake = source_and_fake
    env_path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    data_url = "https://fake.aemet/inv"
    fake.routes = {
        env_path: [_env_ok(data_url)],
        data_url: [
            _Resp(
                status_code=200,
                body=[
                    {
                        "indicativo": "AAA",
                        "nombre": "A",
                        "provincia": "X",
                        "latitud": "400000N",
                        "longitud": "030000W",
                        "altitud": "10",
                    }
                ],
            )
        ],
    }
    ds = src.open("aemet_stations")
    assert "lon" in ds and "lat" in ds
    assert "AAA" in ds["station"].values


def test_unknown_dataset_raises(source_and_fake):
    src, _ = source_and_fake
    with pytest.raises(ValueError, match="unknown AEMET dataset"):
        src.open("aemet_nonsense")


# ---- subset by variables ------------------------------------------------


def test_rate_limit_spaces_requests():
    """Two back-to-back fetches should honour ``min_interval_s``."""
    import time

    fake = FakeClient(
        routes={
            "/valores/climatologicos/inventarioestaciones/todasestaciones": [
                _env_ok("https://fake.aemet/d1"),
                _env_ok("https://fake.aemet/d2"),
            ],
            "https://fake.aemet/d1": [_Resp(status_code=200, body=[])],
            "https://fake.aemet/d2": [_Resp(status_code=200, body=[])],
        }
    )
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        max_retries=0,
        max_workers=1,
        min_interval_s=0.2,
    )
    t0 = time.monotonic()
    src.list_stations()
    src.list_stations()
    elapsed = time.monotonic() - t0
    # Four hops (2 envelope + 2 data) × 0.2s gap = ≥0.6s between first
    # and last. We only need evidence the gate fired, not exact timing.
    assert elapsed >= 0.5, f"expected ≥0.5s, got {elapsed:.3f}s"


def test_rate_limit_zero_is_no_op():
    """``min_interval_s=0`` should not add artificial delay."""
    import time

    fake = FakeClient(
        routes={
            "/valores/climatologicos/inventarioestaciones/todasestaciones": [
                _env_ok("https://fake.aemet/d"),
            ],
            "https://fake.aemet/d": [_Resp(status_code=200, body=[])],
        }
    )
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        max_retries=0,
        max_workers=1,
        min_interval_s=0.0,
    )
    t0 = time.monotonic()
    src.list_stations()
    assert time.monotonic() - t0 < 0.1


def test_variable_subset_drops_others(source_and_fake):
    """Passing ``variables=[x]`` keeps only ``x`` in the result dataset."""
    del source_and_fake
    # Build a minimal dataset via get_hourly machinery with no rows; instead
    # call the subset helper directly.
    import xarray as xr

    from xr_toolz.data._src.aemet.source import _subset_variables

    ds = xr.Dataset(
        {
            "air_temperature": (("time",), [1.0]),
            "precipitation_amount": (("time",), [2.0]),
        },
        coords={"time": [0]},
    )
    out = _subset_variables(ds, ["air_temperature"])
    assert "air_temperature" in out
    assert "precipitation_amount" not in out
