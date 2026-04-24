"""CDSInsituArchive round-trip and resumability tests.

All tests run **offline** — no live CDS calls. A ``FakeCdsClient``
writes a hand-authored zip of CSV rows to the target path; the archive
parses, merges and persists the frames to GeoParquet.

Exercises:

- Long-format CSV (``observed_variable`` / ``observation_value``).
- Wide-format CSV (one column per variable).
- Year-chunk resume via manifest.
- BBox client-side filter.
- Variable + station inventory sidecar round-trips.
- ``load_dataset`` returns a CF-ish ``(station, time)`` Dataset.
- Caller-typo inverted window raises ``ValueError``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from xr_toolz.data import CDSCredentials, CDSInsituArchive, CDSSource
from xr_toolz.types import BBox


# ---- zip builder helpers ------------------------------------------------


def _make_long_csv_zip(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a zip of one CSV in long (``observation_value``) layout."""
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", buf.getvalue())


def _make_wide_csv_zip(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a zip of one CSV in wide (one value column per variable) layout."""
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", buf.getvalue())


class _FixtureCdsClient:
    """Test double: writes a zip of CSVs to the ``target`` path on retrieve."""

    def __init__(self, zip_builder) -> None:
        self.zip_builder = zip_builder
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.calls.append((dataset_id, form, target))
        self.zip_builder(Path(target), form)


# ---- fixtures ------------------------------------------------------------


@pytest.fixture
def long_format_archive(tmp_path):
    """An archive wired to a client that emits a long-format zip per year."""

    def builder(target: Path, form: dict[str, Any]) -> None:
        # CDS in-situ sends ``year`` as a single string.
        year = form["year"]
        rows: list[dict[str, Any]] = []
        for sid, lon, lat in (
            ("STN-A", 5.0, 45.0),
            ("STN-B", 10.0, 50.0),
        ):
            for month in ("01", "07"):
                rows.append(
                    {
                        "station_id": sid,
                        "date_time": f"{year}-{month}-15T00:00:00Z",
                        "longitude": lon,
                        "latitude": lat,
                        "observed_variable": "air_temperature",
                        "observation_value": 15.0 + float(year) / 1000,
                        "units": "degC",
                        "quality_flag": "0",
                    }
                )
        _make_long_csv_zip(target, rows)

    client = _FixtureCdsClient(builder)
    source = CDSSource(
        credentials=CDSCredentials(url="u", key="k"),
        client=client,
    )
    archive = CDSInsituArchive(
        root=tmp_path,
        preset="cds_insitu_land",
        source=source,
        time_aggregation="daily",
    )
    return archive, client


@pytest.fixture
def wide_format_archive(tmp_path):
    """An archive wired to a client that emits a wide-format zip per year."""

    def builder(target: Path, form: dict[str, Any]) -> None:
        year = form["year"]
        rows = []
        for sid, lon, lat in (
            ("MAR-1", -20.0, 35.0),
            ("MAR-2", -10.0, 40.0),
        ):
            for month in ("03", "09"):
                rows.append(
                    {
                        "station_id": sid,
                        "datetime": f"{year}-{month}-10T12:00:00Z",
                        "longitude": lon,
                        "latitude": lat,
                        "air_temperature": 18.0,
                        "wind_speed": 7.5,
                        "water_temperature": 15.0,
                    }
                )
        _make_wide_csv_zip(target, rows)

    client = _FixtureCdsClient(builder)
    source = CDSSource(
        credentials=CDSCredentials(url="u", key="k"),
        client=client,
    )
    archive = CDSInsituArchive(
        root=tmp_path,
        preset="cds_insitu_marine",
        source=source,
        time_aggregation="daily",
    )
    return archive, client


# ---- construction validation --------------------------------------------


def test_archive_rejects_unknown_preset(tmp_path):
    with pytest.raises(ValueError, match="unknown preset"):
        CDSInsituArchive(
            root=tmp_path,
            preset="bogus",
            source=CDSSource(credentials=CDSCredentials(url="u", key="k")),
        )


def test_archive_rejects_unknown_time_aggregation(tmp_path):
    with pytest.raises(ValueError, match="time_aggregation"):
        CDSInsituArchive(
            root=tmp_path,
            preset="cds_insitu_land",
            source=CDSSource(credentials=CDSCredentials(url="u", key="k")),
            time_aggregation="weekly",
        )


def test_archive_dataset_id_wiring(tmp_path):
    archive = CDSInsituArchive(
        root=tmp_path,
        preset="cds_insitu_marine",
        source=CDSSource(credentials=CDSCredentials(url="u", key="k")),
    )
    assert archive.dataset_id == "insitu-observations-surface-marine"


# ---- long-format round trip ---------------------------------------------


def test_long_format_sync_and_load(long_format_archive):
    archive, client = long_format_archive
    archive.sync("2020-01-01", "2021-12-31")
    assert len(client.calls) == 2  # one per year-chunk

    gdf = archive.load()
    assert set(gdf["station_id"]) == {"STN-A", "STN-B"}
    # 2 stations × 2 years × 2 months = 8 rows
    assert len(gdf) == 8
    assert "geometry" in gdf.columns
    assert "variable" in gdf.columns
    assert set(gdf["variable"]) == {"air_temperature"}


def test_long_format_station_inventory_sidecar(long_format_archive):
    archive, _ = long_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    stations = archive.load_stations()
    assert set(stations["station_id"]) == {"STN-A", "STN-B"}
    # One row per station (deduplicated).
    assert len(stations) == 2


def test_long_format_manifest_tracks_completed_chunks(long_format_archive):
    archive, _ = long_format_archive
    archive.sync("2020-01-01", "2021-12-31")
    manifest = pd.read_json(archive.manifest_path, typ="series").to_dict()
    assert set(manifest["completed_chunks"]) == {"2020", "2021"}
    assert manifest["preset"] == "cds_insitu_land"
    assert manifest["time_aggregation"] == "daily"


def test_long_format_resume_is_noop(long_format_archive):
    archive, client = long_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    assert len(client.calls) == 1
    # Re-running the same window should not hit the client again.
    archive.sync("2020-01-01", "2020-12-31")
    assert len(client.calls) == 1


def test_long_format_overwrite_refetches(long_format_archive):
    archive, client = long_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    archive.sync("2020-01-01", "2020-12-31", overwrite=True)
    assert len(client.calls) == 2


def test_long_format_bbox_forwards_to_area(long_format_archive):
    """Archive passes ``bbox`` through to the CDS ``area`` form key."""
    archive, client = long_format_archive
    archive.sync(
        "2020-01-01",
        "2020-12-31",
        bbox=BBox(lon_min=0.0, lon_max=7.0, lat_min=40.0, lat_max=48.0),
    )
    _, form, _ = client.calls[0]
    # [N, W, S, E]
    assert form["area"] == [48.0, 0.0, 40.0, 7.0]


def test_long_format_load_time_filter(long_format_archive):
    archive, _ = long_format_archive
    archive.sync("2020-01-01", "2021-12-31")
    gdf = archive.load(start="2021-01-01")
    assert pd.Timestamp(gdf["time"].min()).year == 2021


# ---- wide-format round trip ---------------------------------------------


def test_wide_format_sync_and_load(wide_format_archive):
    archive, _ = wide_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    gdf = archive.load()
    assert set(gdf["station_id"]) == {"MAR-1", "MAR-2"}
    # Wide → long: 2 stations × 2 months × 3 variables = 12 rows.
    assert len(gdf) == 12
    assert set(gdf["variable"]) == {
        "air_temperature",
        "wind_speed",
        "water_temperature",
    }


def test_wide_format_load_dataset(wide_format_archive):
    archive, _ = wide_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    ds = archive.load_dataset()
    assert ds.attrs.get("featureType") == "timeSeries"
    assert "station" in ds.dims
    assert "time" in ds.dims
    assert "water_temperature" in ds


# ---- coverage -----------------------------------------------------------


def test_coverage_counts(long_format_archive):
    archive, _ = long_format_archive
    archive.sync("2020-01-01", "2020-12-31")
    cov = archive.coverage()
    assert {c.station_id for c in cov} == {"STN-A", "STN-B"}
    for c in cov:
        assert c.n_timesteps == 2  # Jan + Jul
        assert c.first is not None
        assert c.last is not None


def test_coverage_empty_archive(tmp_path):
    archive = CDSInsituArchive(
        root=tmp_path,
        preset="cds_insitu_land",
        source=CDSSource(credentials=CDSCredentials(url="u", key="k")),
    )
    assert archive.coverage() == []


# ---- error handling -----------------------------------------------------


def test_load_raises_when_empty(tmp_path):
    archive = CDSInsituArchive(
        root=tmp_path,
        preset="cds_insitu_land",
        source=CDSSource(credentials=CDSCredentials(url="u", key="k")),
    )
    with pytest.raises(FileNotFoundError, match="empty"):
        archive.load()


def test_sync_inverted_window_raises(long_format_archive):
    archive, _ = long_format_archive
    with pytest.raises(ValueError, match="start year"):
        archive.sync("2022-01-01", "2020-01-01")
