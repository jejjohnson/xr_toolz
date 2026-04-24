"""AemetArchive tests — incremental GeoParquet sync + coverage statistics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.data import AemetArchive, AEMETCredentials, AemetSource
from xr_toolz.data._src.aemet.archive import _merge_long
from xr_toolz.types import Station, StationCollection


def _make_monthly_ds(
    station_ids: list[str], dates: list[str], values: list[list[float]]
) -> xr.Dataset:
    times = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    arr = np.array(values, dtype=np.float64)
    return xr.Dataset(
        {
            "air_temperature_daily_mean": (
                ("station", "time"),
                arr,
                {"standard_name": "air_temperature"},
            ),
        },
        coords={
            "station": (("station",), station_ids),
            "time": (("time",), times),
        },
        attrs={"source": "aemet", "featureType": "timeSeries"},
    )


def _one_station_collection() -> StationCollection:
    return StationCollection.from_iter(
        [Station(id="s1", name="S1", lon=0.0, lat=40.0, source="aemet")]
    )


def _make_archive(tmp_path: Path, ds: xr.Dataset) -> AemetArchive:
    source = MagicMock(spec=AemetSource)
    source.list_stations.return_value = _one_station_collection()
    source.get_daily.return_value = ds
    archive = AemetArchive(root=tmp_path, source=source)
    archive.sync_stations()
    return archive


# ---- merge semantics -----------------------------------------------------


def test_merge_long_overwrites_overlap(tmp_path: Path):
    """Overlapping (station_id, time) rows should take the fresh value."""
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, 11.0]])
    )
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")

    # Overwrite the second day with a new value.
    archive.source.get_daily.return_value = _make_monthly_ds(
        ["s1"], ["2024-01-02", "2024-01-03"], [[99.0, 12.0]]
    )
    archive.sync("aemet_daily", since="2024-01-02", until="2024-01-03")

    gdf = archive.load("aemet_daily")
    by_day = gdf.set_index(pd.to_datetime(gdf["time"]))
    assert by_day.loc["2024-01-01", "air_temperature_daily_mean"] == 10.0
    assert by_day.loc["2024-01-02", "air_temperature_daily_mean"] == 99.0
    assert by_day.loc["2024-01-03", "air_temperature_daily_mean"] == 12.0


def test_merge_long_helper_direct():
    """``_merge_long`` returns a union with fresh winning on key overlap."""
    import geopandas as gpd
    from shapely.geometry import Point

    def _gdf(rows):
        return gpd.GeoDataFrame(
            rows,
            geometry=[Point(0.0, 0.0) for _ in rows],
            crs="EPSG:4326",
        )

    existing = _gdf(
        [
            {"station_id": "s1", "time": pd.Timestamp("2024-01-01"), "v": 1.0},
            {"station_id": "s1", "time": pd.Timestamp("2024-01-02"), "v": 2.0},
        ]
    )
    fresh = _gdf(
        [
            {"station_id": "s1", "time": pd.Timestamp("2024-01-02"), "v": 99.0},
            {"station_id": "s1", "time": pd.Timestamp("2024-01-03"), "v": 3.0},
        ]
    )
    merged = _merge_long(existing, fresh)
    by_day = merged.set_index(pd.to_datetime(merged["time"]))
    assert by_day.loc["2024-01-01", "v"] == 1.0
    assert by_day.loc["2024-01-02", "v"] == 99.0
    assert by_day.loc["2024-01-03", "v"] == 3.0


# ---- sync / load roundtrip ----------------------------------------------


def test_archive_sync_and_load_roundtrip(tmp_path: Path):
    """sync() writes GeoParquet; load() returns a GeoDataFrame with geometry."""
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, 11.0]])
    )
    out = archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    assert "air_temperature_daily_mean" in out

    gdf = archive.load("aemet_daily")
    assert set(gdf.columns) >= {
        "station_id",
        "time",
        "lon",
        "lat",
        "air_temperature_daily_mean",
        "geometry",
    }
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert len(gdf) == 2
    assert list(gdf["station_id"].unique()) == ["s1"]


def test_archive_load_dataset_reconstructs_cube(tmp_path: Path):
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, 11.0]])
    )
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    ds = archive.load_dataset("aemet_daily")
    assert ds.sizes == {"station": 1, "time": 2}
    assert ds["air_temperature_daily_mean"].sel(station="s1").values.tolist() == [
        10.0,
        11.0,
    ]


def test_archive_sync_is_idempotent(tmp_path: Path):
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, 11.0]])
    )
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    gdf = archive.load("aemet_daily")
    assert len(gdf) == 2  # not duplicated


def test_archive_sync_appends_new_window(tmp_path: Path):
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, 11.0]])
    )
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    archive.source.get_daily.return_value = _make_monthly_ds(
        ["s1"], ["2024-01-03", "2024-01-04"], [[12.0, 13.0]]
    )
    archive.sync("aemet_daily", since="2024-01-03", until="2024-01-04")

    gdf = archive.load("aemet_daily")
    assert len(gdf) == 4
    assert sorted(pd.to_datetime(gdf["time"]).dt.strftime("%Y-%m-%d").tolist()) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]


def test_archive_coverage_reports_gap_fraction(tmp_path: Path):
    ds = _make_monthly_ds(["s1"], ["2024-01-01", "2024-01-02"], [[10.0, float("nan")]])
    archive = _make_archive(tmp_path, ds)
    archive.sync("aemet_daily", since="2024-01-01", until="2024-01-02")
    rows = archive.coverage("aemet_daily")
    assert len(rows) == 1
    row = rows[0]
    assert row.station_id == "s1"
    assert row.n_timesteps == 1
    assert row.gap_fraction == 0.5


def test_archive_sync_rejects_unknown_preset(tmp_path: Path):
    archive = AemetArchive(
        root=tmp_path,
        source=AemetSource(credentials=AEMETCredentials(api_key="x")),
    )
    with pytest.raises(ValueError, match="unknown preset"):
        archive.sync("aemet_does_not_exist")


def test_archive_stations_gdf_has_geometry(tmp_path: Path):
    archive = _make_archive(
        tmp_path, _make_monthly_ds(["s1"], ["2024-01-01"], [[10.0]])
    )
    sgdf = archive.load_stations_geodataframe()
    assert sgdf.geometry.iloc[0].x == 0.0
    assert sgdf.geometry.iloc[0].y == 40.0
    assert sgdf.crs is not None and sgdf.crs.to_epsg() == 4326
