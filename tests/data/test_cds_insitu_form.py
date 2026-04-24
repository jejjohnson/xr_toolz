"""CDS in-situ form construction tests.

Exercise the profile-driven ``_build_form`` path:

- In-situ presets emit ``format=zip``, no ``product_type``, no ``area``.
- Missing ``time_aggregation`` raises a clear ``ValueError``.
- Caller-supplied ``time_aggregation`` / ``usage_restrictions`` /
  ``data_quality`` flow through unchanged.
- ERA5 regression: reanalysis presets still emit ``format=netcdf`` +
  ``product_type`` + ``area``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xr_toolz.data import (
    INSITU,
    REANALYSIS,
    CDSCredentials,
    CDSFormProfile,
    CDSSource,
)
from xr_toolz.data._src.cds.profiles import resolve_profile
from xr_toolz.types import BBox, TimeRange


class FakeCdsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.calls.append((dataset_id, form, target))
        Path(target).write_text("stub")


@pytest.fixture
def cds_source() -> tuple[CDSSource, FakeCdsClient]:
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="https://example", key="abc"),
        client=fake,
    )
    return src, fake


# ---- profile lookup ------------------------------------------------------


def test_resolve_profile_insitu_land():
    assert resolve_profile("insitu-observations-surface-land").family == "insitu"


def test_resolve_profile_insitu_marine():
    assert resolve_profile("insitu-observations-surface-marine").family == "insitu"


def test_resolve_profile_reanalysis_default():
    assert resolve_profile("reanalysis-era5-single-levels").family == "reanalysis"


def test_resolve_profile_unknown_falls_back_to_reanalysis():
    assert resolve_profile("totally-made-up-dataset").family == "reanalysis"


# ---- form: in-situ -------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_id",
    ["insitu-observations-surface-land", "insitu-observations-surface-marine"],
)
def test_insitu_form_shape(cds_source, tmp_path, dataset_id):
    src, fake = cds_source
    src.download(
        dataset_id,
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-06-01", "2020-06-30"),
        time_aggregation="daily",
        usage_restrictions="unrestricted",
        data_quality="passed",
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "zip"
    assert "product_type" not in form
    assert "area" not in form
    assert form["time_aggregation"] == "daily"
    assert form["usage_restrictions"] == "unrestricted"
    assert form["data_quality"] == "passed"
    assert form["variable"] == ["air_temperature"]
    assert form["year"] == ["2020"]
    assert form["month"] == ["06"]


def test_insitu_bbox_ignored_in_form(cds_source, tmp_path):
    """BBox should not appear in the in-situ form — CDS has no area filter."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    assert "area" not in form


def test_insitu_missing_time_aggregation_raises(cds_source, tmp_path):
    """Clear error when the caller forgets ``time_aggregation``."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="time_aggregation"):
        src.download(
            "insitu-observations-surface-land",
            tmp_path / "obs.zip",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
        )


@pytest.mark.parametrize("agg", ["sub_daily", "daily", "monthly"])
def test_insitu_all_time_aggregations_supported(cds_source, tmp_path, agg):
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation=agg,
    )
    _, form, _ = fake.calls[0]
    assert form["time_aggregation"] == agg


def test_insitu_variable_alias_resolution(cds_source, tmp_path):
    """Registered Variable names translate to CDS aliases on the way out."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["wind_speed", "precipitation_amount"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    assert form["variable"] == ["wind_speed_at_10m", "accumulated_precipitation"]


# ---- form: reanalysis regression ----------------------------------------


def test_reanalysis_regression_single_levels(cds_source, tmp_path):
    """ERA5 single-levels form is unchanged by the profile refactor."""
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "era5.nc",
        variables=["t2m", "u10"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-29", "2020-02-02"),
    )
    _, form, _ = fake.calls[0]
    assert form["variable"] == ["2m_temperature", "10m_u_component_of_wind"]
    assert form["area"] == [60.0, -10.0, 30.0, 40.0]
    assert form["year"] == ["2020"]
    assert set(form["month"]) == {"01", "02"}
    assert form["format"] == "netcdf"
    assert form["product_type"] == "reanalysis"


def test_source_format_override_wins_over_profile(tmp_path):
    """Explicit ``format`` on CDSSource beats the profile default."""
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="u", key="k"),
        client=fake,
        format="grib",
    )
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.grib",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "grib"


def test_extras_format_wins_over_source_and_profile(cds_source, tmp_path):
    """Per-call ``format`` in extras beats both source and profile."""
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.nc",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        format="grib",
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "grib"


def test_insitu_format_is_zip_even_with_source_default_none(cds_source, tmp_path):
    """Profile default (``zip``) applies when no override is set."""
    src, fake = cds_source
    assert src.format is None  # default construction
    src.download(
        "insitu-observations-surface-marine",
        tmp_path / "m.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "zip"


# ---- profile objects -----------------------------------------------------


def test_profile_identity():
    """Sanity check that the module constants are ``CDSFormProfile`` instances."""
    assert isinstance(INSITU, CDSFormProfile)
    assert isinstance(REANALYSIS, CDSFormProfile)
    assert INSITU.format == "zip"
    assert REANALYSIS.format == "netcdf"
    assert "time_aggregation" in INSITU.required_extras
    assert REANALYSIS.includes_product_type is True
    assert INSITU.uses_area is False


def test_open_zip_format_raises_clear_error(cds_source):
    """Zip-format datasets are not xarray-readable; ``open()`` must say so."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="zip bundle"):
        src.open(
            "insitu-observations-surface-land",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
            time_aggregation="daily",
        )
