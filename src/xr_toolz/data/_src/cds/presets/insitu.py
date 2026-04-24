"""CDS in-situ observation presets (surface-land + surface-marine).

Both products return a **zip archive of CSV files** — one row per
``(station, time, variable)``, filtered server-side by the supplied
``variable`` / ``time_aggregation`` / ``usage_restrictions`` /
``data_quality`` / ``year`` / ``month`` / ``day`` keys.

They follow the :data:`INSITU` form profile: zip output, no
``product_type`` key, no ``area`` filter. Callers must pass
``time_aggregation`` via ``**extras`` on every download:

.. code-block:: python

    source.download(
        "insitu-observations-surface-land",
        output=...,
        variables=["air_temperature", "relative_humidity"],
        time=TimeRange.parse("2020-01-01", "2020-01-31"),
        time_aggregation="daily",          # required
        usage_restrictions="unrestricted", # optional (default unrestricted)
        data_quality="passed",             # optional (default passed)
    )

The :class:`~xr_toolz.data._src.cds.archive.CDSInsituArchive` handles
the unzip + CSV parse + long-format GeoParquet storage path so callers
don't have to.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.data._src.cds.profiles import INSITU
from xr_toolz.types import (
    AIR_TEMPERATURE,
    DEW_POINT_TEMPERATURE,
    DOWNWARD_LONGWAVE_RADIATION,
    DOWNWARD_SHORTWAVE_RADIATION,
    MEAN_SEA_LEVEL_PRESSURE_HPA,
    PRECIPITATION_AMOUNT,
    RELATIVE_HUMIDITY,
    SEA_LEVEL_PRESSURE,
    SEA_SURFACE_TEMPERATURE_INSITU,
    SUNSHINE_DURATION,
    SURFACE_PRESSURE_HPA,
    SURFACE_SNOW_THICKNESS,
    TOTAL_CLOUD_COVER,
    WATER_VAPOUR_PRESSURE,
    WAVE_FROM_DIRECTION,
    WAVE_PERIOD,
    WAVE_SIGNIFICANT_HEIGHT,
    WIND_FROM_DIRECTION,
    WIND_SPEED,
    WIND_SPEED_OF_GUST,
    BBox,
)


INSITU_LAND = DatasetInfo(
    dataset_id="insitu-observations-surface-land",
    source="cds",
    title="Global land surface atmospheric variables from 1755 to present",
    kind=DatasetKind.STATIONS,
    variables=(
        AIR_TEMPERATURE,
        DEW_POINT_TEMPERATURE,
        RELATIVE_HUMIDITY,
        WATER_VAPOUR_PRESSURE,
        SURFACE_PRESSURE_HPA,
        MEAN_SEA_LEVEL_PRESSURE_HPA,
        WIND_SPEED,
        WIND_FROM_DIRECTION,
        WIND_SPEED_OF_GUST,
        PRECIPITATION_AMOUNT,
        DOWNWARD_LONGWAVE_RADIATION,
        DOWNWARD_SHORTWAVE_RADIATION,
        TOTAL_CLOUD_COVER,
        SUNSHINE_DURATION,
        SURFACE_SNOW_THICKNESS,
    ),
    spatial_coverage=BBox.global_(),
    temporal_coverage=("1755-01-01", "present"),
    license="Varies — see the dataset's licences tab on the CDS portal.",
    form_profile=INSITU,
    notes=(
        "Zip-of-CSV per request. Requires ``time_aggregation`` "
        "∈ {'sub_daily', 'daily', 'monthly'}. "
        "Use CDSInsituArchive for resumable multi-year scrapes."
    ),
)


INSITU_MARINE = DatasetInfo(
    dataset_id="insitu-observations-surface-marine",
    source="cds",
    title="Global marine surface observations from 1978 to present",
    kind=DatasetKind.STATIONS,
    variables=(
        AIR_TEMPERATURE,
        DEW_POINT_TEMPERATURE,
        RELATIVE_HUMIDITY,
        SEA_LEVEL_PRESSURE,
        SEA_SURFACE_TEMPERATURE_INSITU,
        WIND_SPEED,
        WIND_FROM_DIRECTION,
        WIND_SPEED_OF_GUST,
        WAVE_SIGNIFICANT_HEIGHT,
        WAVE_PERIOD,
        WAVE_FROM_DIRECTION,
    ),
    spatial_coverage=BBox.global_(),
    temporal_coverage=("1978-01-01", "present"),
    license="Varies — see the dataset's licences tab on the CDS portal.",
    form_profile=INSITU,
    notes=(
        "Zip-of-CSV from buoys / drifters / fixed platforms / ships. "
        "Requires ``time_aggregation`` on every request; wave variables "
        "are only available on specific platform subsets."
    ),
)


INSITU_DATASETS: dict[str, DatasetInfo] = {
    d.dataset_id: d for d in (INSITU_LAND, INSITU_MARINE)
}
