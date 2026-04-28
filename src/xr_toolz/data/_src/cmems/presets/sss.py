"""CMEMS sea-surface-salinity multi-observation presets (MULTIOBS)."""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import DENS, SOS, BBox


SSS_DATASETS: dict[str, DatasetInfo] = {
    "cmems_obs-mob_glo_phy-sss_my_multi_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (reprocessed, 0.125°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_my_multi_P1M": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1M",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (reprocessed, 0.125°, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (NRT, 0.125°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_nrt_multi_P1M": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1M",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (NRT, 0.125°, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    # ---- MULTIOBS L4 weekly OI (reprocessed) ---------------------------
    "cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W",
        source="cmems",
        title=(
            "MULTIOBS L4 — Global SSS optimal interpolation "
            "(reprocessed, 0.25°, weekly)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(SOS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    # ---- L3 SMOS single-mission (ascending / descending) ---------------
    "cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D",
        source="cmems",
        title="SMOS L3 — Global SSS, ascending orbits (0.25°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes=(
            "Single-mission L3 (SMOS) on ascending orbits. Files use "
            "the variable name 'Sea_Surface_Salinity'."
        ),
    ),
    "cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D",
        source="cmems",
        title="SMOS L3 — Global SSS, descending orbits (0.25°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes=(
            "Single-mission L3 (SMOS) on descending orbits. Files use "
            "the variable name 'Sea_Surface_Salinity'."
        ),
    ),
}
