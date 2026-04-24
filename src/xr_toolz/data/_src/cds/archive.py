"""Local GeoParquet mirror of CDS in-situ observations.

Why an archive class at all?
----------------------------
The CDS in-situ surface-land / surface-marine products are *station*
data — unstructured ``(station, time, variable)`` samples wrapped in a
zip-of-CSV. Long-running scrapes need:

- **Idempotent, resumable downloads** — the CDS queue is slow, and
  requests that cross years are rejected or truncated. We chunk by
  year and record completed chunks in a ``manifest.json`` so re-runs
  are no-ops for already-ingested years.
- **Station geometry** — rows must keep their ``(lon, lat)`` pair so
  downstream tools (``geopandas``, ``duckdb-spatial``, PostGIS) can
  filter spatially without a secondary inventory.
- **Dtype-preserving storage** — flag columns, instrument ids, and
  unit strings need to survive round-tripping.

GeoParquet, long-format, year-partitioned solves all three — same
pattern as :class:`~xr_toolz.data._src.aemet.archive.AemetArchive`.

Layout
------

::

    <root>/
      <preset>/
        manifest.json      # completed chunks + provenance
        data.parquet       # long-format (station_id, time, lon, lat, ...)
        stations.parquet   # distinct (station_id, lon, lat, ...) sidecar

One ``CDSInsituArchive`` instance corresponds to one ``(dataset_id,
time_aggregation)`` pair. Keep separate archives for
``time_aggregation="daily"`` vs ``"sub_daily"`` vs ``"monthly"``.
"""

from __future__ import annotations

import contextlib
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from xr_toolz.data._src.cds.source import CDSSource
from xr_toolz.types import BBox, TimeRange, Variable


PRESET_TO_DATASET: dict[str, str] = {
    "cds_insitu_land": "insitu-observations-surface-land",
    "cds_insitu_marine": "insitu-observations-surface-marine",
}

_VALID_TIME_AGGREGATIONS = frozenset({"sub_daily", "daily", "monthly"})


@dataclass(frozen=True)
class ArchiveCoverage:
    """Per-station span of the archive for one preset."""

    preset: str
    station_id: str
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    n_timesteps: int


class CDSInsituArchive:
    """Append-only local mirror of a CDS in-situ preset.

    Args:
        root: Directory that will hold one subtree per preset.
        preset: Either ``"cds_insitu_land"`` or ``"cds_insitu_marine"``.
        source: :class:`CDSSource` used for downloads. If ``None`` a
            default one is constructed (env-var / ``.env`` credentials).
        time_aggregation: One of ``"sub_daily"``, ``"daily"``,
            ``"monthly"`` — determines the CDS form key of the same
            name and the on-disk partition cadence.
        variables: Restrict fetches to these variables. ``None`` fetches
            every variable advertised in the preset's ``DatasetInfo`` —
            CDS in-situ rejects requests with no ``variable`` key, so
            we fall back rather than send an empty list.
    """

    def __init__(
        self,
        root: Path,
        preset: str,
        source: CDSSource | None = None,
        *,
        time_aggregation: str = "daily",
        variables: tuple[Variable, ...] | None = None,
    ) -> None:
        if preset not in PRESET_TO_DATASET:
            raise ValueError(
                f"unknown preset {preset!r}; expected one of "
                f"{sorted(PRESET_TO_DATASET)}"
            )
        if time_aggregation not in _VALID_TIME_AGGREGATIONS:
            raise ValueError(
                f"unknown time_aggregation {time_aggregation!r}; expected "
                f"one of {sorted(_VALID_TIME_AGGREGATIONS)}"
            )
        self.root = Path(root)
        self.preset = preset
        self.dataset_id = PRESET_TO_DATASET[preset]
        self.source = source if source is not None else CDSSource()
        # Marine ignores ``time_aggregation``; we keep the attribute
        # for bookkeeping (manifest / path derivation) but don't send
        # it on marine requests.
        self.time_aggregation = time_aggregation
        self.variables = variables

    # ---- paths -----------------------------------------------------------

    @property
    def preset_root(self) -> Path:
        return self.root / self.preset

    @property
    def data_path(self) -> Path:
        return self.preset_root / "data.parquet"

    @property
    def stations_path(self) -> Path:
        return self.preset_root / "stations.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.preset_root / "manifest.json"

    # ---- sync ------------------------------------------------------------

    def sync(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        bbox: BBox | None = None,
        since: str | pd.Timestamp | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Fetch year-chunks spanning ``[start, end]`` and append to disk.

        Args:
            start: Earliest year to fetch. Inclusive.
            end: Latest year to fetch. Inclusive.
            bbox: Optional server-side spatial filter. CDS in-situ
                accepts an ``area`` key, so we forward this on the
                request rather than filtering after download.
            since: If given, skip years strictly earlier than this.
                Overrides the "completed_chunks" manifest entry.
            overwrite: Re-download years already present in the manifest.

        Returns:
            The concatenated freshly-fetched data (empty if everything
            was already cached).
        """
        # CDS in-situ always takes one year per request (land + marine).
        year_chunks = _year_chunks(start, end)

        manifest = self._load_manifest()
        done = set() if overwrite else set(manifest.get("completed_chunks", []))
        since_yr = _yr(since) if since is not None else None

        fetched: list[pd.DataFrame] = []
        for year_a, year_b in year_chunks:
            key = _chunk_key(year_a, year_b)
            if key in done:
                continue
            if since_yr is not None and year_b < since_yr:
                continue
            # ``bbox`` is server-side for CDS in-situ (profile.uses_area),
            # so we send it on the request and skip the client-side filter.
            df = self._fetch_chunk(year_a, year_b, bbox=bbox)
            if not df.empty:
                self._append(df)
                fetched.append(df)
            manifest.setdefault("completed_chunks", []).append(key)
            manifest["last_sync_utc"] = datetime.now(UTC).isoformat()
            manifest["preset"] = self.preset
            manifest["time_aggregation"] = self.time_aggregation
            self._write_manifest(manifest)

        return pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()

    def _fetch_chunk(
        self, year_a: int, year_b: int, *, bbox: BBox | None = None
    ) -> pd.DataFrame:
        """Download one year (``year_a == year_b``) and return a long frame.

        CDS in-situ accepts a **single** year per request; year-chunks
        wider than one are rejected. Callers (``sync``) loop.
        """
        if year_a != year_b:
            raise ValueError(
                f"CDS in-situ accepts one year per request, got [{year_a}, {year_b}]"
            )
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        time_range = TimeRange.parse(f"{year_a}-01-01", f"{year_a}-12-31")
        extras: dict[str, Any] = {"month": months, "day": days}
        # Marine doesn't accept ``time_aggregation``; only send it for land.
        if self.preset == "cds_insitu_land":
            extras["time_aggregation"] = self.time_aggregation
        # CDS in-situ rejects requests without a ``variable`` key even
        # though the schema marks it optional; fall back to the preset's
        # full variable list when the caller didn't override.
        variables: list[str | Variable] | None
        if self.variables is not None:
            variables = list(self.variables)
        else:
            info = self.source.describe(self.dataset_id)
            variables = list(info.variables) if info.variables else None

        tmp_zip = self.preset_root / f"_tmp_{year_a}.zip"
        try:
            self.source.download(
                self.dataset_id,
                tmp_zip,
                variables=variables,
                bbox=bbox,
                time=time_range,
                **extras,
            )
            return _parse_zip_to_long(tmp_zip)
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()

    # ---- load ------------------------------------------------------------

    def load(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ):
        """Open the archive as a :class:`geopandas.GeoDataFrame`.

        Optionally filter to ``[start, end]`` inclusive.
        """
        import geopandas as gpd

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"archive for {self.preset!r} is empty; run .sync(...) first."
            )
        gdf = gpd.read_parquet(self.data_path)
        if start is not None:
            gdf = gdf[gdf["time"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            gdf = gdf[gdf["time"] <= pd.Timestamp(end, tz="UTC")]
        return gdf

    def load_dataset(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> xr.Dataset:
        """Open the archive as an ``(station, time)`` xarray Dataset."""
        gdf = self.load(start=start, end=end)
        return _long_to_dataset(gdf, source=self.preset)

    def load_stations(self):
        """Distinct ``(station_id, lon, lat)`` inventory as a GeoDataFrame."""
        import geopandas as gpd

        if not self.stations_path.exists():
            raise FileNotFoundError(
                f"station inventory for {self.preset!r} is empty; run .sync(...) first."
            )
        return gpd.read_parquet(self.stations_path)

    def coverage(self) -> list[ArchiveCoverage]:
        """Per-station first/last/n_timesteps for the archive."""
        try:
            gdf = self.load()
        except FileNotFoundError:
            return []
        out: list[ArchiveCoverage] = []
        for sid, sub in gdf.groupby("station_id", sort=True):
            times = pd.to_datetime(sub["time"])
            if len(times) == 0:
                first: pd.Timestamp | None = None
                last: pd.Timestamp | None = None
            else:
                first = cast("pd.Timestamp | None", pd.Timestamp(times.min()))
                last = cast("pd.Timestamp | None", pd.Timestamp(times.max()))
            out.append(
                ArchiveCoverage(
                    preset=self.preset,
                    station_id=str(sid),
                    first=first,
                    last=last,
                    n_timesteps=len(sub),
                )
            )
        return out

    # ---- manifest --------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # ---- write / merge ---------------------------------------------------

    def _append(self, fresh: pd.DataFrame) -> None:
        """Merge ``fresh`` into the archive; fresh wins on (station, time, variable)."""
        import geopandas as gpd
        from shapely.geometry import Point

        # Ensure geometry + CRS are present before writing.
        if "geometry" not in fresh.columns:
            fresh = fresh.copy()
            fresh["geometry"] = [
                Point(x, y) if pd.notna(x) and pd.notna(y) else None
                for x, y in zip(fresh["lon"], fresh["lat"], strict=True)
            ]
        fresh_gdf = gpd.GeoDataFrame(fresh, geometry="geometry", crs="EPSG:4326")
        if self.data_path.exists():
            existing = gpd.read_parquet(self.data_path)
            merged = _merge_long(existing, fresh_gdf)
        else:
            merged = fresh_gdf
        merged.to_parquet(self.data_path)

        # Refresh the station inventory sidecar.
        stations = (
            merged[["station_id", "lon", "lat", "geometry"]]
            .drop_duplicates(subset=["station_id"])
            .reset_index(drop=True)
        )
        stations_gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")
        stations_gdf.to_parquet(self.stations_path)


# ---- chunking helpers ---------------------------------------------------


def _yr(value: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(value).year)


def _chunk_key(year_a: int, year_b: int) -> str:
    return f"{year_a}" if year_a == year_b else f"{year_a}-{year_b}"


def _year_chunks(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[tuple[int, int]]:
    """Return a list of ``(year_a, year_b)`` inclusive chunks of width 1."""
    y0, y1 = _yr(start), _yr(end)
    if y0 > y1:
        raise ValueError(f"start year {y0} is after end year {y1}")
    return [(y, y) for y in range(y0, y1 + 1)]


# ---- zip / CSV parsing --------------------------------------------------


_STATION_ID_CANDIDATES = (
    "station_id",
    "primary_station_id",
    "station",
    "station_name",
    "wmo_id",
    "platform_id",
    "platform",
    "call_sign",
)
_LON_CANDIDATES = ("longitude", "lon", "lon_deg", "x")
_LAT_CANDIDATES = ("latitude", "lat", "lat_deg", "y")
_TIME_CANDIDATES = ("date_time", "datetime", "time", "date", "report_timestamp")
_VARIABLE_COL_CANDIDATES = (
    "observed_variable",
    "variable",
    "observed_variable_code",
    "parameter",
)
_VALUE_COL_CANDIDATES = ("observation_value", "value", "observed_value")


def _parse_zip_to_long(zip_path: Path) -> pd.DataFrame:
    """Unpack a CDS in-situ zip bundle and return a long-format DataFrame.

    The CDS in-situ CSVs come in one of two flavours:

    - **Long already**: one row per ``(station, time, variable)`` with
      ``observed_variable`` / ``observation_value`` columns.
    - **Wide**: one row per ``(station, time)`` with one column per
      variable.

    This function handles both by sniffing the column set and always
    returns a long-format frame with columns
    ``[station_id, time, lon, lat, variable, value]``.
    """
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame(
                columns=["station_id", "time", "lon", "lat", "variable", "value"]
            )
        for name in csv_names:
            with zf.open(name) as handle:
                # CDS in-situ CSVs start with a ``#``-prefixed preamble
                # (licence, dataset url, time extent, variables list);
                # ``pandas`` must skip it before hitting the header row.
                df = pd.read_csv(handle, comment="#", low_memory=False)
            frames.append(_normalise_csv(df))
    if not frames:
        return pd.DataFrame(
            columns=["station_id", "time", "lon", "lat", "variable", "value"]
        )
    combined = pd.concat(frames, ignore_index=True)
    # Drop rows with unparseable key columns.
    combined = combined.dropna(subset=["station_id", "time"])
    return combined


def _normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce one CSV slice to the long-format schema."""
    cols_lower = {c: c.lower() for c in df.columns}
    rev = {v: k for k, v in cols_lower.items()}

    station_col = _pick(rev, _STATION_ID_CANDIDATES)
    lon_col = _pick(rev, _LON_CANDIDATES)
    lat_col = _pick(rev, _LAT_CANDIDATES)
    time_col = _pick(rev, _TIME_CANDIDATES)
    variable_col = _pick(rev, _VARIABLE_COL_CANDIDATES)
    value_col = _pick(rev, _VALUE_COL_CANDIDATES)

    if station_col is None or time_col is None:
        # Can't normalise without keys — return empty so the caller
        # notices via empty combined frame.
        return pd.DataFrame(
            columns=["station_id", "time", "lon", "lat", "variable", "value"]
        )

    out = pd.DataFrame(
        {
            "station_id": df[station_col].astype(str),
            "time": pd.to_datetime(df[time_col], errors="coerce", utc=True),
            "lon": _safe_float(df[lon_col]) if lon_col else np.nan,
            "lat": _safe_float(df[lat_col]) if lat_col else np.nan,
        }
    )

    if variable_col is not None and value_col is not None:
        out["variable"] = df[variable_col].astype(str)
        out["value"] = _safe_float(df[value_col])
        # Carry through units / quality flag columns if present.
        for passthrough in ("units", "quality_flag", "report_type", "source_id"):
            match = rev.get(passthrough)
            if match is not None:
                out[passthrough] = df[match]
        return out

    # Wide layout: melt everything that isn't a key column.
    key_cols = {station_col, lon_col, lat_col, time_col} - {None}
    value_cols = [c for c in df.columns if c not in key_cols]
    numeric = df[value_cols].apply(pd.to_numeric, errors="coerce")
    # Attach keys so ``melt`` preserves the mapping without a manual join.
    numeric = numeric.assign(
        _station=out["station_id"].to_numpy(),
        _time=out["time"].to_numpy(),
        _lon=out["lon"].to_numpy() if "lon" in out else np.nan,
        _lat=out["lat"].to_numpy() if "lat" in out else np.nan,
    )
    melted = numeric.melt(
        id_vars=["_station", "_time", "_lon", "_lat"],
        var_name="variable",
        value_name="value",
    )
    return pd.DataFrame(
        {
            "station_id": melted["_station"].astype(str),
            "time": pd.to_datetime(melted["_time"], utc=True),
            "lon": melted["_lon"].astype(float),
            "lat": melted["_lat"].astype(float),
            "variable": melted["variable"].astype(str),
            "value": melted["value"].astype(float),
        }
    )


def _pick(lower_to_orig: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    """Return the original-case column name matching any candidate, or ``None``."""
    for cand in candidates:
        hit = lower_to_orig.get(cand)
        if hit is not None:
            return hit
    return None


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


# ---- merge / dataset conversion ----------------------------------------


def _merge_long(existing, fresh):
    """Union existing + fresh rows; fresh wins on ``(station, time, variable)``."""
    import geopandas as gpd

    key_cols = [c for c in ("station_id", "time", "variable") if c in existing.columns]
    if not key_cols:
        key_cols = ["station_id", "time"]
    key_fresh = pd.MultiIndex.from_frame(fresh[key_cols])
    key_existing = pd.MultiIndex.from_frame(existing[key_cols])
    keep_mask = ~key_existing.isin(key_fresh)
    survivors = existing[keep_mask]
    merged = pd.concat([survivors, fresh], ignore_index=True)
    merged = merged.sort_values(key_cols, kind="stable").reset_index(drop=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def _as_naive_datetime64(times) -> np.ndarray:
    """Drop tz so xarray stops warning; in-situ timestamps are UTC already."""
    ts = pd.to_datetime(list(times), utc=True).tz_localize(None)
    return ts.to_numpy().astype("datetime64[ns]")


def _long_to_dataset(gdf, source: str) -> xr.Dataset:
    """Pivot a long GeoDataFrame into a ``(station, time)`` Dataset.

    One ``DataArray`` per variable; non-numeric passthrough columns are
    preserved with ``dtype=object``.
    """
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["station_id"] = df["station_id"].astype(str)

    if "variable" in df.columns and "value" in df.columns:
        pivoted = df.pivot_table(
            index="station_id",
            columns=["time", "variable"],
            values="value",
            aggfunc="first",
        )
        stations = pivoted.index.astype(str).tolist()
        times = sorted({t for t, _ in pivoted.columns})
        variables = sorted({v for _, v in pivoted.columns})
        data: dict[str, xr.DataArray] = {}
        for var in variables:
            arr = np.full((len(stations), len(times)), np.nan, dtype=np.float64)
            for i, sid in enumerate(stations):
                for j, t in enumerate(times):
                    with contextlib.suppress(KeyError):
                        arr[i, j] = pivoted.loc[sid, (t, var)]
            data[var] = xr.DataArray(arr, dims=("station", "time"))
        return xr.Dataset(
            data,
            coords={
                "station": ("station", stations),
                "time": ("time", _as_naive_datetime64(times)),
            },
            attrs={"source": source, "featureType": "timeSeries"},
        )

    # Wide layout already: one row per (station, time), variables in columns.
    value_cols = [
        c for c in df.columns if c not in {"station_id", "time", "lon", "lat"}
    ]
    pivoted = df.pivot_table(
        index="station_id",
        columns="time",
        values=value_cols,
        aggfunc="first",
    )
    stations = pivoted.index.astype(str).tolist()
    times = sorted({t for _, t in pivoted.columns})
    data = {}
    for var in value_cols:
        sub = pivoted.get(var)
        if sub is None:
            continue
        sub = sub.reindex(columns=times)
        if pd.api.types.is_numeric_dtype(df[var]):
            arr = sub.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            arr = sub.to_numpy(dtype=object)
        data[var] = xr.DataArray(arr, dims=("station", "time"))
    return xr.Dataset(
        data,
        coords={
            "station": ("station", stations),
            "time": ("time", _as_naive_datetime64(times)),
        },
        attrs={"source": source, "featureType": "timeSeries"},
    )
