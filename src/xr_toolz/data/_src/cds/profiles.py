"""Per-dataset-family form profiles for the CDS adapter.

The CDS API speaks several dialects that share a ``cdsapi.Client.retrieve``
envelope but differ in which form keys are accepted:

- **Reanalysis-style** (ERA5 single/pressure levels, ERA5-Land) uses
  ``format``, ``product_type``, ``area``, ``year/month/day``, and
  optionally ``pressure_level``.
- **In-situ-style** (``insitu-observations-surface-land``,
  ``insitu-observations-surface-marine``) returns a **zip of CSV** per
  request, has no ``product_type``, no ``area`` key, and requires a
  ``time_aggregation`` alongside the date fields.

A :class:`CDSFormProfile` makes these differences explicit so the
single :class:`~xr_toolz.data._src.cds.source.CDSSource` can target both
without branching on dataset id in the adapter. Each
:class:`~xr_toolz.data._src.base.DatasetInfo` entry for a CDS dataset
carries a ``form_profile`` field that the adapter consults when
building the request form.

Adding a new family is: define one :class:`CDSFormProfile` constant
here, attach it to the relevant :class:`DatasetInfo` entries, and add
preset data if required. No changes to the adapter class are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CDSFormProfile:
    """Shape of a CDS ``retrieve()`` form for a dataset family.

    Attributes:
        family: Short slug identifying the family (``"reanalysis"``,
            ``"insitu"``, ...). Used in logs / tests.
        format: Default output format for this family. Overridable via
            ``extras["format"]`` at call time.
        includes_product_type: Whether the ``product_type`` key is part
            of this family's form.
        uses_area: Whether ``bbox`` should be serialised to ``area``.
            In-situ products accept no spatial filter and must set this
            to ``False`` — caller-side spatial filtering happens after
            the download.
        uses_pressure_level: Whether a :class:`PressureLevels` argument
            should be serialised to ``pressure_level``.
        required_extras: Keys the caller must supply via ``**extras``
            (e.g. ``"time_aggregation"`` for in-situ). Empty for
            reanalysis.
    """

    family: str
    format: str
    includes_product_type: bool = False
    uses_area: bool = True
    uses_pressure_level: bool = False
    required_extras: tuple[str, ...] = field(default_factory=tuple)


REANALYSIS = CDSFormProfile(
    family="reanalysis",
    format="netcdf",
    includes_product_type=True,
    uses_pressure_level=True,
)
"""Profile for ERA5 / ERA5-Land style gridded reanalyses.

``format=netcdf``, ``product_type=reanalysis`` (source-configurable),
``area``, ``year/month/day``, and optional ``pressure_level``.
"""


INSITU = CDSFormProfile(
    family="insitu",
    format="zip",
    includes_product_type=False,
    uses_area=False,
    required_extras=("time_aggregation",),
)
"""Profile for CDS in-situ observation archives.

Zip-of-CSV output, no ``product_type``, no ``area`` filter. Caller must
pass ``time_aggregation`` ("daily", "sub_daily", or "monthly") via
``extras`` on each ``download()`` call. ``usage_restrictions``,
``data_quality``, and ``variable`` remain caller-supplied and flow
through via the standard form-building path.
"""


def resolve_profile(
    dataset_id: str, datasets: dict[str, Any] | None = None
) -> CDSFormProfile:
    """Return the :class:`CDSFormProfile` for ``dataset_id``.

    ``datasets`` defaults to :data:`CDS_DATASETS` (imported lazily to
    avoid a circular import). Unknown dataset ids fall back to
    :data:`REANALYSIS` so existing call sites that don't use the
    catalog stay bit-compatible.
    """
    if datasets is None:
        from xr_toolz.data._src.cds.catalog import CDS_DATASETS

        datasets = CDS_DATASETS
    info = datasets.get(dataset_id)
    if info is None:
        return REANALYSIS
    profile: CDSFormProfile | None = getattr(info, "form_profile", None)
    return profile or REANALYSIS
