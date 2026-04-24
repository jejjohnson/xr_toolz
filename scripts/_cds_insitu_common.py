"""Shared setup for the CDS in-situ scrape scripts.

Keeps loguru + archive wiring out of the individual scripts so each
one stays a short, readable period list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from xr_toolz.data import CDSInsituArchive, CDSSource


def _default_scratch_root() -> Path:
    """Where CDS in-situ scrapes land.

    Resolution order (first match wins):

    1. ``CDS_INSITU_SCRATCH_ROOT`` environment variable.
    2. ``XR_TOOLZ_CDS_ROOT`` environment variable (alias).
    3. ``./scratch/cds_insitu`` under the CWD — portable safe default.
    """
    for var in ("CDS_INSITU_SCRATCH_ROOT", "XR_TOOLZ_CDS_ROOT"):
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    return Path.cwd() / "scratch" / "cds_insitu"


SCRATCH_ROOT = _default_scratch_root()
LOG_ROOT = Path(__file__).resolve().parent.parent / ".logs"


def setup_logging(name: str) -> Path:
    """Configure loguru: stderr + per-script rotating file log.

    Returns the log-file path so tmux users can ``tail -f`` it.
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", rotation="20 MB", retention=5)
    return log_path


def build_archive(
    preset: str,
    *,
    subdir: str | None = None,
    time_aggregation: str = "daily",
    usage_restrictions: str = "unrestricted",
    data_quality: str = "passed",
) -> CDSInsituArchive:
    """Build a :class:`CDSInsituArchive` rooted at ``SCRATCH_ROOT/subdir``.

    Args:
        preset: ``"cds_insitu_land"`` or ``"cds_insitu_marine"``.
        subdir: Sub-directory under :data:`SCRATCH_ROOT`. Defaults to
            ``preset``, so each preset/time_aggregation combo gets its
            own tree without stepping on neighbours.
        time_aggregation: ``"sub_daily"`` / ``"daily"`` / ``"monthly"``.
        usage_restrictions: CDS ``usage_restrictions`` form key.
        data_quality: CDS ``data_quality`` form key.
    """
    root = SCRATCH_ROOT / (subdir or f"{preset}_{time_aggregation}")
    source = CDSSource()
    archive = CDSInsituArchive(
        root=root,
        preset=preset,
        source=source,
        time_aggregation=time_aggregation,
        usage_restrictions=usage_restrictions,
        data_quality=data_quality,
    )
    logger.info(f"archive root: {root}")
    logger.info(
        f"source: preset={preset} time_aggregation={time_aggregation} "
        f"usage_restrictions={usage_restrictions} data_quality={data_quality}"
    )
    return archive
