"""Shared setup for the AEMET scrape scripts.

Keeps loguru + archive wiring out of the individual scripts so each one
stays a short, readable list of period windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from xr_toolz.data import AemetArchive, AemetSource


# Where observations land. The parent script imports + uses this.
SCRATCH_ROOT = Path(
    "/home/azureuser/cloudfiles/code/Users/adm.jjohnson72/scratch/aemet"
)

# Where logs go — under the repo, git-ignored.
LOG_ROOT = Path(__file__).resolve().parent.parent / ".logs"


def setup_logging(name: str) -> Path:
    """Configure loguru: stderr + per-script file with default formatting.

    Returns the log-file path for reference (e.g. so tmux users can
    ``tail -f`` it independently of the running process).
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", rotation="20 MB", retention=5)
    return log_path


def build_archive(
    subdir: str,
    *,
    min_interval_s: float = 0.5,
    max_workers: int = 2,
    max_retries: int = 6,
    timeout_s: float = 30.0,
) -> AemetArchive:
    """Build a paced :class:`AemetArchive` pointed at ``SCRATCH_ROOT/subdir``.

    Defaults tuned for long-running scrapes: 120 req/min global pacing
    (~20% headroom under AEMET's ~150/min cap), two workers, six
    retries on transient failures. The archive is idempotent so
    re-running after Ctrl-C or kill is safe.
    """
    root = SCRATCH_ROOT / subdir
    source = AemetSource(
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_workers=max_workers,
        min_interval_s=min_interval_s,
    )
    archive = AemetArchive(root=root, source=source)
    logger.info(f"archive root: {root}")
    logger.info(
        f"source: max_workers={max_workers}, "
        f"min_interval_s={min_interval_s}, timeout_s={timeout_s}"
    )
    return archive
