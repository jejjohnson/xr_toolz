"""Deep-import shim for SSH-composition helpers that moved to
:mod:`xr_toolz.geo._src.altimetry` (D9).

Kept only so that legacy deep imports such as
``from xr_toolz.ocn._src.ssh import calculate_ssh_alongtrack`` resolve
for one release while downstream code migrates.
"""

from xr_toolz.geo._src.altimetry import (
    calculate_ssh_alongtrack,
    calculate_ssh_unfiltered,
)


__all__ = ["calculate_ssh_alongtrack", "calculate_ssh_unfiltered"]
