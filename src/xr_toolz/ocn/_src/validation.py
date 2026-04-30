"""Deep-import shim for SSH / velocity validation that moved to
:mod:`xr_toolz.geo._src.validation` (D9).

Kept only so that legacy deep imports such as
``from xr_toolz.ocn._src.validation import validate_ssh`` resolve for
one release while downstream code migrates.
"""

from xr_toolz.geo._src.validation import validate_ssh, validate_velocity


__all__ = ["validate_ssh", "validate_velocity"]
