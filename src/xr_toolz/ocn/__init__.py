"""Deprecated — content moved per D9.

The ocean-physics surface that used to live here has been split across
two new homes:

- Kinematic / geostrophic primitives → :mod:`xr_toolz.kinematics`
  (sub-organized by domain under :mod:`xr_toolz.kinematics._src.ocean`).
- SSH composition helpers → :mod:`xr_toolz.geo` (``calculate_ssh_alongtrack``,
  ``calculate_ssh_unfiltered``).
- SSH / velocity attribute harmonization → :mod:`xr_toolz.geo`
  (``validate_ssh``, ``validate_velocity``).

Names imported through :mod:`xr_toolz.ocn` continue to resolve for one
release with a :class:`DeprecationWarning` (PEP 562). The module itself
will be removed in the next minor.
"""

from __future__ import annotations

import warnings
from typing import Any


_DEPRECATED_KINEMATICS = (
    "absolute_vorticity",
    "advection",
    "ageostrophic_velocities",
    "brunt_vaisala_frequency",
    "coriolis_normalized",
    "coriolis_parameter",
    "curvature_vorticity",
    "divergence",
    "eddy_kinetic_energy",
    "enstrophy",
    "frontogenesis",
    "geostrophic_velocities",
    "horizontal_velocity_magnitude",
    "kinetic_energy",
    "lapse_rate",
    "mixed_layer_depth",
    "okubo_weiss",
    "potential_vorticity_barotropic",
    "relative_vorticity",
    "shear_strain",
    "shear_vorticity",
    "strain_magnitude",
    "streamfunction",
    "tensor_strain",
    "velocity_magnitude",
)

_DEPRECATED_SSH = ("calculate_ssh_alongtrack", "calculate_ssh_unfiltered")
_DEPRECATED_VALIDATION = ("validate_ssh", "validate_velocity")


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name in _DEPRECATED_KINEMATICS:
        warnings.warn(
            f"xr_toolz.ocn.{name} is deprecated; "
            f"import from xr_toolz.kinematics instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module("xr_toolz.kinematics._src.ocean"), name)
    if name in _DEPRECATED_SSH:
        warnings.warn(
            f"xr_toolz.ocn.{name} is deprecated; "
            f"import from xr_toolz.geo instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module("xr_toolz.geo._src.altimetry"), name)
    if name in _DEPRECATED_VALIDATION:
        warnings.warn(
            f"xr_toolz.ocn.{name} is deprecated; "
            f"import from xr_toolz.geo instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module("xr_toolz.geo._src.validation"), name)
    raise AttributeError(f"module 'xr_toolz.ocn' has no attribute {name!r}")


__all__: list[str] = []
