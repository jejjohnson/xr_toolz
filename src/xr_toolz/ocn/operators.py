"""Deprecated — operator surface moved per D9.

- Kinematics operators (``Streamfunction``, ``GeostrophicVelocities``,
  ``KineticEnergy``, ``RelativeVorticity``, …) →
  :mod:`xr_toolz.kinematics.operators`.
- SSH / velocity operators (``ValidateSSH``, ``ValidateVelocity``,
  ``CalculateSSHAlongtrack``) → :mod:`xr_toolz.geo.operators`.

Names imported through :mod:`xr_toolz.ocn.operators` continue to resolve
for one release with a :class:`DeprecationWarning` (PEP 562). This
module will be removed in the next minor.
"""

from __future__ import annotations

import warnings
from typing import Any


_DEPRECATED_KINEMATICS_OPS = (
    "AbsoluteVorticity",
    "Advection",
    "AgeostrophicVelocities",
    "BruntVaisalaFrequency",
    "CoriolisNormalized",
    "CurvatureVorticity",
    "Divergence",
    "EddyKineticEnergy",
    "Enstrophy",
    "Frontogenesis",
    "GeostrophicVelocities",
    "HorizontalVelocityMagnitude",
    "KineticEnergy",
    "LapseRate",
    "MixedLayerDepth",
    "OkuboWeiss",
    "PotentialVorticityBarotropic",
    "RelativeVorticity",
    "ShearStrain",
    "ShearVorticity",
    "StrainMagnitude",
    "Streamfunction",
    "TensorStrain",
    "VelocityMagnitude",
)

_DEPRECATED_GEO_OPS = (
    "CalculateSSHAlongtrack",
    "ValidateSSH",
    "ValidateVelocity",
)


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name in _DEPRECATED_KINEMATICS_OPS:
        warnings.warn(
            f"xr_toolz.ocn.operators.{name} is deprecated; "
            f"import from xr_toolz.kinematics.operators instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module("xr_toolz.kinematics.operators"), name)
    if name in _DEPRECATED_GEO_OPS:
        warnings.warn(
            f"xr_toolz.ocn.operators.{name} is deprecated; "
            f"import from xr_toolz.geo.operators instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module("xr_toolz.geo.operators"), name)
    raise AttributeError(f"module 'xr_toolz.ocn.operators' has no attribute {name!r}")


__all__: list[str] = []
