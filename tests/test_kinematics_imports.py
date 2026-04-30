"""Import-surface tests for :mod:`xr_toolz.kinematics` (D9).

Guards against regressions in the kinematics taxonomy after the move
from ``xr_toolz.ocn`` to ``xr_toolz.kinematics`` (and the SSH /
validation re-homing under ``xr_toolz.geo``).
"""

from __future__ import annotations

import importlib

import pytest


_OCEAN_KINEMATICS = (
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

_KINEMATICS_OPS = (
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

_GEO_NEW_FUNCS = (
    "calculate_ssh_alongtrack",
    "calculate_ssh_unfiltered",
    "validate_ssh",
    "validate_velocity",
)

_GEO_NEW_OPS = ("CalculateSSHAlongtrack", "ValidateSSH", "ValidateVelocity")


# ---- New canonical surface ------------------------------------------------


@pytest.mark.parametrize("name", _OCEAN_KINEMATICS)
def test_kinematics_root_exposes_ocean_primitives(name):
    import xr_toolz.kinematics as kin

    assert callable(getattr(kin, name))


@pytest.mark.parametrize("name", _KINEMATICS_OPS)
def test_kinematics_operators_module_exposes_classes(name):
    import xr_toolz.kinematics.operators as ops

    cls = getattr(ops, name)
    assert isinstance(cls, type)
    assert cls.__name__ == name


@pytest.mark.parametrize("name", _GEO_NEW_FUNCS)
def test_geo_root_exposes_relocated_ssh_and_validation(name):
    import xr_toolz.geo as geo

    obj = getattr(geo, name)
    assert callable(obj)


@pytest.mark.parametrize("name", _GEO_NEW_OPS)
def test_geo_operators_exposes_relocated_ssh_and_validation_ops(name):
    import xr_toolz.geo.operators as ops

    cls = getattr(ops, name)
    assert isinstance(cls, type)
    assert cls.__name__ == name


def test_kinematics_domain_stubs_are_importable_and_empty():
    """The atmosphere / ice / remote_sensing kinematics submodules are
    reserved name surfaces — they import cleanly today and expose no
    public names. Future epics fill them in."""
    for sub in ("atmosphere", "ice", "remote_sensing"):
        mod = importlib.import_module(f"xr_toolz.kinematics._src.{sub}")
        public = [n for n in dir(mod) if not n.startswith("_")]
        assert public == [], (
            f"xr_toolz.kinematics._src.{sub} unexpectedly exports public names: "
            f"{public}"
        )


# ---- Removed top-level packages -----------------------------------------


def test_removed_top_level_domain_packages_are_gone():
    """F6.3: the legacy domain packages (``ocn``, ``atm``, ``ice``,
    ``rs``) were deleted outright. Pre-1.0 with no external users — no
    deprecation cycle needed."""
    for pkg in ("xr_toolz.ocn", "xr_toolz.atm", "xr_toolz.ice", "xr_toolz.rs"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(pkg)
