"""Import-surface tests for :mod:`xr_toolz.kinematics` (D9).

Guards against regressions in the kinematics taxonomy after the move
from ``xr_toolz.ocn`` to ``xr_toolz.kinematics`` (and the SSH /
validation re-homing under ``xr_toolz.geo``).
"""

from __future__ import annotations

import importlib
import warnings

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


# ---- Legacy deprecation surface ------------------------------------------


@pytest.mark.parametrize("name", _OCEAN_KINEMATICS)
def test_legacy_ocn_kinematics_imports_warn_but_resolve(name):
    import xr_toolz.ocn as ocn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = getattr(ocn, name)

    assert callable(obj)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, f"expected DeprecationWarning on legacy xr_toolz.ocn.{name}"
    assert "xr_toolz.kinematics" in str(deprecations[0].message)


@pytest.mark.parametrize(
    "name", ("calculate_ssh_alongtrack", "calculate_ssh_unfiltered")
)
def test_legacy_ocn_ssh_imports_warn_and_point_at_geo(name):
    import xr_toolz.ocn as ocn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = getattr(ocn, name)

    assert callable(obj)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert "xr_toolz.geo" in str(deprecations[0].message)


@pytest.mark.parametrize("name", ("validate_ssh", "validate_velocity"))
def test_legacy_ocn_validation_imports_warn_and_point_at_geo(name):
    import xr_toolz.ocn as ocn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = getattr(ocn, name)

    assert callable(obj)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert "xr_toolz.geo" in str(deprecations[0].message)


@pytest.mark.parametrize("name", _KINEMATICS_OPS)
def test_legacy_ocn_operators_kinematics_warn_but_resolve(name):
    import xr_toolz.ocn.operators as ops

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = getattr(ops, name)

    assert isinstance(cls, type)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert "xr_toolz.kinematics.operators" in str(deprecations[0].message)


@pytest.mark.parametrize("name", _GEO_NEW_OPS)
def test_legacy_ocn_operators_ssh_validation_warn_and_point_at_geo(name):
    import xr_toolz.ocn.operators as ops

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = getattr(ops, name)

    assert isinstance(cls, type)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert "xr_toolz.geo.operators" in str(deprecations[0].message)


def test_atm_ice_rs_top_level_packages_are_gone():
    """F6.3: the empty domain stubs were deleted outright (no
    deprecation, since they had no public surface)."""
    for pkg in ("xr_toolz.atm", "xr_toolz.ice", "xr_toolz.rs"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(pkg)
