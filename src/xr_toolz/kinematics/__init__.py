"""Kinematic and geostrophic quantities, sub-organized by domain (D9).

Layer-0 primitives live in :mod:`xr_toolz.kinematics._src.<domain>`
(``ocean``, ``atmosphere``, ``ice``, ``remote_sensing``) and are
re-exported here. Tier C ``Operator`` wrappers live in
:mod:`xr_toolz.kinematics.operators`.

Today only :mod:`xr_toolz.kinematics._src.ocean` is populated; the other
domain submodules are empty placeholders reserved for future content.
"""

from xr_toolz.kinematics._src.ocean import (
    absolute_vorticity,
    advection,
    ageostrophic_velocities,
    brunt_vaisala_frequency,
    coriolis_normalized,
    coriolis_parameter,
    curvature_vorticity,
    divergence,
    eddy_kinetic_energy,
    enstrophy,
    frontogenesis,
    geostrophic_velocities,
    horizontal_velocity_magnitude,
    kinetic_energy,
    lapse_rate,
    mixed_layer_depth,
    okubo_weiss,
    potential_vorticity_barotropic,
    relative_vorticity,
    shear_strain,
    shear_vorticity,
    strain_magnitude,
    streamfunction,
    tensor_strain,
    velocity_magnitude,
)


__all__ = [
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
]
