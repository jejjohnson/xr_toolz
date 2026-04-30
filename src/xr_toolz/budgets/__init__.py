"""Conservation-budget diagnostics — V4.2 / V4.3.

Two layers, mirroring the rest of xr_toolz:

- **Layer 0** — pure-function primitives:
  :func:`control_volume_integral`, :func:`boundary_flux`,
  :func:`budget_residual`, plus the tracer-/momentum-specific
  reductions :func:`heat_budget_residual`,
  :func:`salt_budget_residual`, :func:`volume_budget_residual`,
  :func:`kinetic_energy_budget_residual`.
- **Layer 1** — :class:`Operator` wrappers under
  :mod:`xr_toolz.budgets.operators`.

All operators take grid metrics as **explicit constructor arguments**
(see V4.4 / D16). Use :func:`xr_toolz.calc.grid_metrics_from_coords` to
derive ``volume_metrics`` / ``face_metrics`` from a Dataset's coords if
the model output does not already ship them.
"""

from xr_toolz.budgets._src.flux import boundary_flux
from xr_toolz.budgets._src.heat import heat_budget_residual
from xr_toolz.budgets._src.ke import kinetic_energy_budget_residual
from xr_toolz.budgets._src.residual import budget_residual
from xr_toolz.budgets._src.salt import salt_budget_residual
from xr_toolz.budgets._src.volume import control_volume_integral
from xr_toolz.budgets._src.volume_budget import volume_budget_residual
from xr_toolz.budgets.operators import (
    BoundaryFlux,
    BudgetResidual,
    ControlVolumeIntegral,
    HeatBudgetResidual,
    KineticEnergyBudgetResidual,
    SaltBudgetResidual,
    VolumeBudgetResidual,
)


__all__ = [
    "BoundaryFlux",
    "BudgetResidual",
    "ControlVolumeIntegral",
    "HeatBudgetResidual",
    "KineticEnergyBudgetResidual",
    "SaltBudgetResidual",
    "VolumeBudgetResidual",
    "boundary_flux",
    "budget_residual",
    "control_volume_integral",
    "heat_budget_residual",
    "kinetic_energy_budget_residual",
    "salt_budget_residual",
    "volume_budget_residual",
]
