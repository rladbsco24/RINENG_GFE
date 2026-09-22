"""Fixed force-equilibrium endpoint contracts used by production figures.

This module only declares the two task-class settings fixed by the manuscript:
``alpha=10 m^-1`` with isotropic curvature for Single, and
``alpha_force=3000`` with ``diag(1, 1, 10)`` curvature for Triple.  Both use a
10,000-iteration cap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .gorkov_core import MethodSpec
from .multitrap import MultitrapObjectiveConfig, regularized_fe_double_trap_config


FETask = Literal["Single", "Triple"]
CurvatureWeight = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

SINGLE_FE_CURVATURE_WEIGHT: CurvatureWeight = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
TRIPLE_FE_CURVATURE_WEIGHT: CurvatureWeight = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 10.0),
)


@dataclass(frozen=True)
class FixedFEEndpointContract:
    task: FETask
    alpha: float
    curvature_weight_override: CurvatureWeight
    maxiter: int = 10_000

    def __post_init__(self) -> None:
        expected_alpha = {"Single": 10.0, "Triple": 3000.0}[self.task]
        if not np.isclose(float(self.alpha), expected_alpha, rtol=0.0, atol=0.0):
            raise ValueError(
                f"the fixed {self.task} FE alpha is {expected_alpha:g}"
            )
        expected_weight = {
            "Single": SINGLE_FE_CURVATURE_WEIGHT,
            "Triple": TRIPLE_FE_CURVATURE_WEIGHT,
        }[self.task]
        weight = np.asarray(self.curvature_weight_override, dtype=np.float64)
        if not np.array_equal(weight, np.asarray(expected_weight, dtype=np.float64)):
            raise ValueError(
                f"the fixed {self.task} FE curvature weight is {expected_weight!r}"
            )
        if int(self.maxiter) != 10_000:
            raise ValueError("the fixed production FE cap is 10000 iterations")


SINGLE_FE_ENDPOINT = FixedFEEndpointContract(
    "Single",
    alpha=10.0,
    curvature_weight_override=SINGLE_FE_CURVATURE_WEIGHT,
)
TRIPLE_FE_ENDPOINT = FixedFEEndpointContract(
    "Triple",
    alpha=3000.0,
    curvature_weight_override=TRIPLE_FE_CURVATURE_WEIGHT,
)
FIXED_FE_ENDPOINTS: dict[FETask, FixedFEEndpointContract] = {
    "Single": SINGLE_FE_ENDPOINT,
    "Triple": TRIPLE_FE_ENDPOINT,
}


def fixed_fe_endpoint(task: str) -> FixedFEEndpointContract:
    """Return one immutable production endpoint contract."""

    key = str(task).strip().lower()
    if key == "single":
        return SINGLE_FE_ENDPOINT
    if key == "triple":
        return TRIPLE_FE_ENDPOINT
    raise ValueError("task must be 'Single' or 'Triple'")


def fixed_single_fe_method_spec() -> MethodSpec:
    """Build the standard-sign Single FE objective at the fixed weight."""

    contract = SINGLE_FE_ENDPOINT
    return MethodSpec(
        name="Force-Equilibrium",
        alpha_per_m=float(contract.alpha),
        beta_curvature_per_pa=0.0,
        pressure_mode="abs",
        curvature_weight=np.asarray(
            contract.curvature_weight_override, dtype=np.float64
        ),
    )


def fixed_triple_fe_config(
    *,
    stage_maxiters: tuple[int, ...] = (250, 9750),
    stage_factors: tuple[float, ...] = (0.05, 0.01),
    beta_pressure: float = 5.0e-5,
    gamma_uniformity: float = 1.0,
    gtol: float = 0.0,
    report_gradient_tol: float = 1.0e-3,
) -> MultitrapObjectiveConfig:
    """Build the fixed Triple FE formulation and axial-curvature weighting.

    The established two-stage smoothing schedule is retained by default.  A
    caller may redistribute the 10,000 iterations between stages, but cannot
    change the total production cap through this constructor.
    """

    if not stage_maxiters or sum(int(value) for value in stage_maxiters) != 10_000:
        raise ValueError(
            "Triple FE stage_maxiters must sum to the fixed cap of 10000"
        )
    if len(stage_factors) != len(stage_maxiters):
        raise ValueError("stage_factors and stage_maxiters must have equal length")
    base = regularized_fe_double_trap_config(
        alpha_force=float(TRIPLE_FE_ENDPOINT.alpha),
        beta_pressure=float(beta_pressure),
        gamma_uniformity=float(gamma_uniformity),
        stage_maxiters=tuple(int(value) for value in stage_maxiters),
        stage_factors=tuple(float(value) for value in stage_factors),
        gtol=float(gtol),
        report_gradient_tol=float(report_gradient_tol),
    )
    return replace(
        base,
        curvature_weight_override=TRIPLE_FE_ENDPOINT.curvature_weight_override,
    )


__all__ = [
    "FETask",
    "CurvatureWeight",
    "FIXED_FE_ENDPOINTS",
    "FixedFEEndpointContract",
    "SINGLE_FE_CURVATURE_WEIGHT",
    "SINGLE_FE_ENDPOINT",
    "TRIPLE_FE_CURVATURE_WEIGHT",
    "TRIPLE_FE_ENDPOINT",
    "fixed_fe_endpoint",
    "fixed_single_fe_method_spec",
    "fixed_triple_fe_config",
]
