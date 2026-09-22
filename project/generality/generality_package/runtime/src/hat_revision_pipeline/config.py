from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunConfig:
    mode: str = "quick"
    # Numerical settings whose cache identity must be reproduced.  In normal
    # solver modes this equals ``mode``; plot-only may target either an
    # existing quick or full cache without changing any producer payload.
    budget_mode: str = "quick"
    random_seed: int = 260828
    array_side: int = 16
    pitch_m: float = 0.010
    frequency_hz: float = 40_000.0
    # Central r* of the corrected 7 x 7 tilted target chart.  Keeping the
    # canonical Vortex benchmark at the chart centre makes Figures 1--3 and 5
    # refer to the same physical target rather than two nearby examples.
    single_target_m: tuple[float, float, float] = (0.0, 0.0, 0.050)
    stencil_m: float = 0.0005
    conventional_maxiter: int = 10_000
    fe_maxiter: int = 10_000
    sweep_maxiter: int = 160
    multitrap_maxiter: int = 10_000
    # One positive stationarity tolerance is used by every newly executed
    # solver.  Recovered benchmark tables retain their original recorded
    # settings, but new SOTA/sweep/multitrap runs must not silently disagree.
    gtol: float = 1.0e-8
    slice_points: int = 81
    force_slice_points: int = 7
    exact_lmax: int = 5

    @property
    def full(self) -> bool:
        return self.budget_mode == "full"

    @property
    def cache_only(self) -> bool:
        return self.mode == "plot-only"

    @classmethod
    def for_mode(
        cls,
        mode: str,
        *,
        cache_source_mode: str | None = None,
    ) -> "RunConfig":
        if mode not in {"quick", "full", "plot-only"}:
            raise ValueError(mode)
        if mode == "plot-only":
            budget_mode = "quick" if cache_source_mode is None else cache_source_mode
            if budget_mode not in {"quick", "full"}:
                raise ValueError("cache_source_mode must be 'quick' or 'full'")
        else:
            if cache_source_mode is not None and cache_source_mode != mode:
                raise ValueError(
                    "cache_source_mode is only configurable for plot-only mode"
                )
            budget_mode = mode
        if budget_mode == "full":
            return cls(
                mode=mode,
                budget_mode="full",
                conventional_maxiter=10_000,
                fe_maxiter=10_000,
                sweep_maxiter=2_000,
                multitrap_maxiter=10_000,
                slice_points=181,
                force_slice_points=21,
                exact_lmax=8,
            )
        return cls(mode=mode, budget_mode="quick")


def square_positions(side: int = 16, pitch_m: float = 0.010) -> np.ndarray:
    axis = np.arange(side, dtype=float) - (side - 1.0) / 2.0
    return np.array([(pitch_m * x, pitch_m * y, 0.0) for x in axis for y in axis], dtype=float)


def square_normals(side: int = 16) -> np.ndarray:
    result = np.zeros((side * side, 3), dtype=float)
    result[:, 2] = 1.0
    return result


def locate_project_root(start: str | Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "hat_revision_notebook").exists():
            return candidate
        if candidate.name == "hat_revision_notebook" and (candidate / "src").exists():
            return candidate.parent
    raise FileNotFoundError("Could not locate the HAT revision project root")
