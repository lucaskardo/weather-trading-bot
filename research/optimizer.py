"""
Phase 3.2 — Parameter Optimizer.

Uses scipy.optimize.differential_evolution (global optimizer) to find
the parameter set that minimises out-of-sample Brier score from
walk-forward validation.

Replaces coordinate descent (which gets trapped in local minima).

Optimised parameters:
  base_std_f    — forecast uncertainty std-dev in °F    [2.5, 12.0]
  temp_T        — Platt temperature scaling              [0.7, 1.8]

Future extensions can add bias_correction_f, model_weights, etc.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research.walk_forward import walk_forward_brier

try:
    from scipy.optimize import differential_evolution, OptimizeResult
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# Parameter bounds: (min, max)
PARAM_BOUNDS = {
    "base_std_f": (2.5, 12.0),
    "temp_T":     (0.7, 1.8),
}

_PARAM_NAMES = list(PARAM_BOUNDS.keys())
_BOUNDS_LIST = [PARAM_BOUNDS[k] for k in _PARAM_NAMES]


class OptimizerResult:
    """Structured result from optimize_params."""

    def __init__(
        self,
        best_params: dict[str, float],
        best_brier: float,
        converged: bool,
        n_evaluations: int,
        elapsed_s: float,
    ):
        self.best_params = best_params
        self.best_brier = best_brier
        self.converged = converged
        self.n_evaluations = n_evaluations
        self.elapsed_s = elapsed_s

    def __repr__(self) -> str:
        return (
            f"OptimizerResult(brier={self.best_brier:.4f}, "
            f"params={self.best_params}, converged={self.converged})"
        )


def optimize_params(
    trade_log: list[dict[str, Any]],
    n_windows: int = 5,
    maxiter: int = 50,
    popsize: int = 20,
    seed: int = 42,
    tol: float = 0.001,
    current_params: dict[str, float] | None = None,
) -> OptimizerResult:
    """
    Run differential evolution to minimise walk-forward Brier score.

    Args:
        trade_log:      Chronologically sorted resolved trades.
        n_windows:      Walk-forward folds.
        maxiter:        Max DE generations.
        popsize:        DE population size multiplier.
        seed:           Random seed for reproducibility.
        tol:            Convergence tolerance.
        current_params: Starting point hint (used as fallback if scipy unavailable).

    Returns:
        OptimizerResult with best parameters and diagnostics.
    """
    if not _SCIPY_AVAILABLE:
        return _fallback_optimize(trade_log, n_windows, current_params)

    t0 = time.time()
    eval_count = 0

    def objective(x: list[float]) -> float:
        nonlocal eval_count
        eval_count += 1
        candidate = {name: val for name, val in zip(_PARAM_NAMES, x)}
        return walk_forward_brier(candidate, trade_log, n_windows)

    result: OptimizeResult = differential_evolution(
        objective,
        bounds=_BOUNDS_LIST,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=tol,
        polish=True,      # run L-BFGS-B refinement after DE
        workers=1,        # single-threaded (safe with SQLite)
    )

    best_params = {name: float(val) for name, val in zip(_PARAM_NAMES, result.x)}

    return OptimizerResult(
        best_params=best_params,
        best_brier=float(result.fun),
        converged=bool(result.success),
        n_evaluations=eval_count,
        elapsed_s=time.time() - t0,
    )


def _fallback_optimize(
    trade_log: list[dict[str, Any]],
    n_windows: int,
    current_params: dict[str, float] | None,
) -> OptimizerResult:
    """
    Coordinate descent fallback when scipy is not installed.
    Tries a coarse grid and returns the best found.
    """
    t0 = time.time()
    best_brier = float("inf")
    best_params = current_params or {"base_std_f": 5.0, "temp_T": 1.0}
    eval_count = 0

    grid = {
        "base_std_f": [2.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0],
        "temp_T":     [0.7, 0.85, 1.0, 1.2, 1.5, 1.8],
    }

    for std in grid["base_std_f"]:
        for T in grid["temp_T"]:
            candidate = {"base_std_f": std, "temp_T": T}
            score = walk_forward_brier(candidate, trade_log, n_windows)
            eval_count += 1
            if score < best_brier:
                best_brier = score
                best_params = dict(candidate)

    return OptimizerResult(
        best_params=best_params,
        best_brier=best_brier,
        converged=False,  # grid search never "converges"
        n_evaluations=eval_count,
        elapsed_s=time.time() - t0,
    )


def save_params(params: dict[str, float], path: Path | None = None) -> None:
    """Persist optimised parameters to JSON."""
    target = path or (Path(__file__).parent.parent / "data" / "calibrated_params.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(params, f, indent=2)


def load_params(path: Path | None = None) -> dict[str, float] | None:
    """Load persisted parameters. Returns None if file doesn't exist."""
    target = path or (Path(__file__).parent.parent / "data" / "calibrated_params.json")
    if not target.exists():
        return None
    with open(target) as f:
        return json.load(f)
