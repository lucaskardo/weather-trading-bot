"""
Phase 2.2 — Capital Allocator.

Softmax allocation across strategies weighted by their scorecard scores.
Hard caps: 5% min, 40% max per strategy.
Available capital = bankroll × 0.8 (keep 20% buffer).
"""

from __future__ import annotations

import math
from typing import Optional

from shared.params import Params, PARAMS

_MIN_ALLOC_FRAC = 0.05   # 5% floor per strategy
_MAX_ALLOC_FRAC = 0.40   # 40% ceiling per strategy
_CAPITAL_BUFFER = 0.80   # deploy at most 80% of bankroll


def allocate(
    scores: dict[str, Optional[float]],
    bankroll: float,
    params: Params = PARAMS,
) -> dict[str, float]:
    """
    Compute USD capital allocation for each strategy.

    Args:
        scores:   Map of strategy_name → score (0–100). None = not yet scored.
        bankroll: Total available bankroll in USD.
        params:   Params (uses router_temperature).

    Returns:
        Map of strategy_name → allocated USD amount.
        Strategies with None score receive _MIN_ALLOC_FRAC of available capital.
    """
    available = bankroll * _CAPITAL_BUFFER
    if available <= 0 or not scores:
        return {}

    # Separate scored and unscored strategies
    scored = {k: v for k, v in scores.items() if v is not None}
    unscored = [k for k, v in scores.items() if v is None]

    if not scored:
        # All unscored: equal split capped at min allocation
        per_strategy = available / len(scores)
        return {name: per_strategy for name in scores}

    fracs = _softmax_fracs(scored, params.router_temperature)
    fracs = _apply_caps(fracs)   # may sum < 1.0 when all strategies hit their bounds

    # Multiply directly — do NOT renormalize (renormalization undoes the caps)
    allocations: dict[str, float] = {name: frac * available for name, frac in fracs.items()}

    for name in unscored:
        allocations[name] = _MIN_ALLOC_FRAC * available

    return allocations


def _softmax_fracs(
    scored: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    """Compute softmax fractions from 0–100 scores."""
    T = max(temperature, 0.1)  # prevent division by zero
    exp_scores = {name: math.exp(score / T) for name, score in scored.items()}
    total = sum(exp_scores.values())
    if total == 0:
        n = len(scored)
        return {name: 1.0 / n for name in scored}
    return {name: e / total for name, e in exp_scores.items()}


def _apply_caps(fracs: dict[str, float]) -> dict[str, float]:
    """
    Iteratively fix strategies at their bounds and redistribute remaining
    weight to unconstrained strategies.

    Returns capped fracs that may sum to less than 1.0 when all strategies
    simultaneously hit their upper bound (e.g. 2 strategies with MAX=40%
    can only absorb 80% of the weight).  Callers must NOT renormalize.
    """
    if not fracs:
        return {}

    result = dict(fracs)

    for _ in range(len(fracs) + 2):
        fixed: dict[str, float] = {}
        free: dict[str, float] = {}

        for name, frac in result.items():
            if frac >= _MAX_ALLOC_FRAC:
                fixed[name] = _MAX_ALLOC_FRAC
            elif frac <= _MIN_ALLOC_FRAC:
                fixed[name] = _MIN_ALLOC_FRAC
            else:
                free[name] = frac

        if not free:
            result = {**fixed}
            break

        remaining = 1.0 - sum(fixed.values())
        if remaining <= 0:
            result = {**fixed, **{k: 0.0 for k in free}}
            break

        free_total = sum(free.values())
        if free_total <= 0:
            result = {**fixed}
            break

        scale = remaining / free_total
        rescaled_free = {k: v * scale for k, v in free.items()}

        result = {**fixed, **rescaled_free}

        # Converged if all free strategies are within bounds
        if all(_MIN_ALLOC_FRAC <= v <= _MAX_ALLOC_FRAC for v in rescaled_free.values()):
            break

    return result
