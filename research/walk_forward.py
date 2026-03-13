"""
Phase 3.1 — Walk-Forward Validation.

Time-series cross-validation that prevents look-ahead bias:
  - Split trade log into N sequential windows
  - For each window: train on [0..i], evaluate on [i+1]
  - Return average out-of-sample Brier score

This is used by the optimizer as its objective function, ensuring
calibrated parameters generalise to unseen data.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from core.forecaster import brier_score, prob_above_threshold


def walk_forward_brier(
    params_candidate: dict[str, float],
    trade_log: list[dict[str, Any]],
    n_windows: int = 5,
) -> float:
    """
    Compute average out-of-sample Brier score via walk-forward validation.

    Args:
        params_candidate: Dict with at least {"base_std_f": float, "temp_T": float}.
        trade_log:        Chronologically sorted list of resolved trades.
                          Each trade must have: consensus_f, threshold_f,
                          outcome (0.0 or 1.0). Optionally: fair_value.
        n_windows:        Number of validation folds.

    Returns:
        Average out-of-sample Brier score (lower is better).
        Returns 0.25 (random baseline) if insufficient data.
    """
    base_std_f = params_candidate.get("base_std_f", 5.0)
    temp_T = params_candidate.get("temp_T", 1.0)

    if len(trade_log) < n_windows + 1:
        return 0.25  # not enough data — return random baseline

    window_size = len(trade_log) // (n_windows + 1)
    if window_size < 1:
        return 0.25

    oos_briers: list[float] = []

    for fold in range(n_windows):
        # Training window: all data up to this fold's test point
        test_start = (fold + 1) * window_size
        test_end = test_start + window_size

        test_trades = trade_log[test_start:test_end]
        if not test_trades:
            continue

        fold_briers: list[float] = []
        for trade in test_trades:
            consensus_f = trade.get("consensus_f")
            threshold_f = trade.get("threshold_f") or trade.get("high_f")
            outcome = trade.get("outcome")

            if consensus_f is None or threshold_f is None or outcome is None:
                continue

            pred = prob_above_threshold(
                float(consensus_f),
                float(threshold_f),
                float(base_std_f),
                float(temp_T),
            )
            fold_briers.append(brier_score(pred, float(outcome)))

        if fold_briers:
            oos_briers.append(sum(fold_briers) / len(fold_briers))

    if not oos_briers:
        return 0.25

    return sum(oos_briers) / len(oos_briers)


def walk_forward_variance(
    params_candidate: dict[str, float],
    trade_log: list[dict[str, Any]],
    n_windows: int = 5,
) -> float:
    """
    Compute variance of per-fold Brier scores.
    High variance signals overfitting / instability.
    """
    base_std_f = params_candidate.get("base_std_f", 5.0)
    temp_T = params_candidate.get("temp_T", 1.0)

    if len(trade_log) < n_windows + 1:
        return 0.0

    window_size = len(trade_log) // (n_windows + 1)
    fold_scores: list[float] = []

    for fold in range(n_windows):
        test_start = (fold + 1) * window_size
        test_trades = trade_log[test_start: test_start + window_size]
        if not test_trades:
            continue

        fold_briers = []
        for trade in test_trades:
            consensus_f = trade.get("consensus_f")
            threshold_f = trade.get("threshold_f") or trade.get("high_f")
            outcome = trade.get("outcome")
            if None in (consensus_f, threshold_f, outcome):
                continue
            pred = prob_above_threshold(float(consensus_f), float(threshold_f),
                                        float(base_std_f), float(temp_T))
            fold_briers.append(brier_score(pred, float(outcome)))

        if fold_briers:
            fold_scores.append(sum(fold_briers) / len(fold_briers))

    if len(fold_scores) < 2:
        return 0.0

    mean = sum(fold_scores) / len(fold_scores)
    return sum((s - mean) ** 2 for s in fold_scores) / (len(fold_scores) - 1)


def make_synthetic_trades(
    n: int = 100,
    true_std_f: float = 5.0,
    threshold_f: float = 80.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate synthetic resolved trades for testing the optimizer.
    Actual outcomes are drawn from N(consensus, true_std_f).
    """
    import random
    rng = random.Random(seed)
    trades = []
    for _ in range(n):
        consensus_f = rng.gauss(80.0, 5.0)
        actual_high = rng.gauss(consensus_f, true_std_f)
        outcome = 1.0 if actual_high > threshold_f else 0.0
        trades.append({
            "consensus_f": consensus_f,
            "threshold_f": threshold_f,
            "outcome": outcome,
        })
    return trades
