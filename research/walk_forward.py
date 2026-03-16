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
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from core.forecaster import brier_score, compute_fair_value
from shared.params import PARAMS
from shared.types import ModelForecast


def _trade_predicted_prob(trade: dict[str, Any], params_candidate: dict[str, float]) -> float | None:
    """Recompute a trade probability using the canonical fair-value pipeline."""
    consensus_f = trade.get("consensus_f")
    high_f = trade.get("high_f") or trade.get("threshold_f")
    low_f = trade.get("low_f")
    market_type = trade.get("market_type") or "above"
    if consensus_f is None or high_f is None:
        return None

    city = trade.get("city") or "SYN"
    target_date = trade.get("target_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).isoformat()
    fake_forecast = ModelForecast(
        model_name="consensus",
        city=city,
        target_date=target_date,
        predicted_high_f=float(consensus_f),
        run_id="walkforward",
        publish_time=ts,
        source_url="walkforward://synthetic",
        fetched_at=ts,
    )
    params = replace(
        PARAMS,
        base_std_f=float(params_candidate.get("base_std_f", PARAMS.base_std_f)),
        temp_scaling_T=float(params_candidate.get("temp_T", params_candidate.get("temp_scaling_T", PARAMS.temp_scaling_T))),
        use_monte_carlo=False,
    )
    fv, _, _, _ = compute_fair_value(
        [fake_forecast],
        city,
        target_date,
        market_type,
        float(high_f),
        float(low_f) if low_f is not None else None,
        params,
        conn=None,
        use_mc=False,
    )
    return fv


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
            outcome = trade.get("outcome")
            if outcome is None:
                continue

            pred = _trade_predicted_prob(trade, params_candidate)
            if pred is None:
                continue
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
            outcome = trade.get("outcome")
            if outcome is None:
                continue
            pred = _trade_predicted_prob(trade, params_candidate)
            if pred is None:
                continue
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
