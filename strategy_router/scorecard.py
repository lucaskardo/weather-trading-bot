"""
Phase 2.1 — Strategy Scorecard.

Scores each strategy 0–100 based on recent performance:

  score = w_sharpe       × norm_sharpe
        + w_calibration  × (1 - avg_brier)
        + w_exec         × norm_exec_quality
        - w_dd           × dd_penalty
        - w_instability  × edge_volatility_penalty

Requires >= MIN_TRADES_FOR_SCORE resolved trades; returns None otherwise.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from shared.params import Params, PARAMS

MIN_TRADES_FOR_SCORE = 10   # spec default; production should be 50


def score_strategy(
    strategy_name: str,
    conn: sqlite3.Connection,
    params: Params = PARAMS,
    lookback_days: int = 30,
    min_trades: int = MIN_TRADES_FOR_SCORE,
) -> Optional[float]:
    """
    Compute a 0–100 score for *strategy_name* using its last *lookback_days* days
    of resolved predictions from the DB.

    Returns None if fewer than *min_trades* resolved trades exist.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute(
        """SELECT fair_value, outcome, brier_score, executable_edge, realized_pnl
           FROM predictions
           WHERE strategy_name = ?
             AND is_shadow = 0
             AND outcome IS NOT NULL
             AND created_at >= ?
           ORDER BY created_at""",
        (strategy_name, cutoff),
    ).fetchall()

    if len(rows) < min_trades:
        return None

    trades = [dict(r) for r in rows]
    return _compute_score(trades, params)


def score_all_strategies(
    strategy_names: list[str],
    conn: sqlite3.Connection,
    params: Params = PARAMS,
    lookback_days: int = 30,
    min_trades: int = MIN_TRADES_FOR_SCORE,
) -> dict[str, Optional[float]]:
    """Score all strategies; returns map of name → score (None if insufficient data)."""
    return {
        name: score_strategy(name, conn, params, lookback_days, min_trades)
        for name in strategy_names
    }


def compute_score_from_trades(
    trades: list[dict[str, Any]],
    params: Params = PARAMS,
    min_trades: int = MIN_TRADES_FOR_SCORE,
) -> Optional[float]:
    """
    Compute score directly from a list of trade dicts (without DB).
    Used for testing and backtesting.
    Each dict must have: fair_value, outcome, brier_score, executable_edge, realized_pnl.
    """
    if len(trades) < min_trades:
        return None
    return _compute_score(trades, params)


# --------------------------------------------------------------------------- #
# Internal scoring logic
# --------------------------------------------------------------------------- #

def _compute_score(trades: list[dict[str, Any]], params: Params) -> float:
    pnls = [t.get("realized_pnl", 0.0) or 0.0 for t in trades]
    briers = [t.get("brier_score") for t in trades if t.get("brier_score") is not None]
    edges = [t.get("executable_edge", 0.0) or 0.0 for t in trades]

    # --- Sharpe component ---
    sharpe = _sharpe(pnls)
    norm_sharpe = _sigmoid_normalize(sharpe, scale=2.0)  # 0→0.5, 2→~0.88

    # --- Calibration (Brier) component ---
    avg_brier = statistics.mean(briers) if briers else 0.25
    calibration = max(0.0, 1.0 - avg_brier)  # lower Brier → higher score

    # --- Execution quality component ---
    avg_edge = statistics.mean(edges) if edges else 0.0
    norm_exec = _sigmoid_normalize(avg_edge, scale=0.10)  # 10% avg edge → ~0.88

    # --- Drawdown penalty ---
    dd_penalty = _max_drawdown_penalty(pnls)

    # --- Instability penalty (edge volatility) ---
    edge_vol = statistics.stdev(edges) if len(edges) > 1 else 0.0
    instability = min(1.0, edge_vol / 0.20)  # 20% vol → full penalty

    raw = (
        params.router_w_sharpe       * norm_sharpe
        + params.router_w_calibration * calibration
        + params.router_w_exec        * norm_exec
        - params.router_w_dd          * dd_penalty
        - params.router_w_instability * instability
    )

    # Scale 0–1 → 0–100, clamp
    return max(0.0, min(100.0, raw * 100.0))


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls)
    return mean / std if std > 0 else 0.0


def _sigmoid_normalize(x: float, scale: float = 1.0) -> float:
    """Map x ∈ (-∞, ∞) → (0, 1) via sigmoid centred at 0, with given scale."""
    return 1.0 / (1.0 + math.exp(-x / scale))


def _max_drawdown_penalty(pnls: list[float]) -> float:
    """
    Compute max drawdown as fraction of peak cumulative PnL.
    Returns a 0–1 penalty (1 = catastrophic drawdown).
    """
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / max(peak, 1.0)
        if dd > max_dd:
            max_dd = dd
    return min(1.0, max_dd)
