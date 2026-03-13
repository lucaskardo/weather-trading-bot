"""
Phase 5.2 — Shadow Strategy Auto-Promotion Rules.

Flags shadow strategies as PROMOTION CANDIDATES when they meet the bar.
Does NOT auto-promote — human review is required.

Promotion criteria (all must be true):
  1. Strategy is not currently live (is_live=False)
  2. trade_count >= MIN_TRADES_FOR_PROMOTION  (50)
  3. scorecard score > PROMOTION_SCORE_THRESHOLD  (75)
  4. (Future) walk-forward Brier beats baseline on held-out data

Logs a clear "PROMOTION CANDIDATE" message and records in strategy_metrics.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from strategy_router.scorecard import score_strategy

MIN_TRADES_FOR_PROMOTION = 50
PROMOTION_SCORE_THRESHOLD = 75.0


class PromotionCandidate:
    """A shadow strategy that has earned promotion consideration."""

    def __init__(
        self,
        strategy_name: str,
        score: float,
        trade_count: int,
        avg_brier: float,
        win_rate: float,
        sharpe: float,
        detected_at: str,
    ):
        self.strategy_name = strategy_name
        self.score = score
        self.trade_count = trade_count
        self.avg_brier = avg_brier
        self.win_rate = win_rate
        self.sharpe = sharpe
        self.detected_at = detected_at

    def __repr__(self) -> str:
        return (
            f"PromotionCandidate(strategy={self.strategy_name!r}, "
            f"score={self.score:.1f}, trades={self.trade_count}, "
            f"brier={self.avg_brier:.3f})"
        )


def scan_for_candidates(
    conn: sqlite3.Connection,
    shadow_strategy_names: list[str],
    min_trades: int = MIN_TRADES_FOR_PROMOTION,
    score_threshold: float = PROMOTION_SCORE_THRESHOLD,
    lookback_days: int = 30,
) -> list[PromotionCandidate]:
    """
    Scan shadow strategies and return those that meet the promotion bar.

    Args:
        conn:                   SQLite connection.
        shadow_strategy_names:  Names of strategies with is_live=False.
        min_trades:             Minimum resolved shadow trades required.
        score_threshold:        Minimum scorecard score required.
        lookback_days:          Lookback window for scoring.

    Returns:
        List of PromotionCandidate objects (empty if none qualify).
    """
    candidates: list[PromotionCandidate] = []
    now = datetime.now(timezone.utc).isoformat()

    for name in shadow_strategy_names:
        trade_count = _count_resolved_shadow_trades(conn, name)
        if trade_count < min_trades:
            continue

        score = score_strategy(
            name, conn,
            lookback_days=lookback_days,
            min_trades=min_trades,
        )
        if score is None or score <= score_threshold:
            continue

        metrics = _get_shadow_metrics(conn, name)
        candidate = PromotionCandidate(
            strategy_name=name,
            score=score,
            trade_count=trade_count,
            avg_brier=metrics.get("avg_brier", 0.25),
            win_rate=metrics.get("win_rate", 0.0),
            sharpe=metrics.get("sharpe", 0.0),
            detected_at=now,
        )
        candidates.append(candidate)
        _log_candidate(candidate, conn)

    return candidates


def _count_resolved_shadow_trades(conn: sqlite3.Connection, strategy_name: str) -> int:
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM predictions
           WHERE strategy_name=? AND is_shadow=1 AND outcome IS NOT NULL""",
        (strategy_name,),
    ).fetchone()
    return row["n"] if row else 0


def _get_shadow_metrics(conn: sqlite3.Connection, strategy_name: str) -> dict[str, Any]:
    rows = conn.execute(
        """SELECT brier_score, realized_pnl
           FROM predictions
           WHERE strategy_name=? AND is_shadow=1 AND outcome IS NOT NULL""",
        (strategy_name,),
    ).fetchall()

    if not rows:
        return {}

    pnls = [r["realized_pnl"] or 0.0 for r in rows]
    briers = [r["brier_score"] for r in rows if r["brier_score"] is not None]
    wins = sum(1 for p in pnls if p > 0)

    import statistics
    mean_pnl = statistics.mean(pnls) if pnls else 0.0
    std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1e-9

    return {
        "avg_brier": statistics.mean(briers) if briers else 0.25,
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "sharpe": mean_pnl / std_pnl if std_pnl > 0 else 0.0,
    }


def _log_candidate(candidate: PromotionCandidate, conn: sqlite3.Connection) -> None:
    import sys
    print(
        f"[promotion] *** PROMOTION CANDIDATE: {candidate.strategy_name} ***\n"
        f"  score={candidate.score:.1f} (threshold={PROMOTION_SCORE_THRESHOLD})\n"
        f"  trades={candidate.trade_count} (min={MIN_TRADES_FOR_PROMOTION})\n"
        f"  brier={candidate.avg_brier:.3f}  win_rate={candidate.win_rate:.1%}  "
        f"sharpe={candidate.sharpe:.2f}\n"
        f"  → Human review required before promotion to live.",
        file=sys.stderr,
    )
    # Record in strategy_metrics for audit trail
    conn.execute(
        """INSERT INTO strategy_metrics
           (strategy_name, computed_at, trade_count, win_rate, sharpe,
            avg_brier, score, is_live)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            candidate.strategy_name,
            candidate.detected_at,
            candidate.trade_count,
            candidate.win_rate,
            candidate.sharpe,
            candidate.avg_brier,
            candidate.score,
            0,   # still shadow
        ),
    )
    conn.commit()
