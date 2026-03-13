"""
Phase 5.1 — Shadow Strategy Logging.

Persists shadow signals to the predictions table with is_shadow=1.
At settlement, computes hypothetical PnL so shadow strategies can be scored
using the same metrics as live strategies (Sharpe, Brier, win_rate).

Also handles settlement of shadow predictions when NWS data arrives.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from strategies.base import Signal


def log_shadow_signal(
    conn: sqlite3.Connection,
    signal: Signal,
    market_id: Optional[str] = None,
) -> int:
    """
    Persist a shadow signal to the predictions table.

    Args:
        conn:      SQLite connection.
        signal:    Signal with is_shadow=True.
        market_id: Optional DB market id (uses signal.market_id if omitted).

    Returns:
        Row id of the inserted prediction.
    """
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO predictions
           (strategy_name, market_id, ticker, city, target_date,
            fair_value, market_price, executable_price,
            edge, executable_edge, confidence,
            consensus_f, agreement, n_models,
            is_shadow, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            signal.strategy_name,
            market_id or signal.market_id,
            signal.ticker,
            signal.city,
            signal.target_date,
            signal.fair_value,
            signal.market_price,
            signal.executable_price,
            signal.edge,
            signal.executable_edge,
            signal.confidence,
            signal.consensus_f,
            signal.agreement,
            signal.n_models,
            1,   # is_shadow
            signal.created_at or now,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def log_shadow_signals_batch(
    conn: sqlite3.Connection,
    signals: list[Signal],
) -> list[int]:
    """Log a list of shadow signals. Returns list of inserted row ids."""
    return [log_shadow_signal(conn, s) for s in signals if s.is_shadow]


def settle_shadow_predictions(
    conn: sqlite3.Connection,
    city: str,
    target_date: str,
    actual_high_f: float,
    station: str,
) -> int:
    """
    Resolve all shadow predictions for city/date using official settlement data.

    Computes hypothetical PnL: if prediction said YES and market resolved YES,
    PnL = (1 - entry_price) * notional; otherwise -entry_price * notional.
    Uses a synthetic notional of $100 per shadow trade for comparability.

    Args:
        conn:          SQLite connection.
        city:          City being settled.
        target_date:   Contract date.
        actual_high_f: Official NWS high temperature.
        station:       ICAO station used for settlement.

    Returns:
        Number of shadow predictions settled.
    """
    rows = conn.execute(
        """SELECT p.id, p.fair_value, p.market_price, p.executable_edge,
                  m.high_f
           FROM predictions p
           LEFT JOIN markets m ON m.id = p.market_id
           WHERE p.city = ?
             AND p.target_date = ?
             AND p.is_shadow = 1
             AND p.outcome IS NULL""",
        (city, target_date),
    ).fetchall()

    settled = 0
    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        high_f = row["high_f"]
        if high_f is None:
            continue  # can't settle without threshold

        outcome = 1.0 if actual_high_f > high_f else 0.0
        fair_value = row["fair_value"] or 0.5
        brier = (fair_value - outcome) ** 2

        # Hypothetical PnL: assumes we bet YES at market_price with $100 notional
        market_price = row["market_price"] or 0.5
        if outcome == 1.0:
            hypo_pnl = (1.0 - market_price) * 100.0
        else:
            hypo_pnl = -market_price * 100.0

        conn.execute(
            """UPDATE predictions
               SET outcome=?, actual_high_f=?, brier_score=?,
                   realized_pnl=?, resolved_at=?
               WHERE id=?""",
            (outcome, actual_high_f, brier, hypo_pnl, now, row["id"]),
        )
        settled += 1

    if settled:
        conn.commit()

    return settled


def get_shadow_performance(
    conn: sqlite3.Connection,
    strategy_name: str,
    min_trades: int = 10,
) -> Optional[dict[str, Any]]:
    """
    Return performance summary for a shadow strategy.
    Returns None if fewer than min_trades are resolved.
    """
    rows = conn.execute(
        """SELECT fair_value, outcome, brier_score, realized_pnl, executable_edge
           FROM predictions
           WHERE strategy_name = ?
             AND is_shadow = 1
             AND outcome IS NOT NULL
           ORDER BY created_at""",
        (strategy_name,),
    ).fetchall()

    if len(rows) < min_trades:
        return None

    pnls = [r["realized_pnl"] or 0.0 for r in rows]
    briers = [r["brier_score"] for r in rows if r["brier_score"] is not None]
    wins = sum(1 for p in pnls if p > 0)

    import statistics
    mean_pnl = statistics.mean(pnls)
    std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1e-9
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    return {
        "strategy_name": strategy_name,
        "trade_count": len(rows),
        "win_rate": wins / len(rows),
        "avg_brier": statistics.mean(briers) if briers else 0.25,
        "sharpe": sharpe,
        "total_hypo_pnl": sum(pnls),
    }
