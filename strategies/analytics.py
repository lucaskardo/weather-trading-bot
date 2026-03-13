"""
Phase 5.3 — Per-Strategy Analytics.

Records rich per-trade analytics for both live and shadow positions:
  hold_time_hours           — how long the position was open
  max_favorable_excursion   — best price reached while open (MFE)
  max_adverse_excursion     — worst price reached while open (MAE)
  cluster                   — geographic cluster
  edge_at_entry             — executable_edge when position was opened
  edge_at_exit              — executable_edge when position was closed
  forecast_run_ids          — comma-separated run IDs of forecasts used

Also provides a per-strategy breakdown query for reporting.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from shared.params import get_cluster


def record_position_analytics(
    conn: sqlite3.Connection,
    position_id: int,
    analytics: dict[str, Any],
) -> None:
    """
    Update a position row with rich analytics fields.

    analytics dict may contain:
      hold_time_hours, max_favorable_excursion, max_adverse_excursion,
      realized_pnl, exit_reason, exit_price, closed_at
    """
    allowed_fields = {
        "hold_time_hours", "max_favorable_excursion", "max_adverse_excursion",
        "realized_pnl", "exit_reason", "exit_price", "closed_at", "status",
    }
    updates = {k: v for k, v in analytics.items() if k in allowed_fields}
    if not updates:
        return

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [position_id]
    conn.execute(
        f"UPDATE positions SET {set_clause} WHERE id=?",
        values,
    )
    conn.commit()


def compute_position_analytics(
    position: dict[str, Any],
    price_history: list[float],
    exit_price: float,
    closed_at: str,
    edge_at_exit: Optional[float] = None,
    forecast_run_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute analytics for a closed position from its price history.

    Args:
        position:         Position dict (must have entry_price, side, opened_at,
                          size_usd, city, executable_edge).
        price_history:    All market prices observed while position was open.
        exit_price:       Final exit price.
        closed_at:        ISO datetime when position was closed.
        edge_at_exit:     Executable edge at time of exit.
        forecast_run_ids: List of model run IDs used while holding.

    Returns:
        Analytics dict ready for record_position_analytics().
    """
    side = position.get("side", "YES")
    entry_price = position.get("entry_price", 0.5)
    opened_at_str = position.get("opened_at", "")
    city = position.get("city", "")

    # Hold time
    hold_hours = _compute_hold_time(opened_at_str, closed_at)

    # MFE / MAE
    all_prices = price_history + [exit_price]
    if side == "YES":
        mfe = max(all_prices) - entry_price if all_prices else 0.0
        mae = entry_price - min(all_prices) if all_prices else 0.0
    else:
        mfe = entry_price - min(all_prices) if all_prices else 0.0
        mae = max(all_prices) - entry_price if all_prices else 0.0

    # Realized PnL
    size = position.get("size_usd", 0.0)
    if side == "YES":
        realized_pnl = (exit_price - entry_price) * size
    else:
        realized_pnl = (entry_price - exit_price) * size

    run_ids_str = ",".join(forecast_run_ids) if forecast_run_ids else ""

    return {
        "hold_time_hours": max(0.0, hold_hours),
        "max_favorable_excursion": max(0.0, mfe),
        "max_adverse_excursion": max(0.0, mae),
        "realized_pnl": realized_pnl,
        "exit_price": exit_price,
        "closed_at": closed_at,
        "status": _exit_status(realized_pnl),
        # Stored as JSON string in exit_reason for rich metadata
        "exit_reason": json.dumps({
            "edge_at_entry": position.get("executable_edge"),
            "edge_at_exit": edge_at_exit,
            "cluster": get_cluster(city),
            "forecast_run_ids": run_ids_str,
        }),
    }


def get_strategy_analytics(
    conn: sqlite3.Connection,
    strategy_name: str,
) -> dict[str, Any]:
    """
    Return aggregated analytics for a strategy from closed positions.
    """
    rows = conn.execute(
        """SELECT hold_time_hours, max_favorable_excursion,
                  max_adverse_excursion, realized_pnl, city
           FROM positions
           WHERE strategy_name=?
             AND status IN ('WON','LOST','EXITED_CONVERGENCE',
                            'EXITED_STOP','EXITED_PRE_SETTLEMENT')
             AND realized_pnl IS NOT NULL""",
        (strategy_name,),
    ).fetchall()

    if not rows:
        return {"strategy_name": strategy_name, "trade_count": 0}

    import statistics

    pnls = [r["realized_pnl"] for r in rows]
    hold_times = [r["hold_time_hours"] for r in rows if r["hold_time_hours"] is not None]
    mfes = [r["max_favorable_excursion"] for r in rows if r["max_favorable_excursion"] is not None]
    maes = [r["max_adverse_excursion"] for r in rows if r["max_adverse_excursion"] is not None]

    # Cluster breakdown
    cluster_counts: dict[str, int] = {}
    for r in rows:
        c = get_cluster(r["city"] or "")
        if c:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    return {
        "strategy_name": strategy_name,
        "trade_count": len(rows),
        "total_pnl": sum(pnls),
        "avg_pnl": statistics.mean(pnls),
        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
        "avg_hold_hours": statistics.mean(hold_times) if hold_times else None,
        "avg_mfe": statistics.mean(mfes) if mfes else None,
        "avg_mae": statistics.mean(maes) if maes else None,
        "mfe_to_mae_ratio": (
            statistics.mean(mfes) / statistics.mean(maes)
            if mfes and maes and statistics.mean(maes) > 0
            else None
        ),
        "cluster_breakdown": cluster_counts,
    }


def get_liquidity_bucket(size_usd: float) -> str:
    """Classify trade size into liquidity bucket."""
    if size_usd < 25:
        return "micro"
    if size_usd < 100:
        return "small"
    if size_usd < 500:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _compute_hold_time(opened_at: str, closed_at: str) -> float:
    """Return hold time in hours between two ISO timestamps."""
    try:
        t_open = datetime.fromisoformat(opened_at)
        t_close = datetime.fromisoformat(closed_at)
        if t_open.tzinfo is None:
            t_open = t_open.replace(tzinfo=timezone.utc)
        if t_close.tzinfo is None:
            t_close = t_close.replace(tzinfo=timezone.utc)
        return (t_close - t_open).total_seconds() / 3600
    except (ValueError, TypeError):
        return 0.0


def _exit_status(realized_pnl: float) -> str:
    return "WON" if realized_pnl > 0 else "LOST"
