"""
Phase 3.4 — Refactored Calibrator.

Workflow:
  1. Query resolved predictions from SQLite (not JSONL)
  2. Resolve outcomes using NWS settlement (not band midpoints)
  3. Compute Brier score + decomposition
  4. Run walk-forward optimizer (differential evolution)
  5. Apply temperature scaling to all future predictions
  6. Save updated params to JSON
  7. Log Brier Index = (1 - sqrt(Brier)) × 100

Replaces: core/calibrator.py (deleted band midpoint _infer_temp_kalshi logic)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clients.nws_settlement import fetch_settlement, SettlementError
from core.forecaster import brier_decomposition, prob_above_threshold
from research.walk_forward import walk_forward_brier
from research.optimizer import optimize_params, save_params, OptimizerResult
from shared.params import Params, PARAMS


class CalibrationResult:
    """Summary of one calibration run."""

    def __init__(
        self,
        n_trades: int,
        brier: float,
        brier_index: float,
        old_params: dict[str, float],
        new_params: dict[str, float],
        optimizer_result: OptimizerResult | None,
        resolved_count: int,
    ):
        self.n_trades = n_trades
        self.brier = brier
        self.brier_index = brier_index
        self.old_params = old_params
        self.new_params = new_params
        self.optimizer_result = optimizer_result
        self.resolved_count = resolved_count

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(n={self.n_trades}, brier={self.brier:.4f}, "
            f"index={self.brier_index:.1f}, new_params={self.new_params})"
        )


def run_calibration(
    conn: sqlite3.Connection,
    params: Params = PARAMS,
    n_windows: int = 5,
    min_trades: int = 20,
    save: bool = True,
    params_path: Path | None = None,
) -> CalibrationResult:
    """
    Run a full calibration cycle.

    Args:
        conn:        SQLite connection (must have predictions + settlement_cache).
        params:      Current Params (will be updated in-place if save=True).
        n_windows:   Walk-forward folds.
        min_trades:  Minimum resolved trades before optimizing.
        save:        If True, write optimised params to JSON.
        params_path: Override path for JSON output.

    Returns:
        CalibrationResult with diagnostics.
    """
    # Step 1: Fetch resolved predictions from DB
    rows = _load_resolved_predictions(conn)
    _log(f"[calibrator] {len(rows)} predictions with outcomes in DB")

    # Step 2: Resolve any remaining outcomes via NWS settlement
    newly_resolved = _resolve_outstanding(conn, rows)
    _log(f"[calibrator] resolved {newly_resolved} new outcomes via NWS")

    # Reload after resolution
    rows = _load_resolved_predictions(conn)
    trade_log = _build_trade_log(rows)

    n_trades = len(trade_log)
    old_params = {"base_std_f": params.base_std_f, "temp_T": params.temp_scaling_T}

    if n_trades < min_trades:
        _log(f"[calibrator] only {n_trades} trades — need {min_trades} to optimize")
        decomp = brier_decomposition(
            [t["fair_value"] for t in trade_log],
            [t["outcome"] for t in trade_log],
        ) if trade_log else {"brier": 0.25, "brier_index": 0.0}
        return CalibrationResult(
            n_trades=n_trades,
            brier=decomp["brier"],
            brier_index=decomp.get("brier_index", 0.0),
            old_params=old_params,
            new_params=old_params,
            optimizer_result=None,
            resolved_count=newly_resolved,
        )

    # Step 3: Compute Brier decomposition on current params
    # Use stored fair_value directly — it's the probability that was predicted at trade time.
    # Recalculate only for trades that have both consensus_f and threshold_f.
    predictions = []
    outcomes = []
    for t in trade_log:
        if t.get("outcome") is None:
            continue
        if t.get("consensus_f") is not None and t.get("threshold_f") is not None:
            pred = prob_above_threshold(
                t["consensus_f"], t["threshold_f"],
                params.base_std_f, params.temp_scaling_T,
            )
        else:
            # Fallback: use stored fair_value (the prediction made at entry time)
            pred = t.get("fair_value", 0.5) or 0.5
        predictions.append(pred)
        outcomes.append(float(t["outcome"]))
    decomp = brier_decomposition(predictions, outcomes)
    _log(
        f"[calibrator] current Brier={decomp['brier']:.4f} "
        f"index={decomp['brier_index']:.1f} "
        f"(reliability={decomp['reliability']:.4f}, resolution={decomp['resolution']:.4f})"
    )

    # Step 4: Walk-forward optimization
    opt_result = optimize_params(
        trade_log=trade_log,
        n_windows=n_windows,
        current_params=old_params,
    )
    _log(f"[calibrator] optimizer: {opt_result}")

    new_params = opt_result.best_params

    # Step 5: Apply temperature scaling (update params in-place)
    params.base_std_f = new_params.get("base_std_f", params.base_std_f)
    params.temp_scaling_T = new_params.get("temp_T", params.temp_scaling_T)

    # Step 6: Persist
    if save:
        _save_to_db(conn, decomp, new_params)
        save_params(new_params, params_path)

    _log(
        f"[calibrator] updated params: base_std_f={params.base_std_f:.2f} "
        f"temp_T={params.temp_scaling_T:.3f}"
    )

    return CalibrationResult(
        n_trades=n_trades,
        brier=decomp["brier"],
        brier_index=decomp["brier_index"],
        old_params=old_params,
        new_params=new_params,
        optimizer_result=opt_result,
        resolved_count=newly_resolved,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _load_resolved_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, strategy_name, city, target_date, fair_value,
                  consensus_f, market_price, outcome, brier_score,
                  actual_high_f, created_at
           FROM predictions
           WHERE outcome IS NOT NULL
             AND is_shadow = 0
           ORDER BY created_at"""
    ).fetchall()
    return [dict(r) for r in rows]


def _resolve_outstanding(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """
    Find predictions without outcomes and try to resolve them via NWS.
    Updates the DB and returns count of newly resolved predictions.
    """
    unresolved = conn.execute(
        """SELECT id, city, target_date, fair_value, market_price
           FROM predictions
           WHERE outcome IS NULL AND is_shadow = 0"""
    ).fetchall()

    resolved = 0
    for row in unresolved:
        pred_id = row["id"]
        city = row["city"]
        target_date = row["target_date"]

        try:
            settlement = fetch_settlement(city, target_date)
        except (SettlementError, KeyError):
            continue

        actual_high_f = settlement["actual_high_f"]

        # Find the market's threshold (high_f) from the markets table
        market_row = conn.execute(
            "SELECT high_f FROM markets WHERE id=(SELECT market_id FROM predictions WHERE id=?)",
            (pred_id,)
        ).fetchone()
        threshold_f = market_row["high_f"] if market_row else None

        if threshold_f is None:
            continue

        outcome = 1.0 if actual_high_f > threshold_f else 0.0
        fair_value = row["fair_value"] or 0.5
        bs = (fair_value - outcome) ** 2

        conn.execute(
            """UPDATE predictions
               SET outcome=?, actual_high_f=?, brier_score=?, resolved_at=?
               WHERE id=?""",
            (outcome, actual_high_f, bs, datetime.now(timezone.utc).isoformat(), pred_id),
        )
        resolved += 1

    if resolved:
        conn.commit()

    return resolved


def _build_trade_log(rows: list[dict]) -> list[dict[str, Any]]:
    """
    Convert DB rows into trade_log format expected by walk_forward.

    walk_forward_brier requires consensus_f + threshold_f to recalculate
    predictions.  When those aren't stored, we set threshold_f = a sentinel
    so the fallback path in walk_forward uses fair_value directly.
    """
    result = []
    for r in rows:
        if r.get("outcome") is None:
            continue
        result.append({
            "consensus_f": r.get("consensus_f"),
            "threshold_f": r.get("threshold_f"),   # may be None
            "fair_value": r.get("fair_value", 0.5) or 0.5,
            "outcome": float(r["outcome"]),
            "city": r.get("city"),
            "target_date": r.get("target_date"),
        })
    return result


def _save_to_db(
    conn: sqlite3.Connection,
    decomp: dict[str, float],
    new_params: dict[str, float],
) -> None:
    """Record calibration run in strategy_metrics table."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO strategy_metrics
           (strategy_name, computed_at, avg_brier, score)
           VALUES (?, ?, ?, ?)""",
        ("_calibrator", now, decomp.get("brier", 0.25), decomp.get("brier_index", 0.0)),
    )
    conn.commit()


def _log(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr)
