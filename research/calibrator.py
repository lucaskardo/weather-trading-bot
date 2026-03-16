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
from core.forecaster import (
    brier_decomposition, compute_fair_value, lead_bucket_from_hours, regime_bucket,
)
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
        segment_metrics: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.n_trades = n_trades
        self.brier = brier
        self.brier_index = brier_index
        self.old_params = old_params
        self.new_params = new_params
        self.optimizer_result = optimizer_result
        self.resolved_count = resolved_count
        self.segment_metrics = segment_metrics or {}

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
            segment_metrics={},
        )

    # Step 3: Compute Brier decomposition on current params
    # Use stored fair_value directly — it's the probability that was predicted at trade time.
    # Recalculate only for trades that have both consensus_f and threshold_f.
    predictions = []
    outcomes = []
    for t in trade_log:
        if t.get("outcome") is None:
            continue
        if t.get("consensus_f") is not None:
            market_type = t.get("market_type") or "above"
            high_f = t.get("high_f") or t.get("threshold_f")
            low_f = t.get("low_f")
            # Use the same math the live strategy uses: dynamic_std + temp_scaling
            # No conn → no bias correction (we're testing parameter sensitivity, not bias)
            from shared.types import ModelForecast
            city = t.get("city") or "SYN"
            target_date = t.get("target_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ts = datetime.now(timezone.utc).isoformat()
            fake_forecast = ModelForecast(
                city=city,
                target_date=target_date,
                model_name="consensus",
                predicted_high_f=float(t["consensus_f"]),
                run_id="calibration",
                publish_time=ts,
                source_url="calibration://recompute",
                fetched_at=ts,
            )
            fv, _, _, _ = compute_fair_value(
                [fake_forecast],
                city, target_date,
                market_type,
                float(high_f) if high_f is not None else None,
                float(low_f) if low_f is not None else None,
                params,
                conn=conn, use_mc=False,
            )
            pred = fv if fv is not None else (t.get("fair_value", 0.5) or 0.5)
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
    segment_metrics = _compute_segment_metrics(trade_log, predictions, outcomes)

    # Step 4: Walk-forward optimization
    opt_result = optimize_params(
        trade_log=trade_log,
        n_windows=n_windows,
        current_params=old_params,
    )
    _log(f"[calibrator] optimizer: {opt_result}")

    new_params = opt_result.best_params

    # Step 5: Persist (never mutate the passed params — caller decides what to apply)
    if save:
        _save_to_db(conn, decomp, new_params)
        _save_segment_metrics(conn, segment_metrics)
        save_params(new_params, params_path)

    _log(
        f"[calibrator] optimised params: base_std_f={new_params.get('base_std_f', params.base_std_f):.2f} "
        f"temp_T={new_params.get('temp_T', params.temp_scaling_T):.3f}"
    )

    return CalibrationResult(
        n_trades=n_trades,
        brier=decomp["brier"],
        brier_index=decomp["brier_index"],
        old_params=old_params,
        new_params=new_params,
        optimizer_result=opt_result,
        resolved_count=newly_resolved,
        segment_metrics=segment_metrics,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _trade_lead_hours(target_date: str | None) -> float:
    if not target_date:
        return 48.0
    try:
        eod = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
        return max(0.0, (eod - datetime.now(timezone.utc)).total_seconds() / 3600)
    except Exception:
        return 48.0


def _compute_segment_metrics(trade_log: list[dict[str, Any]], predictions: list[float], outcomes: list[float]) -> dict[str, list[dict[str, Any]]]:
    segment_buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for trade, pred, outcome in zip(trade_log, predictions, outcomes):
        mtype = (trade.get("market_type") or "above").lower()
        lead_seg = lead_bucket_from_hours(_trade_lead_hours(trade.get("target_date")))
        regime_seg = regime_bucket(float(trade.get("agreement") or 0.0), mtype)
        for kind, value in (("market_type", mtype), ("lead_bucket", lead_seg), ("regime", regime_seg), ("city", trade.get("city") or "UNK")):
            segment_buckets.setdefault((kind, value), []).append((pred, outcome))

    out: dict[str, list[dict[str, Any]]] = {"market_type": [], "lead_bucket": [], "regime": [], "city": []}
    for (kind, value), vals in segment_buckets.items():
        preds = [p for p, _ in vals]
        outs = [o for _, o in vals]
        decomp = brier_decomposition(preds, outs)
        out[kind].append({
            "segment": value,
            "trade_count": len(vals),
            "avg_brier": decomp["brier"],
            "avg_prediction": sum(preds) / len(preds),
            "avg_outcome": sum(outs) / len(outs),
        })
    for kind in out:
        out[kind].sort(key=lambda r: (-r["trade_count"], r["segment"]))
    return out


def _save_segment_metrics(conn: sqlite3.Connection, segment_metrics: dict[str, list[dict[str, Any]]]) -> None:
    conn.execute("DELETE FROM calibration_segments")
    for kind, rows in segment_metrics.items():
        for row in rows:
            conn.execute(
                """INSERT INTO calibration_segments
                   (segment_kind, segment_value, trade_count, avg_brier, avg_outcome, avg_prediction)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kind, row["segment"], row["trade_count"], row["avg_brier"], row["avg_outcome"], row["avg_prediction"]),
            )

def _save_calibration_profiles(conn: sqlite3.Connection, segment_metrics: dict[str, list[dict[str, Any]]]) -> None:
    """Persist lightweight per-segment calibration adjustments.

    We intentionally use a simple empirical adjustment (avg_outcome - avg_prediction)
    instead of fitting a fragile per-segment model on small samples.
    """
    conn.execute("DELETE FROM calibration_profiles")
    for kind, rows in segment_metrics.items():
        for row in rows:
            trade_count = int(row.get("trade_count") or 0)
            if trade_count < 5:
                continue
            avg_outcome = float(row.get("avg_outcome") or 0.0)
            avg_prediction = float(row.get("avg_prediction") or 0.0)
            prob_adjustment = max(-0.15, min(0.15, avg_outcome - avg_prediction))
            conn.execute(
                """INSERT INTO calibration_profiles
                   (segment_kind, segment_value, trade_count, avg_brier, avg_outcome, avg_prediction, prob_adjustment)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (kind, row["segment"], trade_count, row.get("avg_brier"), avg_outcome, avg_prediction, prob_adjustment),
            )


def _load_resolved_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT p.id, p.strategy_name, p.city, p.target_date, p.fair_value,
                  p.consensus_f, p.market_price, p.outcome, p.brier_score,
                  p.actual_high_f, p.created_at,
                  m.market_type, m.high_f, m.low_f
           FROM predictions p
           LEFT JOIN markets m ON p.market_id = m.id
           WHERE p.outcome IS NOT NULL
             AND p.is_shadow = 0
           ORDER BY p.created_at"""
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
            "threshold_f": r.get("threshold_f"),   # may be None (legacy)
            "market_type": r.get("market_type") or "above",
            "high_f": r.get("high_f"),
            "low_f": r.get("low_f"),
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
