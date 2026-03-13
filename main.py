"""
Weather Trading Bot — main entry point.

Runs one trading cycle (or loops on a schedule via autoloop).
The Brain orchestrates everything; main.py handles startup checks + scheduling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from shared.params import PARAMS
from state.db import init_db, assert_db_integrity, migrate_from_json
from strategy_router.brain import Brain
from risk.reconciliation import reconcile_positions


def startup_checks(conn) -> None:
    """
    Full startup validation before entering the main loop.

    Spec requirements:
      init_db()              — tables exist (called before startup_checks)
      assert_db_integrity()  — PRAGMA integrity_check passes
      reconcile_positions()  — sync local DB with exchange (uses empty list = no-op)
      check_params_sanity()  — params are within safe bounds
    """
    assert_db_integrity(conn)
    _reconcile_on_startup(conn)
    _check_params_sanity(PARAMS)


def _reconcile_on_startup(conn) -> None:
    """
    Run reconciliation on startup with empty exchange list.
    In production this would call the exchange API; here it's a no-op
    that validates the local DB is internally consistent.
    """
    result = reconcile_positions(conn, exchange_positions=[], auto_correct=False)
    if result.critical_count > 0:
        import sys
        print(
            f"[startup] WARNING: {result.critical_count} critical reconciliation "
            f"discrepancies found. Review positions before trading.",
            file=sys.stderr,
        )


def _check_params_sanity(params) -> None:
    assert 1 <= params.base_std_f <= 20, f"base_std_f out of range: {params.base_std_f}"
    assert 0 < params.taker_fee_pct < 0.10, f"taker_fee_pct looks wrong: {params.taker_fee_pct}"
    assert 0 < params.min_executable_edge < 0.50, f"min_executable_edge suspicious: {params.min_executable_edge}"
    weights = (
        params.router_w_sharpe
        + params.router_w_calibration
        + params.router_w_exec
        + params.router_w_dd
        + params.router_w_instability
    )
    assert abs(weights - 1.0) < 1e-6, f"Router weights don't sum to 1: {weights}"


def run_once(dry_run: bool = True) -> dict:
    """Run a single trading cycle. Returns brain cycle summary."""
    conn = init_db()
    migrate_from_json()   # no-op if state_legacy/ doesn't exist
    startup_checks(conn)

    brain = Brain(conn=conn, params=PARAMS, dry_run=dry_run)

    # In production these come from exchange APIs + weather client
    markets: list = []
    forecasts: list = []
    open_positions: list = _load_open_positions(conn)

    summary = brain.run_cycle(
        markets=markets,
        forecasts=forecasts,
        open_positions=open_positions,
    )
    return summary


def _load_open_positions(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM positions
           WHERE status NOT IN ('WON','LOST','EXITED_CONVERGENCE',
                                'EXITED_STOP','EXITED_PRE_SETTLEMENT')"""
    ).fetchall()
    return [dict(r) for r in rows]


def autoloop(interval_seconds: int = 300, dry_run: bool = True) -> None:
    """Run trading cycles on a fixed interval (default every 5 minutes)."""
    print(f"[main] Starting autoloop (interval={interval_seconds}s, dry_run={dry_run})")
    while True:
        try:
            summary = run_once(dry_run=dry_run)
            print(
                f"[main] cycle complete — signals={summary['signals_generated']} "
                f"exec={summary['executed']} exits={summary['exits']}"
            )
        except Exception as exc:
            print(f"[main] cycle error: {exc}", file=sys.stderr)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Trading Bot")
    parser.add_argument("--live", action="store_true", help="Execute real trades (default: dry-run)")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds")
    args = parser.parse_args()

    if args.loop:
        autoloop(interval_seconds=args.interval, dry_run=not args.live)
    else:
        result = run_once(dry_run=not args.live)
        print(result)
