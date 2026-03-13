"""
Weather Trading Bot — main entry point.

Runs one trading cycle (or loops on a schedule via autoloop).
The Brain orchestrates everything; main.py handles startup + data wiring.

Usage:
    python main.py --paper           # single paper cycle with real market data
    python main.py --paper --loop    # continuous paper trading
    python main.py --live --loop     # live trading (requires API keys)
    python main.py --calibrate       # run calibration cycle
    python main.py --dashboard       # start dashboard API server
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.params import PARAMS
from state.db import init_db, assert_db_integrity, migrate_from_json
from strategy_router.brain import Brain
from risk.reconciliation import reconcile_positions


def startup_checks(conn) -> None:
    assert_db_integrity(conn)
    _reconcile_on_startup(conn)
    _check_params_sanity(PARAMS)


def _reconcile_on_startup(conn) -> None:
    result = reconcile_positions(conn, exchange_positions=[], auto_correct=False)
    if result.critical_count > 0:
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


def _fetch_markets() -> list:
    """Fetch markets from Kalshi and Polymarket. Returns empty list on error."""
    markets = []
    try:
        from clients.kalshi_client import fetch_all_weather_markets as fetch_kalshi
        km = fetch_kalshi()
        markets.extend(km)
        print(f"[main] kalshi: {len(km)} markets fetched")
    except Exception as exc:
        print(f"[main] kalshi fetch error: {exc}", file=sys.stderr)

    try:
        from clients.polymarket_client import fetch_all_weather_markets as fetch_poly
        pm = fetch_poly()
        markets.extend(pm)
        print(f"[main] polymarket: {len(pm)} markets fetched")
    except Exception as exc:
        print(f"[main] polymarket fetch error: {exc}", file=sys.stderr)

    return markets


def _persist_markets(markets: list, conn) -> None:
    """Upsert fetched markets into the DB before signal generation."""
    now = datetime.now(timezone.utc).isoformat()
    for m in markets:
        conn.execute(
            """INSERT OR REPLACE INTO markets
               (id, ticker, city, target_date, market_type, low_f, high_f,
                exchange, status, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                m["id"], m["ticker"], m["city"], m["target_date"],
                m.get("market_type", "above"),
                m.get("low_f"), m.get("high_f"),
                m.get("exchange", "kalshi"),
                "open", now,
            ),
        )
    conn.commit()
    print(f"[main] persisted {len(markets)} market(s) to DB")


def _fetch_forecasts(markets: list, conn) -> list:
    """Fetch weather forecasts for all unique (city, target_date) pairs."""
    from clients.weather import fetch_and_store

    forecasts = []
    seen: set = set()
    for m in markets:
        key = (m.get("city"), m.get("target_date"))
        if None in key or key in seen:
            continue
        seen.add(key)
        try:
            city_forecasts = fetch_and_store(m["city"], m["target_date"], conn)
            forecasts.extend(city_forecasts)
        except Exception as exc:
            print(f"[main] forecast error for {key}: {exc}", file=sys.stderr)

    print(f"[main] {len(forecasts)} forecast(s) for {len(seen)} city/date pair(s)")
    return forecasts


def run_once(paper: bool = True, live: bool = False) -> dict:
    """
    Run a single trading cycle.

    paper=True  → fetch real market data, execute paper trades (log to DB only)
    live=True   → fetch real data AND place real exchange orders
    """
    dry_run = not live

    conn = init_db()
    migrate_from_json()
    startup_checks(conn)

    from execution.exchange_executor import PaperExecutor, KalshiExecutor
    if live:
        try:
            executor = KalshiExecutor()
        except ValueError as exc:
            print(f"[main] Cannot start live mode: {exc}", file=sys.stderr)
            raise
    else:
        executor = PaperExecutor()

    brain = Brain(conn=conn, params=PARAMS, dry_run=dry_run, executor=executor)

    if paper or live:
        markets = _fetch_markets()
        _persist_markets(markets, conn)
        forecasts = _fetch_forecasts(markets, conn)
    else:
        markets = []
        forecasts = []

    open_positions = _load_open_positions(conn)

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


def _post_last_cycle(summary: dict, port: int = 5001) -> None:
    """Push last-cycle summary to dashboard API (best-effort)."""
    try:
        import urllib.request, json as _json
        data = _json.dumps(summary).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/api/last_cycle",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass  # dashboard may not be running


def autoloop(interval_seconds: int = 300, paper: bool = True, live: bool = False) -> None:
    mode = "live" if live else "paper"
    print(f"[main] Starting autoloop (interval={interval_seconds}s, mode={mode})")
    while True:
        try:
            summary = run_once(paper=paper, live=live)
            print(
                f"[main] cycle complete — signals={summary['signals_generated']} "
                f"executable={summary['signals_executable']} "
                f"executed={summary['executed']} exits={summary['exits']}"
            )
            _post_last_cycle(summary)
        except Exception as exc:
            print(f"[main] cycle error: {exc}", file=sys.stderr)
        time.sleep(interval_seconds)


def run_calibrate() -> None:
    """Run the calibration pipeline."""
    from research.calibrator import run_calibration
    conn = init_db()
    result = run_calibration(conn, PARAMS)
    print(f"[calibrate] done: {result}")


def run_autoresearch() -> None:
    """Propose and run one autoresearch experiment cycle."""
    from research.autoresearch import ExperimentRegistry
    from research.calibrator import _build_trade_log
    conn = init_db()
    trade_log = _build_trade_log(conn)
    if len(trade_log) < 6:
        print(f"[autoresearch] Not enough resolved trades ({len(trade_log)}) to run experiments.")
        return
    registry = ExperimentRegistry(conn, PARAMS)
    result = registry.run_cycle(trade_log)
    print(
        f"[autoresearch] experiment={result['experiment_id']} "
        f"baseline={result['baseline_brier']:.4f} "
        f"candidate={result['candidate_brier']:.4f} "
        f"improvement={result['improvement_pct']:.1f}% "
        f"promoted={result['promoted']}"
    )


def run_dashboard(port: int = 5001) -> None:
    """Start the dashboard API server."""
    try:
        from dashboard.api import run
        run(port=port)
    except ImportError:
        print("[dashboard] flask not installed. Run: pip install flask")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Trading Bot")
    parser.add_argument("--paper", action="store_true",
                        help="Paper trade with real market data (default mode)")
    parser.add_argument("--live", action="store_true",
                        help="Execute real trades (requires API keys)")
    parser.add_argument("--loop", action="store_true",
                        help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=300,
                        help="Loop interval in seconds (default: 300)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration cycle and exit")
    parser.add_argument("--dashboard", action="store_true",
                        help="Start dashboard API server")
    parser.add_argument("--port", type=int, default=5001,
                        help="Dashboard port (default: 5001)")
    parser.add_argument("--autoresearch", action="store_true",
                        help="Run one autoresearch experiment cycle")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard(port=args.port)
    elif args.autoresearch:
        run_autoresearch()
    elif args.calibrate:
        run_calibrate()
    elif args.loop:
        autoloop(
            interval_seconds=args.interval,
            paper=args.paper or not args.live,
            live=args.live,
        )
    else:
        result = run_once(paper=args.paper or not args.live, live=args.live)
        print(result)
