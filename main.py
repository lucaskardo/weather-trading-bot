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

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from shared.params import PARAMS
from state.db import init_db, assert_db_integrity, migrate_from_json
from strategy_router.brain import Brain
from risk.reconciliation import reconcile_positions


def _header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def startup_checks(conn) -> None:
    assert_db_integrity(conn)
    _reconcile_on_startup(conn)
    _load_promoted_params(conn)
    _check_params_sanity(PARAMS)


def _load_promoted_params(conn) -> None:
    """Apply any previously promoted experiment params from the DB."""
    try:
        from research.autoresearch import load_promoted_params
        load_promoted_params(conn, PARAMS)
    except Exception as exc:
        print(f"[startup] Could not load promoted params: {exc}", file=sys.stderr)


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


def _print_config() -> None:
    """Print active configuration at startup so user knows what's running."""
    env_exists   = Path(".env").exists()
    bankroll     = float(os.environ.get("BANKROLL", 1000.0))
    loss_limit   = float(os.environ.get("DAILY_LOSS_LIMIT", 50.0))
    interval     = int(os.environ.get("POLL_INTERVAL_S", 300))
    noaa_token   = "set" if os.environ.get("NOAA_CDO_TOKEN") else "MISSING (settlement uses IEM fallback)"
    kalshi_key   = "set" if os.environ.get("KALSHI_KEY_ID") else "not set (only needed for live trading)"

    _header("CONFIG")
    print(f"  .env file        {'found' if env_exists else 'NOT found — using defaults'}")
    print(f"  Bankroll         ${bankroll:.2f}")
    print(f"  Daily loss limit ${loss_limit:.2f}")
    print(f"  Poll interval    {interval}s")
    print(f"  NOAA CDO token   {noaa_token}")
    print(f"  Kalshi API key   {kalshi_key}")


def _settle_expired_positions(conn, open_positions: list[dict]) -> dict:
    """
    For each open position whose target_date has passed, fetch actual
    settlement temperature and return settlement results for brain.run_cycle().

    Returns: {position_id: {"won": bool, "actual_high_f": float}}
    """
    from clients.nws_settlement import fetch_settlement, SettlementError
    from datetime import date

    results = {}
    today = date.today()

    for pos in open_positions:
        td = pos.get("target_date", "")
        try:
            if date.fromisoformat(td) >= today:
                continue  # not yet settled
        except ValueError:
            continue

        city        = pos.get("city", "")
        pos_id      = pos.get("id")
        high_f      = pos.get("high_f")
        low_f       = pos.get("low_f")
        market_type = pos.get("market_type", "above")
        side        = pos.get("side", "YES")

        # Check settlement cache first
        cached = conn.execute(
            "SELECT actual_high_f FROM settlement_cache WHERE city=? AND target_date=?",
            (city, td)
        ).fetchone()

        if cached:
            actual_high = float(cached["actual_high_f"])
        else:
            try:
                result = fetch_settlement(city, td)
                actual_high = result["actual_high_f"]
                conn.execute(
                    """INSERT OR REPLACE INTO settlement_cache
                       (city, target_date, actual_high_f, station, fetched_at)
                       VALUES (?, ?, ?, 'NWS', ?)""",
                    (city, td, actual_high, datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
                print(f"[settle] {city}/{td}: actual high = {actual_high:.1f}°F")
            except (SettlementError, KeyError, Exception) as exc:
                print(f"[settle] {city}/{td} error: {exc}", file=sys.stderr)
                continue

        if market_type == "above":
            outcome = 1.0 if actual_high >= high_f else 0.0
        elif market_type == "below":
            outcome = 1.0 if actual_high < high_f else 0.0
        elif market_type == "band" and low_f is not None:
            outcome = 1.0 if (low_f <= actual_high < high_f) else 0.0
        else:
            outcome = 0.0

        won = (outcome == 1.0) if side == "YES" else (outcome == 0.0)
        results[pos_id] = {"won": won, "actual_high_f": actual_high}
        print(f"[settle] position {pos_id} {city}/{td} → {'WON' if won else 'LOST'} "
              f"(actual={actual_high:.1f}°F, threshold={high_f}°F, type={market_type})")

    return results


def run_diagnose() -> None:
    """Test all external APIs and DB connectivity."""
    _header("DIAGNOSE")

    print("\n[1] Kalshi — connectivity")
    try:
        import requests
        r = requests.get("https://trading-api.kalshi.com/trade-api/v2/exchange/status", timeout=5)
        print(f"  OK  status={r.status_code}")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[1b] Kalshi — live weather market fetch")
    try:
        from clients.kalshi_client import fetch_all_weather_markets as _k
        km = _k()
        if km:
            print(f"  {len(km)} future weather markets found")
            for m in km[:5]:
                print(f"     {m['ticker']:<38} {m['city']}  {m['target_date']}  price={m['market_price']:.2f}")
            if len(km) > 5:
                print(f"     ... and {len(km)-5} more")
        else:
            print("  0 markets — Kalshi may have no active weather series today")
            print("  The bot will try keyword search automatically during --paper")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[2] Open-Meteo — weather forecast")
    try:
        from clients.weather import fetch_and_store
        conn = init_db()
        from datetime import date, timedelta
        td = (date.today() + timedelta(days=2)).isoformat()
        fs = fetch_and_store("NYC", td, conn)
        print(f"  OK  {len(fs)} model forecast(s) for NYC/{td}")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[3] SQLite DB")
    try:
        conn = init_db()
        assert_db_integrity(conn)
        print(f"  OK  {Path('data/bot.db').resolve()}")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[4] Polymarket — connectivity")
    try:
        import requests
        r = requests.get("https://gamma-api.polymarket.com/markets?limit=1", timeout=5)
        print(f"  OK  status={r.status_code}")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[4b] Polymarket — live weather market fetch")
    try:
        from clients.polymarket_client import fetch_all_weather_markets as _p
        pm = _p()
        if pm:
            print(f"  {len(pm)} future weather markets found")
            for m in pm[:3]:
                print(f"     {m['city']}  {m['target_date']}  price={m['market_price']:.2f}")
        else:
            print("  0 future markets on Polymarket today")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print("\n[5] NOAA settlement (IEM fallback)")
    try:
        from clients.nws_settlement import fetch_settlement
        from datetime import date, timedelta
        td = (date.today() - timedelta(days=3)).isoformat()
        r = fetch_settlement("NYC", td)
        print(f"  OK  NYC/{td}: {r['actual_high_f']:.1f}°F (source={r['source']})")
    except Exception as exc:
        print(f"  FAIL  {exc}")

    print()


def run_status() -> None:
    """Show current portfolio and open positions."""
    conn = init_db()
    _header("STATUS")

    row = conn.execute("SELECT * FROM portfolio WHERE id=1").fetchone()
    if row:
        print(f"  Bankroll         ${row['bankroll']:.2f}")
        print(f"  Cash available   ${row['cash_available']:.2f}")
        print(f"  Total PnL        ${row['total_pnl'] or 0:.2f}")

    positions = conn.execute(
        """SELECT * FROM positions
           WHERE status NOT IN ('WON','LOST','EXITED_CONVERGENCE',
                                'EXITED_STOP','EXITED_PRE_SETTLEMENT')
           ORDER BY opened_at DESC"""
    ).fetchall()
    print(f"\n  Open Positions: {len(positions)}")
    for p in positions:
        print(f"    {p['ticker']:<30} {p['side']:<4} ${p['size_usd']:.0f}  "
              f"entry={p['entry_price']:.3f}  status={p['status']}")

    closed = conn.execute(
        "SELECT COUNT(*), SUM(realized_pnl) FROM positions WHERE realized_pnl IS NOT NULL"
    ).fetchone()
    print(f"\n  Closed Positions: {closed[0] or 0}  Total realized PnL: ${closed[1] or 0:.2f}")


def run_once(paper: bool = True, live: bool = False) -> dict:
    """
    Run a single trading cycle.

    paper=True  → fetch real market data, execute paper trades (log to DB only)
    live=True   → fetch real data AND place real exchange orders
    """
    _print_config()

    dry_run = False  # execution is controlled by executor choice (Paper vs Kalshi), not this flag

    conn = init_db()
    migrate_from_json()

    # Apply BANKROLL from env only if DB was just initialised with the default $1000
    bankroll_env = float(os.environ.get("BANKROLL", 1000.0))
    row = conn.execute("SELECT bankroll FROM portfolio WHERE id=1").fetchone()
    if row and abs(float(row["bankroll"]) - 1000.0) < 0.01 and abs(bankroll_env - 1000.0) > 0.01:
        conn.execute(
            "UPDATE portfolio SET bankroll=?, cash_available=?, updated_at=? WHERE id=1",
            (bankroll_env, bankroll_env, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        print(f"  Bankroll set from .env: ${bankroll_env:.2f}")

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
    settlement_results = _settle_expired_positions(conn, open_positions)

    summary = brain.run_cycle(
        markets=markets,
        forecasts=forecasts,
        open_positions=open_positions,
        settlement_results=settlement_results,
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


def autoloop(interval_seconds: int | None = None, paper: bool = True, live: bool = False) -> None:
    interval_seconds = interval_seconds or int(os.environ.get("POLL_INTERVAL_S", 300))
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
    parser.add_argument("--interval", type=int, default=None,
                        help="Loop interval in seconds (default: POLL_INTERVAL_S from .env or 300)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration cycle and exit")
    parser.add_argument("--dashboard", action="store_true",
                        help="Start dashboard API server")
    parser.add_argument("--port", type=int, default=5001,
                        help="Dashboard port (default: 5001)")
    parser.add_argument("--autoresearch", action="store_true",
                        help="Run one autoresearch experiment cycle")
    parser.add_argument("--diagnose", action="store_true",
                        help="Test all APIs and DB connectivity")
    parser.add_argument("--status", action="store_true",
                        help="Show portfolio + open positions, no trading")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard(port=args.port)
    elif args.diagnose:
        run_diagnose()
    elif args.status:
        run_status()
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
