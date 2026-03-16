#!/usr/bin/env python3
"""Non-destructive reconciliation smoke flow.

Fetches live/public market snapshots, maps them to local open positions by
 ticker, and runs reconciliation in preview mode by default. This validates
 that local DB positions are still visible in current exchange market data and
 highlights price/status mismatches before enabling live capital.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clients.kalshi_client import fetch_all_weather_markets as fetch_kalshi_markets
from clients.polymarket_client import fetch_all_weather_markets as fetch_polymarket_markets
from risk.reconciliation import build_exchange_positions_from_markets, reconcile_positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "bot.db"))
    parser.add_argument("--exchange", choices=["kalshi", "polymarket", "all"], default="all")
    parser.add_argument("--auto-correct", action="store_true", help="Apply DB corrections instead of preview only")
    args = parser.parse_args()

    markets = []
    try:
        if args.exchange in ("kalshi", "all"):
            markets.extend(fetch_kalshi_markets())
        if args.exchange in ("polymarket", "all"):
            markets.extend(fetch_polymarket_markets())
    except Exception as exc:
        print(f"Reconciliation smoke fetch failed: {exc}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        exchange_positions = build_exchange_positions_from_markets(conn, markets)
        result = reconcile_positions(conn, exchange_positions, auto_correct=args.auto_correct)
        payload = {
            "db": args.db,
            "exchange": args.exchange,
            "markets_fetched": len(markets),
            "mapped_exchange_positions": len(exchange_positions),
            "local_positions": result.local_positions,
            "exchange_positions": result.exchange_positions,
            "discrepancies": len(result.discrepancies),
            "critical": result.critical_count,
            "corrections": result.corrections,
            "auto_correct": args.auto_correct,
            "items": [
                {
                    "type": d.discrepancy_type,
                    "ticker": d.ticker,
                    "detail": d.detail,
                    "critical": d.is_critical(),
                }
                for d in result.discrepancies
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 1 if result.critical_count else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
