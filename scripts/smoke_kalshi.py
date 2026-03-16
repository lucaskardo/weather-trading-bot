#!/usr/bin/env python3
"""Non-destructive Kalshi API smoke test.

Validates auth/env wiring and basic market-data connectivity without placing
orders. Safe to run before enabling live trading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from clients.kalshi_client import WEATHER_SERIES, fetch_all_weather_markets, fetch_orderbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default=WEATHER_SERIES[0])
    parser.add_argument("--ticker", default="")
    args = parser.parse_args()

    print(json.dumps({
        "api_key_present": bool(os.environ.get("KALSHI_API_KEY")),
        "private_key_present": bool(os.environ.get("KALSHI_PRIVATE_KEY")),
        "series": args.series,
    }, indent=2))

    try:
        markets = [m for m in fetch_all_weather_markets() if m.get("ticker", "").startswith(args.series)]
        print(json.dumps({"markets_found": len(markets)}, indent=2))
        if not markets:
            print("No markets returned for requested series", file=sys.stderr)
            return 1
        m = next((x for x in markets if x["ticker"] == args.ticker), markets[0])
        ob = fetch_orderbook(m["ticker"])
        print(json.dumps({
            "sample_market": {
                "ticker": m["ticker"],
                "city": m["city"],
                "target_date": m["target_date"],
                "market_type": m["market_type"],
                "market_price": m["market_price"],
            },
            "orderbook_levels": len(ob),
            "best_yes_price": ob[0]["price"] if ob else None,
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"Kalshi smoke test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
