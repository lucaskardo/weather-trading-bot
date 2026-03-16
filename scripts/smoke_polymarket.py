#!/usr/bin/env python3
"""Non-destructive Polymarket API smoke test."""
from __future__ import annotations

import json
import sys

from clients.polymarket_client import fetch_all_weather_markets


def main() -> int:
    try:
        markets = fetch_all_weather_markets()
        print(json.dumps({"markets_found": len(markets)}, indent=2))
        if not markets:
            print("No active polymarket weather markets found", file=sys.stderr)
            return 1
        m = markets[0]
        print(json.dumps({
            "sample_market": {
                "ticker": m["ticker"],
                "city": m["city"],
                "target_date": m["target_date"],
                "market_type": m["market_type"],
                "market_price": m["market_price"],
            }
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"Polymarket smoke test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
