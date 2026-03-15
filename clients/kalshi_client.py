"""
Kalshi market data client.

Fetches live weather markets from the Kalshi public API (no auth required
for read endpoints). Normalises markets to the bot's standard dict format.

Hierarchy: Series → Events → Markets

Supported series (US temperature highs):
    KXHIGHNY, KXHIGHCHI, KXHIGHLA, KXHIGHDC,
    KXHIGHSF, KXHIGHHOU, KXHIGHMIA, KXHIGHBOS
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

WEATHER_SERIES = [
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHLA", "KXHIGHDC",
    "KXHIGHSF", "KXHIGHHOU", "KXHIGHMIA", "KXHIGHBOS",
]

SERIES_CITY: dict[str, str] = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "CHI",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHSF":  "SF",
    "KXHIGHHOU": "HOU",
    "KXHIGHMIA": "MIA",
    "KXHIGHBOS": "BOS",
}

# Matches: KXHIGHNY-26MAR13-T68  or  KXHIGHNY-26MAR13-B65T66
_TICKER_RE = re.compile(
    r"(?P<series>[A-Z]+)-(?P<date>\d{2}[A-Z]{3}\d{2})-"
    r"(?:T(?P<threshold>\d+(?:\.\d+)?)"
    r"|B(?P<low>\d+(?:\.\d+)?)T(?P<high>\d+(?:\.\d+)?))"
)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# Subtitle patterns for band markets: "65 to 66 degrees" or "above 68 degrees"
_BAND_RE = re.compile(r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)")
_ABOVE_RE = re.compile(r"above\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_BELOW_RE = re.compile(r"below\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def fetch_all_weather_markets(
    session: Optional[requests.Session] = None,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch all open Kalshi weather markets and return normalised market dicts.

    Uses the official Kalshi v2 /markets endpoint with series_ticker filter:
      GET /markets?series_ticker=KXHIGHNY&status=open

    Each dict contains:
        id, ticker, city, target_date, market_type (above/below/band),
        high_f, low_f, market_price (0-1), exchange, volume, open_interest
    """
    s = session or requests.Session()
    markets: list[dict[str, Any]] = []

    for series_ticker in WEATHER_SERIES:
        city = SERIES_CITY.get(series_ticker)
        if not city:
            continue
        try:
            raw_markets = _fetch_markets_for_series(s, series_ticker, timeout)
            for m in raw_markets:
                if m.get("status") != "open":
                    continue
                parsed = _parse_market(m, city)
                if parsed:
                    markets.append(parsed)
        except Exception as exc:
            print(f"[kalshi] error fetching {series_ticker}: {exc}", file=sys.stderr)
        time.sleep(0.5)

    return markets


def fetch_orderbook(
    ticker: str,
    session: Optional[requests.Session] = None,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch the orderbook for a ticker.

    Returns list of {"price": float, "size_usd": float} dicts (YES side).
    """
    s = session or requests.Session()
    url = f"{KALSHI_BASE}/markets/{ticker}/orderbook"
    try:
        resp = s.get(url, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
        data = raw.get("orderbook", {})
        data_fp = raw.get("orderbook_fp", {})
    except Exception as exc:
        print(f"[kalshi] orderbook error for {ticker}: {exc}", file=sys.stderr)
        return []

    levels = []
    # Prefer new fixed-point format (orderbook_fp.yes_dollars) over legacy cents
    if data_fp and data_fp.get("yes_dollars"):
        for price_dollars, count_fp in (data_fp.get("yes_dollars", []) or []):
            price = float(price_dollars)
            size_usd = price * float(count_fp)
            levels.append({"price": price, "size_usd": size_usd})
    else:
        for price_cents, size in (data.get("yes", []) or []):
            price = price_cents / 100.0
            size_usd = price * size
            levels.append({"price": price, "size_usd": size_usd})

    return sorted(levels, key=lambda x: x["price"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fetch_markets_for_series(
    session: requests.Session,
    series_ticker: str,
    timeout: int,
) -> list[dict]:
    """
    Fetch all open markets for a series directly from /markets.

    Per Kalshi official docs (March 2026):
      GET /trade-api/v2/markets?series_ticker=KXHIGHNY&status=open

    Handles cursor-based pagination for series with many markets.
    """
    url = f"{KALSHI_BASE}/markets"
    all_markets: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("markets", [])
        all_markets.extend(batch)

        # Cursor pagination — empty cursor means no more pages
        cursor = data.get("cursor")
        if not cursor or not batch:
            break

    return all_markets


def _parse_market(m: dict, city: str) -> Optional[dict[str, Any]]:
    """Parse a raw Kalshi market dict into the bot's normalised format."""
    ticker = m.get("ticker", "")
    subtitle = m.get("subtitle") or m.get("title") or ""

    # Parse target date and threshold from ticker
    target_date, market_type, high_f, low_f = _parse_ticker(ticker, subtitle)
    if not target_date:
        return None

    # Market price: support both legacy cents (yes_bid/yes_ask) and
    # new fixed-point dollar strings (yes_bid_dollars/yes_ask_dollars).
    # Kalshi deprecated integer cents fields as of March 2026.
    def _read_price(m: dict, field_cents: str, field_dollars: str) -> float:
        """Read price as 0-1 probability, preferring _dollars fields."""
        dollars_val = m.get(field_dollars)
        if dollars_val is not None:
            try:
                return float(dollars_val)
            except (ValueError, TypeError):
                pass
        cents_val = m.get(field_cents)
        if cents_val is not None and cents_val != 0:
            try:
                return int(cents_val) / 100.0
            except (ValueError, TypeError):
                pass
        return 0.0

    yes_bid = _read_price(m, "yes_bid", "yes_bid_dollars")
    yes_ask = _read_price(m, "yes_ask", "yes_ask_dollars")
    if yes_ask > 0:
        market_price = yes_ask
    elif yes_bid > 0:
        market_price = yes_bid
    else:
        market_price = 0.5

    market_price = max(0.01, min(0.99, market_price))

    return {
        "id": ticker,
        "ticker": ticker,
        "city": city,
        "target_date": target_date,
        "market_type": market_type,
        "high_f": high_f,
        "low_f": low_f,
        "market_price": market_price,
        "exchange": "kalshi",
        "volume": m.get("volume", 0),
        "open_interest": m.get("open_interest", 0),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
    }


def _parse_ticker(
    ticker: str,
    subtitle: str,
) -> tuple[Optional[str], str, Optional[float], Optional[float]]:
    """
    Parse ticker and subtitle into (target_date, market_type, high_f, low_f).

    Ticker formats:
        KXHIGHNY-26MAR13-T68       → above/below, threshold=68
        KXHIGHNY-26MAR13-B65T66    → band, low=65, high=66
    """
    m = _TICKER_RE.search(ticker)
    if not m:
        return None, "above", None, None

    # Parse date: 26MAR13 → 2026-03-13
    date_str = m.group("date")
    target_date = _parse_kalshi_date(date_str)
    if not target_date:
        return None, "above", None, None

    # Band market
    if m.group("low") is not None:
        low_f = float(m.group("low"))
        high_f = float(m.group("high"))
        return target_date, "band", high_f, low_f

    # Above/below — determine from subtitle
    threshold = float(m.group("threshold"))
    if _BELOW_RE.search(subtitle):
        return target_date, "below", threshold, None

    # Default to "above" (most common for high-temp markets)
    return target_date, "above", threshold, None


def _parse_kalshi_date(date_str: str) -> Optional[str]:
    """
    Convert Kalshi date format to ISO date string.
    '26MAR13' → '2026-03-13'
    """
    try:
        year_short = int(date_str[:2])
        month_str = date_str[2:5].upper()
        day = int(date_str[5:])
        month = _MONTH_MAP.get(month_str)
        if not month:
            return None
        year = 2000 + year_short
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None
