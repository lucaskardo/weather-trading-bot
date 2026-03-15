"""
Polymarket market data client.

Fetches live weather markets from the Polymarket Gamma API (no auth required).
Normalises markets to the bot's standard dict format.

API: https://gamma-api.polymarket.com/events?tag_slug=weather&active=true
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
WEATHER_TAGS = ["weather", "temperature", "daily-temperature", "climate-weather"]

# Map Polymarket city name patterns to bot city codes
_CITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnew york\b", re.I),      "NYC"),
    (re.compile(r"\bnyc\b",       re.I),      "NYC"),
    (re.compile(r"\bchicago\b",   re.I),      "CHI"),
    (re.compile(r"\blos angeles\b", re.I),    "LA"),
    (re.compile(r"\b\bla\b\b",    re.I),      "LA"),
    (re.compile(r"\bwashington\b", re.I),     "DC"),
    (re.compile(r"\bsan francisco\b", re.I),  "SF"),
    (re.compile(r"\bhouston\b",   re.I),      "HOU"),
    (re.compile(r"\bmiami\b",     re.I),      "MIA"),
    (re.compile(r"\bboston\b",    re.I),      "BOS"),
    (re.compile(r"\batlanta\b",   re.I),      "ATL"),
    (re.compile(r"\bdallas\b",    re.I),      "DAL"),
    (re.compile(r"\bseattle\b",   re.I),      "SEA"),
    (re.compile(r"\blondon\b",    re.I),      "LON"),
    (re.compile(r"\bparis\b",     re.I),      "PAR"),
]

# Temperature threshold patterns
# "above 68°F", "below 72 degrees", "exceed 65°F", "between 65 and 70"
_ABOVE_RE     = re.compile(r"(?:above|exceed|over|higher than)\s+(\d+(?:\.\d+)?)\s*°?[FC]?", re.I)
_BELOW_RE     = re.compile(r"(?:below|under|less than|lower than)\s+(\d+(?:\.\d+)?)\s*°?[FC]?", re.I)
_BAND_RE      = re.compile(r"(?:between|from)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", re.I)
_DATE_RE      = re.compile(r"(\w+ \d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})")
# Polymarket outcome formats: "56°F or higher", "50°F or lower", "54-55°F"
_PM_OR_HIGHER = re.compile(r"(\d+(?:\.\d+)?)\s*°?F?\s+or\s+higher", re.I)
_PM_OR_LOWER  = re.compile(r"(\d+(?:\.\d+)?)\s*°?F?\s+or\s+lower", re.I)
_PM_RANGE     = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*°?F", re.I)


def fetch_all_weather_markets(
    session: Optional[requests.Session] = None,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch all active Polymarket weather markets and return normalised dicts.

    Each dict contains:
        id, ticker, city, target_date, market_type (above/below/band),
        high_f, low_f, market_price (0-1), exchange, volume, open_interest
    """
    s = session or requests.Session()
    markets: list[dict[str, Any]] = []

    events = _fetch_weather_events(s, timeout)
    seen_market_ids: set = set()

    for event in events:
        for market in event.get("markets", []):
            parsed = _parse_market(market, event)
            if parsed:
                mid = parsed["id"]
                if mid and mid not in seen_market_ids:
                    seen_market_ids.add(mid)
                    markets.append(parsed)

    return markets


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fetch_weather_events(
    session: requests.Session,
    timeout: int,
) -> list[dict]:
    url = f"{GAMMA_BASE}/events"
    seen_ids: set = set()
    all_events: list[dict] = []

    for tag in WEATHER_TAGS:
        try:
            params = {"tag_slug": tag, "active": "true", "limit": 100}
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Gamma API returns list or {"events": [...]}
            events = data if isinstance(data, list) else data.get("events", [])
            for event in events:
                eid = event.get("id") or event.get("slug")
                if eid:
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        all_events.append(event)
                else:
                    all_events.append(event)
        except Exception as exc:
            print(f"[polymarket] error fetching tag {tag}: {exc}", file=sys.stderr)

    return all_events


def _parse_market(market: dict, event: dict) -> Optional[dict[str, Any]]:
    """Parse a raw Polymarket market into the bot's normalised format."""
    question = market.get("question") or event.get("title") or ""
    description = market.get("description") or ""
    full_text = f"{question} {description}"

    city = _extract_city(full_text)
    if not city:
        return None

    target_date = _extract_date(full_text)
    if not target_date:
        return None

    market_type, high_f, low_f = _extract_threshold(full_text)
    if high_f is None and low_f is None:
        return None

    # Market price from best ask or midpoint
    outcomes = market.get("outcomePrices") or []
    try:
        # outcomePrices = ["0.72", "0.28"] → YES price is first
        yes_price = float(outcomes[0]) if outcomes else 0.5
    except (IndexError, ValueError, TypeError):
        yes_price = 0.5

    yes_price = max(0.01, min(0.99, yes_price))

    ticker = market.get("conditionId") or market.get("id") or ""

    return {
        "id": ticker,
        "ticker": ticker,
        "city": city,
        "target_date": target_date,
        "market_type": market_type,
        "high_f": high_f,
        "low_f": low_f,
        "market_price": yes_price,
        "exchange": "polymarket",
        "volume": float(market.get("volume") or 0),
        "open_interest": float(market.get("liquidity") or 0),
    }


def _extract_city(text: str) -> Optional[str]:
    for pattern, city in _CITY_PATTERNS:
        if pattern.search(text):
            return city
    return None


def _extract_date(text: str) -> Optional[str]:
    """Try to extract an ISO date from the market text."""
    from datetime import datetime, date as date_type

    # Direct ISO format
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        return iso.group(1)

    # Natural language with year: "March 13, 2026" or "March 13 2026"
    natural = re.search(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
        text, re.I
    )
    if natural:
        try:
            dt = datetime.strptime(
                f"{natural.group(1)} {natural.group(2)} {natural.group(3)}",
                "%B %d %Y"
            )
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Polymarket "on March 15" format — no year, assume current year
    no_year = re.search(
        r"\bon\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{1,2})\b",
        text, re.I
    )
    if no_year:
        try:
            current_year = date_type.today().year
            dt = datetime.strptime(
                f"{no_year.group(1)} {no_year.group(2)} {current_year}",
                "%B %d %Y"
            )
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def _extract_threshold(text: str) -> tuple[str, Optional[float], Optional[float]]:
    """Return (market_type, high_f, low_f)."""
    # Polymarket outcome formats (checked first — more specific)
    pm_higher = _PM_OR_HIGHER.search(text)
    if pm_higher:
        return "above", float(pm_higher.group(1)), None

    pm_lower = _PM_OR_LOWER.search(text)
    if pm_lower:
        return "below", float(pm_lower.group(1)), None

    pm_range = _PM_RANGE.search(text)
    if pm_range:
        low_f = float(pm_range.group(1))
        high_f = float(pm_range.group(2))
        return "band", high_f, low_f

    band = _BAND_RE.search(text)
    if band:
        return "band", float(band.group(2)), float(band.group(1))

    above = _ABOVE_RE.search(text)
    if above:
        return "above", float(above.group(1)), None

    below = _BELOW_RE.search(text)
    if below:
        return "below", float(below.group(1)), None

    return "above", None, None
