"""
Phase B3 — HRRR intraday nowcast client.

HRRR (High-Resolution Rapid Refresh) updates hourly at 3km resolution.
For same-day temperature markets it provides the most current forecast.

Strategy:
  - Fetch HRRR hourly max temperature for the target city/date
  - For a same-day contract: combine observed max-so-far with HRRR
    remaining-day forecast to estimate final daily high
  - Signal when HRRR-implied probability significantly disagrees with
    current market price (delta > HRRR_MIN_EDGE threshold)

Data source: Open-Meteo HRRR endpoint (no API key required).
    https://api.open-meteo.com/v1/forecast?models=best_match&...
    We request hourly temperature_2m_max at 1-hour resolution.

Fallback: if HRRR unavailable, returns None so callers can degrade gracefully.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from typing import Any, Optional

import requests

_OM_URL = "https://api.open-meteo.com/v1/forecast"

# Minimum edge (model_prob - market_price) for HRRR signal to fire
HRRR_MIN_EDGE = 0.07

# City coordinates (lat, lon) — same as in clients/weather.py
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "NYC": (40.7128, -74.0060),
    "CHI": (41.8781, -87.6298),
    "LA":  (34.0522, -118.2437),
    "DC":  (38.9072, -77.0369),
    "SF":  (37.7749, -122.4194),
    "HOU": (29.7604, -95.3698),
    "MIA": (25.7617, -80.1918),
    "BOS": (42.3601, -71.0589),
    "DAL": (32.7767, -96.7970),
    "ATL": (33.7490, -84.3880),
    "SEA": (47.6062, -122.3321),
}


def fetch_hrrr_nowcast(
    city: str,
    target_date: str,
    session: Optional[requests.Session] = None,
    timeout: int = 10,
) -> Optional[dict[str, Any]]:
    """
    Fetch the HRRR-based temperature nowcast for a city on a given date.

    For same-day contracts:
      - Combines observed max-so-far (hours before now) with HRRR forecast
        for remaining hours to estimate the final daily high.

    For future-day contracts:
      - Returns the HRRR 24-hour max temperature directly.

    Args:
        city:        City code (e.g. "NYC").
        target_date: ISO date string (e.g. "2026-03-13").
        session:     Optional requests session for connection reuse.
        timeout:     HTTP timeout in seconds.

    Returns:
        Dict with keys:
            hrrr_high_f:      HRRR-implied daily high (°F)
            observed_max_f:   Max temperature observed so far today (or None)
            hours_remaining:  Hours left in the day at time of call
            source:           "hrrr_observed_blend" or "hrrr_forecast_only"
        Returns None on any error.
    """
    coords = _CITY_COORDS.get(city)
    if not coords:
        return None

    lat, lon = coords
    s = session or requests.Session()

    try:
        raw = _fetch_om_hrrr(s, lat, lon, target_date, timeout)
    except Exception as exc:
        print(f"[hrrr] fetch error for {city}/{target_date}: {exc}", file=sys.stderr)
        return None

    return _compute_nowcast(raw, target_date)


def hrrr_signal_edge(
    hrrr_high_f: float,
    threshold_f: float,
    base_std_f: float,
    market_price: float,
) -> float:
    """
    Compute the edge of a HRRR-based signal vs. current market price.

    Uses a simple normal distribution to convert HRRR temperature forecast
    to a probability, then computes (model_prob - market_price).

    Returns:
        Edge as float. Positive = market underpricing YES.
    """
    from math import erfc, sqrt
    z = (threshold_f - hrrr_high_f) / (base_std_f * sqrt(2))
    model_prob = max(1e-6, min(1 - 1e-6, 0.5 * erfc(z)))
    return model_prob - market_price


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fetch_om_hrrr(
    session: requests.Session,
    lat: float,
    lon: float,
    target_date: str,
    timeout: int,
) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "models": "gfs_seamless",   # Open-Meteo best available hourly
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "UTC",
    }
    resp = session.get(_OM_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _compute_nowcast(raw: dict, target_date: str) -> Optional[dict[str, Any]]:
    """Parse Open-Meteo hourly response and compute daily high estimate."""
    hourly = raw.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None

    # Filter to target date
    date_temps = [
        t for time_str, t in zip(times, temps)
        if time_str and time_str.startswith(target_date) and t is not None
    ]

    if not date_temps:
        return None

    now_utc = datetime.now(timezone.utc)
    target_dt = date.fromisoformat(target_date)
    is_today = (target_dt == now_utc.date())

    if is_today:
        # Split into observed (past hours) and forecast (future hours)
        current_hour_utc = now_utc.hour
        observed = [
            t for i, t in enumerate(date_temps) if i < current_hour_utc and t is not None
        ]
        remaining = [
            t for i, t in enumerate(date_temps) if i >= current_hour_utc and t is not None
        ]
        observed_max = max(observed) if observed else None
        forecast_max = max(remaining) if remaining else None

        # Daily high = max(observed so far, HRRR forecast for remaining)
        candidates = [x for x in [observed_max, forecast_max] if x is not None]
        if not candidates:
            return None
        hrrr_high_f = max(candidates)

        return {
            "hrrr_high_f": hrrr_high_f,
            "observed_max_f": observed_max,
            "hours_remaining": len(remaining),
            "source": "hrrr_observed_blend",
        }
    else:
        # Future day — use full HRRR 24h max
        valid = [t for t in date_temps if t is not None]
        if not valid:
            return None
        return {
            "hrrr_high_f": max(valid),
            "observed_max_f": None,
            "hours_remaining": 24,
            "source": "hrrr_forecast_only",
        }
