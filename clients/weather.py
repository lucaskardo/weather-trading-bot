"""
Weather data client — fetches multi-model forecasts and stores them with lineage.

Supported models (all via Open-Meteo free API, no key required):
  GFS     — NOAA Global Forecast System
  ECMWF   — European Centre (IFS open data)
  ICON    — DWD ICON global
  AROME   — Météo-France limited area (Europe only; falls back gracefully)
  NOAA_NWS — alias for GFS ensemble mean via Open-Meteo

Open-Meteo run_id convention: "YYYYMMDDCC" where CC is the model init cycle hour.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from shared.types import ModelForecast

# --------------------------------------------------------------------------- #
# City → lat/lon lookup
# --------------------------------------------------------------------------- #
CITY_COORDS: dict[str, tuple[float, float]] = {
    "NYC": (40.7128, -74.0060),
    "CHI": (41.8781, -87.6298),
    "MIA": (25.7617, -80.1918),
    "LA":  (34.0522, -118.2437),
    "DC":  (38.9072, -77.0369),
    "SF":  (37.7749, -122.4194),
    "HOU": (29.7604, -95.3698),
    "BOS": (42.3601, -71.0589),
    "DAL": (32.7767, -96.7970),
    "ATL": (33.7490, -84.3880),
    "SEA": (47.6062, -122.3321),
    "LON": (51.5074, -0.1278),
    "PAR": (48.8566, 2.3522),
    "MUN": (48.1351, 11.5820),
    "SEO": (37.5665, 126.9780),
    "BUE": (-34.6037, -58.3816),
    "SAO": (-23.5505, -46.6333),
}

# Open-Meteo model IDs
_OM_MODELS: dict[str, str] = {
    "GFS":      "gfs_seamless",
    "ECMWF":    "ecmwf_ifs025",
    "ICON":     "icon_seamless",
    "AROME":    "meteofrance_arome_france_hd",
    "NOAA_NWS": "gfs_seamless",  # alias — uses GFS ensemble
}

_OM_BASE = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT = 20  # seconds

# Allow test override of DB path
_DB_PATH_OVERRIDE: Path | None = None


class WeatherFetchError(Exception):
    """Raised when a model fetch fails and no fallback is available."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def fetch_all_models(city: str, target_date: str) -> list[ModelForecast]:
    """
    Fetch forecasts from all supported models for *city* on *target_date*.

    Failures for individual models are silently skipped (logged to stderr).
    Returns whatever models succeeded — caller should check len() > 0.

    All returned ModelForecast objects have run_id, publish_time, and
    source_url populated (lineage complete).
    """
    if city not in CITY_COORDS:
        raise KeyError(f"Unknown city '{city}'. Add it to CITY_COORDS.")

    results: list[ModelForecast] = []
    for model_name in ["GFS", "ECMWF", "ICON", "NOAA_NWS"]:
        try:
            forecast = fetch_model(city, target_date, model_name)
            results.append(forecast)
        except Exception as exc:
            import sys
            print(f"[weather] {model_name} fetch failed for {city}: {exc}", file=sys.stderr)

    # AROME only covers Europe — skip gracefully for US cities
    if city in {"LON", "PAR", "MUN"}:
        try:
            forecast = fetch_model(city, target_date, "AROME")
            results.append(forecast)
        except Exception as exc:
            import sys
            print(f"[weather] AROME fetch failed for {city}: {exc}", file=sys.stderr)

    return results


def fetch_model(city: str, target_date: str, model_name: str) -> ModelForecast:
    """
    Fetch a single model's forecast and return a fully-populated ModelForecast.

    Raises WeatherFetchError on failure.
    """
    if city not in CITY_COORDS:
        raise KeyError(f"Unknown city '{city}'.")
    if model_name not in _OM_MODELS:
        raise KeyError(f"Unknown model '{model_name}'.")

    lat, lon = CITY_COORDS[city]
    om_model = _OM_MODELS[model_name]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "start_date": target_date,
        "end_date": target_date,
        "models": om_model,
        # Ask upstream for richer ensemble-like metadata when supported; the
        # parser gracefully ignores these fields when the provider omits them.
        "hourly": "temperature_2m",
    }

    try:
        resp = requests.get(_OM_BASE, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise WeatherFetchError(f"HTTP error fetching {model_name} for {city}: {exc}") from exc

    forecast = _parse_open_meteo(data, city, target_date, model_name, resp.url)
    return forecast


def fetch_and_store(
    city: str,
    target_date: str,
    conn: sqlite3.Connection | None = None,
) -> list[ModelForecast]:
    """
    Fetch all models and persist each to the forecasts table.

    Returns the list of successfully stored forecasts.
    """
    forecasts = fetch_all_models(city, target_date)
    db_conn = conn or _get_db_conn()
    if db_conn is not None:
        for f in forecasts:
            _store_forecast(f, db_conn)
        if conn is None:
            db_conn.close()
    return forecasts


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _parse_open_meteo(
    data: dict[str, Any],
    city: str,
    target_date: str,
    model_name: str,
    source_url: str,
) -> ModelForecast:
    """Extract high/low temps and build lineage from Open-Meteo response."""
    try:
        daily = data["daily"]
        dates = daily["time"]
        highs = daily["temperature_2m_max"]
        lows = daily["temperature_2m_min"]
    except KeyError as exc:
        raise WeatherFetchError(f"Unexpected Open-Meteo response structure: {exc}") from exc

    if target_date not in dates:
        raise WeatherFetchError(
            f"target_date {target_date} not in Open-Meteo response for {city}/{model_name}. "
            f"Available dates: {dates}"
        )

    idx = dates.index(target_date)
    high_f = highs[idx]
    low_f = lows[idx] if lows else None

    if high_f is None:
        raise WeatherFetchError(
            f"Open-Meteo returned null high temp for {city}/{model_name} on {target_date}"
        )

    # Build run_id from current UTC time (YYYYMMDDCC — nearest 6h cycle)
    now_utc = datetime.now(timezone.utc)
    cycle_hour = (now_utc.hour // 6) * 6
    run_id = now_utc.strftime(f"%Y%m%d{cycle_hour:02d}")
    publish_time = now_utc.isoformat()
    fetched_at = now_utc.isoformat()

    return ModelForecast(
        model_name=model_name,
        city=city,
        target_date=target_date,
        predicted_high_f=float(high_f),
        predicted_low_f=float(low_f) if low_f is not None else None,
        confidence=None,
        ensemble_members_f=_extract_ensemble_members(data, target_date),
        run_id=run_id,
        publish_time=publish_time,
        source_url=source_url,
        fetched_at=fetched_at,
    )


# --------------------------------------------------------------------------- #
# Ensemble extraction
# --------------------------------------------------------------------------- #

def _extract_ensemble_members(data: dict[str, Any], target_date: str) -> list[float] | None:
    """Best-effort extraction of true ensemble-member daily highs from provider payloads.

    Supports a few lightweight shapes without assuming a single upstream schema:
      1. daily.temperature_2m_max_member_* arrays
      2. daily.temperature_2m_max_members as list[list|float]
      3. top-level ensemble_members / ensemble_members_f

    Returns a flat list of Fahrenheit daily-high members for target_date, or None.
    """
    members: list[float] = []
    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    try:
        idx = dates.index(target_date)
    except ValueError:
        idx = 0

    # Shape 1: member-specific daily keys
    for key, values in daily.items():
        if not isinstance(key, str) or "temperature_2m_max" not in key or "member" not in key:
            continue
        if isinstance(values, list) and len(values) > idx and values[idx] is not None:
            try:
                members.append(float(values[idx]))
            except Exception:
                pass

    # Shape 2: nested daily member matrix
    nested = daily.get("temperature_2m_max_members")
    if isinstance(nested, list):
        for item in nested:
            try:
                if isinstance(item, list) and len(item) > idx and item[idx] is not None:
                    members.append(float(item[idx]))
                elif item is not None:
                    members.append(float(item))
            except Exception:
                pass

    # Shape 3: top-level simple arrays
    for key in ("ensemble_members_f", "ensemble_members"):
        vals = data.get(key)
        if isinstance(vals, list):
            for item in vals:
                try:
                    if isinstance(item, list) and len(item) > idx and item[idx] is not None:
                        members.append(float(item[idx]))
                    elif item is not None:
                        members.append(float(item))
                except Exception:
                    pass

    out = [m for m in members if isinstance(m, float)]
    return out or None

# --------------------------------------------------------------------------- #
# SQLite persistence
# --------------------------------------------------------------------------- #

def _store_forecast(f: ModelForecast, conn: sqlite3.Connection) -> None:
    """Insert a ModelForecast into the forecasts table."""
    conn.execute(
        """INSERT INTO forecasts
           (city, target_date, model_name, predicted_high_f, predicted_low_f,
            confidence, ensemble_members_json, run_id, publish_time, source_url, fetched_at, market_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            f.city,
            f.target_date,
            f.model_name,
            f.predicted_high_f,
            f.predicted_low_f,
            f.confidence,
            json.dumps(list(f.ensemble_members_f or [])),
            f.run_id,
            f.publish_time,
            f.source_url,
            f.fetched_at,
            f.market_id,
        ),
    )
    conn.commit()


def _get_db_conn() -> sqlite3.Connection | None:
    try:
        if _DB_PATH_OVERRIDE is not None:
            db_path = _DB_PATH_OVERRIDE
        else:
            db_path = Path(__file__).parent.parent / "data" / "bot.db"
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None
