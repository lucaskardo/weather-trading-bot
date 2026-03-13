"""
NWS Settlement Truth — fetch official daily high temperatures.

Uses official station data (NOT band midpoints) for contract settlement.

Primary:  NOAA Climate Data Online API (requires NOAA_CDO_TOKEN env var)
Fallback: Iowa Environmental Mesonet (IEM) ASOS — no auth required
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Allow overriding DB path for tests
_DB_PATH_OVERRIDE: Path | None = None

# --------------------------------------------------------------------------- #
# Station map: city → ICAO station code
# --------------------------------------------------------------------------- #
STATION_MAP: dict[str, str] = {
    "NYC": "KLGA",   # LaGuardia
    "CHI": "KORD",   # O'Hare
    "MIA": "KMIA",   # Miami International
    "LA":  "KLAX",   # Los Angeles International
    "DC":  "KDCA",   # Reagan National
    "SF":  "KSFO",   # San Francisco International
    "HOU": "KIAH",   # Houston George Bush
    "BOS": "KBOS",   # Logan
    "DAL": "KDFW",   # Dallas/Fort Worth
    "ATL": "KATL",   # Hartsfield-Jackson
    "SEA": "KSEA",   # Seattle-Tacoma
    # Europe / international
    "LON": "EGLL",   # London Heathrow
    "PAR": "LFPG",   # Paris CDG
    "MUN": "EDDM",   # Munich
    "SEO": "RKSI",   # Seoul Incheon
    "BUE": "SAEZ",   # Buenos Aires Ezeiza
    "SAO": "SBGR",   # São Paulo Guarulhos
}

# NOAA CDO base URL
_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
# IEM ASOS base URL
_IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_REQUEST_TIMEOUT = 15  # seconds


class SettlementError(Exception):
    """Raised when settlement temperature cannot be determined."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def fetch_settlement(city: str, target_date: str) -> dict[str, Any]:
    """
    Return the official daily high temperature for *city* on *target_date*.

    Args:
        city:        City code, e.g. "NYC". Must exist in STATION_MAP.
        target_date: ISO date string "YYYY-MM-DD".

    Returns:
        {
            "actual_high_f": float,
            "station": str,       # ICAO code
            "source_url": str,    # URL used to retrieve the data
            "source": str,        # "noaa_cdo" | "iem_asos" | "cache"
        }

    Raises:
        SettlementError: if neither source can provide data.
        KeyError: if city not in STATION_MAP.
    """
    if city not in STATION_MAP:
        raise KeyError(f"Unknown city '{city}'. Add it to STATION_MAP.")

    station = STATION_MAP[city]

    # 1. Check cache first
    cached = _load_cache(city, target_date)
    if cached is not None:
        return cached

    # 2. Try NOAA CDO
    noaa_token = os.environ.get("NOAA_CDO_TOKEN", "")
    if noaa_token:
        try:
            result = _fetch_noaa_cdo(station, target_date, noaa_token)
            _save_cache(city, target_date, result)
            return result
        except Exception:
            pass  # fall through to IEM

    # 3. Fallback: IEM ASOS
    try:
        result = _fetch_iem_asos(station, target_date)
        _save_cache(city, target_date, result)
        return result
    except Exception as exc:
        raise SettlementError(
            f"Could not fetch settlement for {city} ({station}) on {target_date}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# NOAA Climate Data Online
# --------------------------------------------------------------------------- #

def _fetch_noaa_cdo(station: str, target_date: str, token: str) -> dict[str, Any]:
    """
    Fetch TMAX from NOAA CDO Daily Summaries.

    CDO returns temperature in tenths of °C; convert to °F.
    """
    url = f"{_CDO_BASE}/data"
    params = {
        "datasetid": "GHCND",
        "stationid": f"GHCND:{station}",
        "datatypeid": "TMAX",
        "startdate": target_date,
        "enddate": target_date,
        "units": "standard",   # returns °F directly
        "limit": 1,
    }
    headers = {"token": token}
    resp = requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        raise SettlementError(f"NOAA CDO returned no TMAX data for {station} on {target_date}")

    high_f = float(results[0]["value"])
    source_url = resp.url
    return {
        "actual_high_f": high_f,
        "station": station,
        "source_url": source_url,
        "source": "noaa_cdo",
    }


# --------------------------------------------------------------------------- #
# IEM ASOS fallback
# --------------------------------------------------------------------------- #

def _fetch_iem_asos(station: str, target_date: str) -> dict[str, Any]:
    """
    Fetch daily max temperature from IEM ASOS hourly observations.

    IEM does not have a dedicated daily-max endpoint, so we pull all
    hourly obs for the day and take the max of the 'tmpf' (air temp °F) column.
    """
    params = {
        "station": station,
        "data": "tmpf",
        "year1": target_date[:4],
        "month1": target_date[5:7],
        "day1": target_date[8:10],
        "year2": target_date[:4],
        "month2": target_date[5:7],
        "day2": target_date[8:10],
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "1",  # 1=ASOS, 3=hourly
    }
    url = _IEM_BASE
    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    high_f = _parse_iem_response(resp.text, station, target_date)
    return {
        "actual_high_f": high_f,
        "station": station,
        "source_url": resp.url,
        "source": "iem_asos",
    }


def _parse_iem_response(text: str, station: str, target_date: str) -> float:
    """Parse IEM CSV text and return daily max °F."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Skip header line(s) — IEM includes a header row
    data_lines = [l for l in lines if not l.startswith("station") and not l.startswith("#")]

    highs: list[float] = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_temp = parts[2].strip()
        if raw_temp in ("M", "T", ""):
            continue
        try:
            highs.append(float(raw_temp))
        except ValueError:
            continue

    if not highs:
        raise SettlementError(
            f"IEM ASOS returned no valid temperature observations for {station} on {target_date}"
        )
    return max(highs)


# --------------------------------------------------------------------------- #
# SQLite cache helpers
# --------------------------------------------------------------------------- #

def _get_conn() -> sqlite3.Connection | None:
    """Return a connection to the bot DB, or None if unavailable."""
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


def _load_cache(city: str, target_date: str) -> dict[str, Any] | None:
    conn = _get_conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT * FROM settlement_cache WHERE city=? AND target_date=?",
            (city, target_date),
        ).fetchone()
        if row is None:
            return None
        return {
            "actual_high_f": row["actual_high_f"],
            "station": row["station"],
            "source_url": row["source_url"] or "",
            "source": "cache",
        }
    except Exception:
        return None
    finally:
        conn.close()


def _save_cache(city: str, target_date: str, result: dict[str, Any]) -> None:
    conn = _get_conn()
    if conn is None:
        return
    try:
        conn.execute(
            """INSERT OR REPLACE INTO settlement_cache
               (city, target_date, actual_high_f, station, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                city,
                target_date,
                result["actual_high_f"],
                result["station"],
                result.get("source_url", ""),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
