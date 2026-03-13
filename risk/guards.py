"""
Phase 4.1 — Risk Guards.

Hard safety checks that can HALT trading or reject individual trades.
All guards raise typed exceptions so callers can handle them explicitly.

Guards:
  check_stale_forecast()    — HALT if best available forecast is too old
  check_cluster_exposure()  — REJECT trade if cluster cap would be breached
  check_daily_loss_limit()  — HALT if today's loss exceeds the daily limit
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from shared.params import Params, PARAMS, get_cluster
from shared.types import ModelForecast


# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #

class RiskViolation(Exception):
    """Base class for all risk guard violations."""


class StaleDataHalt(RiskViolation):
    """Raised when forecast data is too stale to trade safely."""


class ClusterCapExceeded(RiskViolation):
    """Raised when a new trade would breach the cluster exposure cap."""


class DailyLossHalt(RiskViolation):
    """Raised when daily loss limit has been reached."""


class CityLimitExceeded(RiskViolation):
    """Raised when per-city position limit would be breached."""


# --------------------------------------------------------------------------- #
# Guard 1: Stale Forecast
# --------------------------------------------------------------------------- #

def check_stale_forecast(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    params: Params = PARAMS,
) -> None:
    """
    Raise StaleDataHalt if all forecasts for city/date are older than
    params.stale_forecast_hours.

    Args:
        forecasts:   Available model forecasts.
        city:        City being traded.
        target_date: Contract target date.
        params:      Parameters (reads stale_forecast_hours).

    Raises:
        StaleDataHalt: if no fresh forecast exists.
    """
    relevant = [
        f for f in forecasts
        if f.city == city and f.target_date == target_date and f.fetched_at
    ]

    if not relevant:
        raise StaleDataHalt(
            f"No forecasts available for {city}/{target_date}. Trading halted."
        )

    now = datetime.now(timezone.utc)
    max_age = timedelta(hours=params.stale_forecast_hours)

    for f in relevant:
        try:
            fetched = datetime.fromisoformat(f.fetched_at)
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if (now - fetched) <= max_age:
                return  # at least one fresh forecast — OK
        except ValueError:
            continue

    oldest = min(
        (datetime.fromisoformat(f.fetched_at) for f in relevant if f.fetched_at),
        default=None,
    )
    age_str = f"{(now - oldest).total_seconds() / 3600:.1f}h" if oldest else "unknown"
    raise StaleDataHalt(
        f"All forecasts for {city}/{target_date} are stale "
        f"(oldest={age_str}, limit={params.stale_forecast_hours}h). Trading halted."
    )


def is_forecast_fresh(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    params: Params = PARAMS,
) -> bool:
    """Non-raising version of check_stale_forecast."""
    try:
        check_stale_forecast(forecasts, city, target_date, params)
        return True
    except StaleDataHalt:
        return False


# --------------------------------------------------------------------------- #
# Guard 2: Cluster Exposure
# --------------------------------------------------------------------------- #

def check_cluster_exposure(
    city: str,
    proposed_size_usd: float,
    open_positions: list[dict[str, Any]],
    bankroll: float,
    params: Params = PARAMS,
) -> None:
    """
    Raise ClusterCapExceeded if adding proposed_size_usd to the cluster
    containing *city* would breach params.max_cluster_exposure_pct.

    Args:
        city:             City of the proposed trade.
        proposed_size_usd: Size of the proposed trade in USD.
        open_positions:   All current open positions.
        bankroll:         Total bankroll in USD.
        params:           Parameters.

    Raises:
        ClusterCapExceeded: if the cap would be breached.
    """
    cluster = get_cluster(city, params)
    if cluster is None:
        return  # unknown cluster — no cap to enforce

    current_exposure = sum(
        pos.get("size_usd", 0.0)
        for pos in open_positions
        if get_cluster(pos.get("city", ""), params) == cluster
    )

    cap_usd = params.max_cluster_exposure_pct * bankroll
    if current_exposure + proposed_size_usd > cap_usd:
        raise ClusterCapExceeded(
            f"Cluster '{cluster}' exposure would be "
            f"${current_exposure + proposed_size_usd:.2f} "
            f"(cap=${cap_usd:.2f}, bankroll=${bankroll:.2f}). "
            f"Trade for {city} rejected."
        )


def check_city_limit(
    city: str,
    open_positions: list[dict[str, Any]],
    params: Params = PARAMS,
) -> None:
    """
    Raise CityLimitExceeded if adding another position in *city* would
    exceed params.max_positions_per_city.

    Raises:
        CityLimitExceeded
    """
    city_count = sum(1 for p in open_positions if p.get("city") == city)
    if city_count >= params.max_positions_per_city:
        raise CityLimitExceeded(
            f"Already have {city_count} open position(s) in {city} "
            f"(limit={params.max_positions_per_city})."
        )


# --------------------------------------------------------------------------- #
# Guard 3: Daily Loss Limit
# --------------------------------------------------------------------------- #

_DEFAULT_MAX_DAILY_LOSS_PCT = 0.05   # 5% of bankroll

def check_daily_loss_limit(
    conn: sqlite3.Connection,
    open_positions: list[dict[str, Any]],
    bankroll: float,
    max_loss_pct: float = _DEFAULT_MAX_DAILY_LOSS_PCT,
) -> None:
    """
    Raise DailyLossHalt if today's realised + unrealised loss exceeds the limit.

    Args:
        conn:            SQLite connection (reads daily_pnl).
        open_positions:  Open positions for unrealised mark-to-market.
        bankroll:        Current bankroll.
        max_loss_pct:    Max fraction of bankroll that can be lost today.

    Raises:
        DailyLossHalt
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE date=?", (today,)
    ).fetchone()

    realized_loss = -float(row["realized_pnl"]) if row and row["realized_pnl"] < 0 else 0.0

    # Mark-to-market on open positions
    unrealized_loss = 0.0
    for pos in open_positions:
        entry = pos.get("entry_price", 0.0)
        current = pos.get("current_price", entry)
        size = pos.get("size_usd", 0.0)
        side = pos.get("side", "YES")
        if side == "YES":
            mtm = (current - entry) * size
        else:
            mtm = (entry - current) * size
        if mtm < 0:
            unrealized_loss += abs(mtm)

    total_loss = realized_loss + unrealized_loss
    limit_usd = max_loss_pct * bankroll

    if total_loss >= limit_usd:
        raise DailyLossHalt(
            f"Daily loss limit reached: realised=${realized_loss:.2f} + "
            f"unrealised=${unrealized_loss:.2f} = ${total_loss:.2f} "
            f">= limit ${limit_usd:.2f} ({max_loss_pct*100:.0f}% of ${bankroll:.2f}). "
            f"Trading halted for the day."
        )


def get_daily_loss(
    conn: sqlite3.Connection,
    open_positions: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Return current daily loss breakdown without raising.
    Returns {realized_loss, unrealized_loss, total_loss}.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE date=?", (today,)
    ).fetchone()
    realized_loss = -float(row["realized_pnl"]) if row and row["realized_pnl"] < 0 else 0.0

    unrealized_loss = 0.0
    for pos in open_positions:
        entry = pos.get("entry_price", 0.0)
        current = pos.get("current_price", entry)
        size = pos.get("size_usd", 0.0)
        side = pos.get("side", "YES")
        mtm = (current - entry) * size if side == "YES" else (entry - current) * size
        if mtm < 0:
            unrealized_loss += abs(mtm)

    return {
        "realized_loss": realized_loss,
        "unrealized_loss": unrealized_loss,
        "total_loss": realized_loss + unrealized_loss,
    }
