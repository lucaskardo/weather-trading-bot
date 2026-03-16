"""
Phase 4.1 — Risk Guards.

Hard safety checks that can HALT trading or reject individual trades.
All guards raise typed exceptions so callers can handle them explicitly.

Guards:
  check_stale_forecast()    — HALT if best available forecast is too old
  check_cluster_exposure()  — REJECT trade if cluster cap would be breached
  check_daily_loss_limit()  — HALT if today's loss exceeds the daily limit
  check_portfolio_var_limit() — REJECT trade if proxy VaR95 would be breached
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


class PortfolioVaRExceeded(RiskViolation):
    """Raised when proxy portfolio VaR95 would be breached."""


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




def check_city_exposure(
    city: str,
    proposed_size_usd: float,
    open_positions: list[dict[str, Any]],
    bankroll: float,
    params: Params = PARAMS,
) -> None:
    """Raise if gross city exposure would breach max_city_exposure_pct."""
    current_exposure = sum(float(p.get("size_usd", 0.0) or 0.0) for p in open_positions if p.get("city") == city)
    cap_usd = params.max_city_exposure_pct * bankroll
    if current_exposure + proposed_size_usd > cap_usd:
        raise CityLimitExceeded(
            f"City '{city}' gross exposure would be ${current_exposure + proposed_size_usd:.2f} "
            f"(cap=${cap_usd:.2f}, bankroll=${bankroll:.2f}). Trade rejected."
        )


def _market_temp_sign(market_type: str) -> float:
    m = (market_type or "above").lower()
    if m == "below":
        return -1.0
    if m == "band":
        return 0.5
    return 1.0


def _position_temp_delta(position: dict[str, Any]) -> float:
    """Proxy PnL sensitivity to a 1-sigma temperature shock in USD.

    This is intentionally conservative and lightweight rather than a full derivative model.
    """
    size_usd = float(position.get("size_usd", 0.0) or 0.0)
    side_sign = 1.0 if (position.get("side") or "YES").upper() == "YES" else -1.0
    market_sign = _market_temp_sign(position.get("market_type") or "above")
    # Scale notional to a proxy daily shock; avoids treating full notional as 1F delta.
    return 0.20 * size_usd * side_sign * market_sign


def _pairwise_temp_corr(a: dict[str, Any], b: dict[str, Any], params: Params = PARAMS) -> float:
    city_a = a.get("city", "")
    city_b = b.get("city", "")
    if city_a and city_a == city_b:
        return params.same_city_corr
    cluster_a = get_cluster(city_a, params)
    cluster_b = get_cluster(city_b, params)
    if cluster_a and cluster_a == cluster_b:
        return params.same_cluster_corr
    return params.cross_cluster_corr


def estimate_portfolio_var95(
    open_positions: list[dict[str, Any]],
    bankroll: float,
    proposed_position: Optional[dict[str, Any]] = None,
    params: Params = PARAMS,
) -> float:
    """Estimate a lightweight 95% one-day VaR in USD using proxy payoff deltas and city correlations."""
    items = [dict(p) for p in open_positions]
    if proposed_position is not None:
        items.append(dict(proposed_position))
    if not items or bankroll <= 0:
        return 0.0

    deltas = [_position_temp_delta(p) for p in items]
    variance = 0.0
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            corr = 1.0 if i == j else _pairwise_temp_corr(a, b, params)
            variance += deltas[i] * deltas[j] * corr
    variance = max(0.0, variance)
    return 1.65 * (variance ** 0.5)


def check_portfolio_var_limit(
    open_positions: list[dict[str, Any]],
    bankroll: float,
    proposed_position: Optional[dict[str, Any]] = None,
    params: Params = PARAMS,
) -> float:
    """Raise PortfolioVaRExceeded if proxy portfolio VaR95 exceeds configured bankroll fraction."""
    var95_usd = estimate_portfolio_var95(open_positions, bankroll, proposed_position, params)
    cap_usd = params.max_portfolio_var95_pct * bankroll
    if var95_usd > cap_usd:
        raise PortfolioVaRExceeded(
            f"Proxy portfolio VaR95 would be ${var95_usd:.2f} "
            f"(cap=${cap_usd:.2f}, bankroll=${bankroll:.2f}). Trade rejected."
        )
    return var95_usd

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
    import os as _os
    daily_limit_env = _os.environ.get("DAILY_LOSS_LIMIT")
    if daily_limit_env is not None:
        limit_usd = float(daily_limit_env)
    else:
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
