"""
Phase 1.4 — Position Lifecycle Engine.

State machine for position management:

  OPENED → HOLDING → EXITED_CONVERGENCE
                   → EXITED_STOP
                   → EXITED_PRE_SETTLEMENT
                   → WON
                   → LOST
         ↘ TAKE_PROFIT_PARTIAL → HOLDING

Per-cycle: check market price, get updated forecast, compute current edge,
apply exit rules, update position status, execute if exiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from shared.params import Params, PARAMS
from shared.types import ModelForecast


class PositionStatus(str, Enum):
    OPENED = "OPENED"
    HOLDING = "HOLDING"
    TAKE_PROFIT_PARTIAL = "TAKE_PROFIT_PARTIAL"
    EXITED_CONVERGENCE = "EXITED_CONVERGENCE"
    EXITED_STOP = "EXITED_STOP"
    EXITED_PRE_SETTLEMENT = "EXITED_PRE_SETTLEMENT"
    WON = "WON"
    LOST = "LOST"


# Terminal states — no further transitions possible
TERMINAL_STATES = {
    PositionStatus.EXITED_CONVERGENCE,
    PositionStatus.EXITED_STOP,
    PositionStatus.EXITED_PRE_SETTLEMENT,
    PositionStatus.WON,
    PositionStatus.LOST,
}

# Valid state transitions
TRANSITIONS: dict[PositionStatus, set[PositionStatus]] = {
    PositionStatus.OPENED: {
        PositionStatus.HOLDING,
        PositionStatus.EXITED_STOP,
    },
    PositionStatus.HOLDING: {
        PositionStatus.HOLDING,
        PositionStatus.TAKE_PROFIT_PARTIAL,
        PositionStatus.EXITED_CONVERGENCE,
        PositionStatus.EXITED_STOP,
        PositionStatus.EXITED_PRE_SETTLEMENT,
        PositionStatus.WON,
        PositionStatus.LOST,
    },
    PositionStatus.TAKE_PROFIT_PARTIAL: {
        PositionStatus.HOLDING,
        PositionStatus.EXITED_CONVERGENCE,
        PositionStatus.EXITED_STOP,
        PositionStatus.EXITED_PRE_SETTLEMENT,
        PositionStatus.WON,
        PositionStatus.LOST,
    },
}


@dataclass
class LifecycleAction:
    """An action produced by the lifecycle engine for a single position."""
    position_id: int
    current_status: PositionStatus
    next_status: PositionStatus
    reason: str
    should_execute: bool          # True = send exit order to exchange
    exit_price: Optional[float]   # price to exit at (None = market order)
    updated_at: str


def process_position(
    position: dict[str, Any],
    forecasts: list[ModelForecast],
    settlement_result: Optional[dict[str, Any]] = None,
    params: Params = PARAMS,
) -> LifecycleAction:
    """
    Evaluate a single position and return the appropriate lifecycle action.

    Args:
        position:          Position dict from DB (must have id, status, city,
                           target_date, side, entry_price, current_price, high_f,
                           opened_at).
        forecasts:         Current model forecasts.
        settlement_result: If contract has settled, pass {"won": bool,
                           "actual_high_f": float}. None if still open.
        params:            Trading parameters.

    Returns:
        LifecycleAction describing the transition.
    """
    pos_id = position["id"]
    status = PositionStatus(position.get("status", PositionStatus.OPENED))
    now_str = datetime.now(timezone.utc).isoformat()

    # Already terminal — no further action
    if status in TERMINAL_STATES:
        return LifecycleAction(
            position_id=pos_id,
            current_status=status,
            next_status=status,
            reason="already_terminal",
            should_execute=False,
            exit_price=None,
            updated_at=now_str,
        )

    # --- Settlement check (takes priority) ---
    if settlement_result is not None:
        won = settlement_result.get("won", False)
        next_status = PositionStatus.WON if won else PositionStatus.LOST
        return LifecycleAction(
            position_id=pos_id,
            current_status=status,
            next_status=next_status,
            reason="settlement",
            should_execute=False,
            exit_price=None,
            updated_at=now_str,
        )

    city = position.get("city", "")
    target_date = position.get("target_date", "")
    side = position.get("side", "YES")
    entry_price = position.get("entry_price", 0.5)
    current_price = position.get("current_price", entry_price)
    high_f = position.get("high_f")
    low_f = position.get("low_f")
    market_type = position.get("market_type", "above")

    # Promote OPENED → HOLDING on first cycle
    if status == PositionStatus.OPENED:
        return LifecycleAction(
            position_id=pos_id,
            current_status=status,
            next_status=PositionStatus.HOLDING,
            reason="position_opened",
            should_execute=False,
            exit_price=None,
            updated_at=now_str,
        )

    # --- Compute updated edge from latest forecasts ---
    updated_fair_value = _compute_fair_value(
        forecasts, city, target_date, high_f, params, market_type, low_f
    )

    # --- Exit Rule 1: Stale forecast (check before trusting forecast values) ---
    if _is_forecast_stale(forecasts, city, target_date, params.stale_forecast_hours):
        return LifecycleAction(
            position_id=pos_id,
            current_status=status,
            next_status=PositionStatus.EXITED_STOP,
            reason="stale_forecast",
            should_execute=True,
            exit_price=current_price,
            updated_at=now_str,
        )

    # --- Exit Rule 2: Forecast reversal (stop loss) ---
    if updated_fair_value is not None:
        if side == "YES":
            updated_edge = updated_fair_value - current_price
        else:
            updated_edge = (1 - updated_fair_value) - (1 - current_price)
        if updated_edge < -0.05:
            return LifecycleAction(
                position_id=pos_id,
                current_status=status,
                next_status=PositionStatus.EXITED_STOP,
                reason="forecast_reversal",
                should_execute=True,
                exit_price=current_price,
                updated_at=now_str,
            )

    # --- Exit Rule 3: Convergence ---
    if updated_fair_value is not None:
        if abs(current_price - updated_fair_value) < 0.03:
            return LifecycleAction(
                position_id=pos_id,
                current_status=status,
                next_status=PositionStatus.EXITED_CONVERGENCE,
                reason="market_converged",
                should_execute=True,
                exit_price=current_price,
                updated_at=now_str,
            )

    # --- Exit Rule 4: Pre-settlement thin edge ---
    hours_left = _hours_to_settlement(target_date)
    if hours_left is not None and hours_left < 4:
        unrealized = (current_price - entry_price) if side == "YES" else (entry_price - current_price)
        if unrealized < 0.03:
            return LifecycleAction(
                position_id=pos_id,
                current_status=status,
                next_status=PositionStatus.EXITED_PRE_SETTLEMENT,
                reason="pre_settlement_thin_edge",
                should_execute=True,
                exit_price=current_price,
                updated_at=now_str,
            )

    # --- Hold ---
    return LifecycleAction(
        position_id=pos_id,
        current_status=status,
        next_status=PositionStatus.HOLDING,
        reason="holding",
        should_execute=False,
        exit_price=None,
        updated_at=now_str,
    )


def run_lifecycle_cycle(
    open_positions: list[dict[str, Any]],
    forecasts: list[ModelForecast],
    settlement_results: Optional[dict[int, dict[str, Any]]] = None,
    params: Params = PARAMS,
) -> list[LifecycleAction]:
    """
    Process all open positions in one cycle.

    Args:
        open_positions:    All non-terminal positions from DB.
        forecasts:         Latest model forecasts.
        settlement_results: Map of position_id → settlement result dict.
        params:            Trading parameters.

    Returns:
        List of LifecycleAction, one per position.
    """
    results: list[LifecycleAction] = []
    sr = settlement_results or {}

    for pos in open_positions:
        settlement = sr.get(pos.get("id"))
        action = process_position(pos, forecasts, settlement, params)
        results.append(action)

    return results


def is_valid_transition(from_status: PositionStatus, to_status: PositionStatus) -> bool:
    """Return True if the state transition is allowed."""
    if from_status == to_status:
        return True  # self-transition (hold) always valid
    allowed = TRANSITIONS.get(from_status, set())
    return to_status in allowed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _compute_fair_value(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    high_f: Optional[float],
    params: Params,
    market_type: str = "above",
    low_f: Optional[float] = None,
) -> Optional[float]:
    from core.forecaster import compute_fair_value
    fair_value, _, _, _ = compute_fair_value(
        forecasts, city, target_date, market_type, high_f, low_f, params,
        use_mc=False,
    )
    return fair_value


def _is_forecast_stale(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    max_hours: float,
) -> bool:
    relevant = [
        f for f in forecasts
        if f.city == city and f.target_date == target_date and f.fetched_at
    ]
    if not relevant:
        return True
    now = datetime.now(timezone.utc)
    for f in relevant:
        try:
            fetched = datetime.fromisoformat(f.fetched_at)
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if (now - fetched).total_seconds() / 3600 <= max_hours:
                return False
        except ValueError:
            continue
    return True


def _hours_to_settlement(target_date: str) -> Optional[float]:
    try:
        eod = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
        return (eod - datetime.now(timezone.utc)).total_seconds() / 3600
    except ValueError:
        return None
