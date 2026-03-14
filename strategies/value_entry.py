"""
Phase 1.2 — ValueEntry Strategy.

The primary live strategy: enters when executable_edge >= threshold and
manages positions with 4 exit rules.

Exit rules:
  1. Forecast reversal  — updated forecast flips edge below -0.05
  2. Convergence        — market price within 3% of fair value
  3. Stale forecast     — no new model run within stale_forecast_hours
  4. Pre-settlement     — < 4 hours to settlement with thin edge (< 0.03)
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

from core.forecaster import (
    dynamic_std_f,
    compute_fair_value,
    _compute_fair_value_for_market,  # re-exported for backward compatibility
)
from execution.orderbook import OrderbookLevel, get_executable_price
from shared.params import Params, PARAMS
from shared.types import ModelForecast
from strategies.base import BaseStrategy, Signal


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat()


def _build_consensus(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    bias_table: dict | None = None,
    fine_bias_table: dict | None = None,
    lead_hours: float = 48.0,
):
    """Return (consensus_f, agreement_std, model_highs, n_models) for a city/date."""
    relevant = [
        f for f in forecasts
        if f.city == city and f.target_date == target_date
    ]
    if not relevant:
        return None, None, [], 0

    if bias_table:
        from research.bias_correction import apply_bias
        highs = [
            apply_bias(
                f.predicted_high_f, city, f.model_name, bias_table,
                fine_bias_table=fine_bias_table,
                target_date=target_date,
                lead_hours=lead_hours,
            )
            for f in relevant
        ]
    else:
        highs = [f.predicted_high_f for f in relevant]

    consensus = statistics.mean(highs)
    agreement = statistics.stdev(highs) if len(highs) > 1 else 0.0
    return consensus, agreement, highs, len(highs)


def _prob_above_threshold(consensus_f: float, threshold_f: float, std_f: float) -> float:
    """
    P(actual_high > threshold) under a Gaussian with mean=consensus, std=std_f.
    Uses the complementary error function (erfc).
    """
    if std_f <= 0:
        return 1.0 if consensus_f > threshold_f else 0.0
    z = (threshold_f - consensus_f) / (std_f * math.sqrt(2))
    return 0.5 * math.erfc(z)



class ValueEntryStrategy(BaseStrategy):
    """
    Enter positions where executable edge >= min_executable_edge.
    Exit on reversal, convergence, stale data, or pre-settlement.
    """

    name = "value_entry"
    is_live = True
    version = "1.0.0"

    def generate_signals(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
        conn: Any = None,
    ) -> list[Signal]:
        # Load bias tables once per cycle if DB connection available
        bias_table: dict | None = None
        fine_bias_table: dict | None = None
        if conn is not None:
            from research.bias_correction import learn_biases, learn_fine_biases
            bias_table = learn_biases(conn)
            fine_bias_table = learn_fine_biases(conn)

        signals: list[Signal] = []

        for market in markets:
            ticker = market.get("ticker", "")
            city = market.get("city", "")
            target_date = market.get("target_date", "")
            market_price = market.get("market_price", 0.5)
            high_f = market.get("high_f")
            low_f = market.get("low_f")

            if not (ticker and city and target_date and high_f is not None):
                continue

            consensus_f, agreement, model_highs, n_models = _build_consensus(
                forecasts, city, target_date,
                bias_table=bias_table,
                fine_bias_table=fine_bias_table,
                lead_hours=_hours_to_settlement(target_date) or 48.0,
            )
            if consensus_f is None or n_models == 0:
                continue

            # Dynamic uncertainty: adjust std_f for model spread + lead time
            lead_hours = _hours_to_settlement(target_date) or 48.0
            std_f = dynamic_std_f(
                city=city,
                target_date=target_date,
                model_spread=agreement or 0.0,
                lead_hours=lead_hours,
                base_std_f=params.base_std_f,
            )
            # Correct probability per market type (band / below / above)
            market_type = market.get("market_type", "above")
            if getattr(params, "use_monte_carlo", True) and model_highs:
                from core.forecaster import monte_carlo_prob
                fair_value = monte_carlo_prob(
                    model_forecasts=model_highs,
                    market_type=market_type,
                    high_f=high_f,
                    low_f=low_f,
                    sigma=std_f,
                    n_samples=getattr(params, "monte_carlo_samples", 2000),
                )
            else:
                fair_value = _compute_fair_value_for_market(
                    consensus_f, market_type, high_f, low_f, std_f
                )
                if fair_value is None:
                    continue

            # Apply temperature scaling
            T = params.temp_scaling_T
            if T != 1.0 and 0 < fair_value < 1:
                logit = math.log(fair_value / (1 - fair_value))
                fair_value = 1 / (1 + math.exp(-logit / T))

            # Determine side
            side = "YES" if fair_value > market_price else "NO"
            if side == "NO":
                # For NO, effective model prob is (1 - fair_value) vs (1 - market_price)
                eff_model_prob = 1.0 - fair_value
                eff_market_price = 1.0 - market_price
            else:
                eff_model_prob = fair_value
                eff_market_price = market_price

            raw_edge = eff_model_prob - eff_market_price

            # Executable price
            raw_ob = market.get("orderbook")
            orderbook: Optional[list[OrderbookLevel]] = None
            if raw_ob:
                orderbook = [
                    OrderbookLevel(price=lv["price"], size_usd=lv["size_usd"])
                    for lv in raw_ob
                ]

            exec_info = get_executable_price(
                ticker=ticker,
                side=side,
                target_size_usd=market.get("target_size_usd", 100.0),
                fallback_price=eff_market_price,
                orderbook=orderbook,
                params=params,
            )

            executable_edge = eff_model_prob - exec_info.executable_price

            # Time-decaying edge threshold: require less edge near settlement
            # Cap lead_hours at 72h so distant contracts get max (not runaway) threshold
            alpha = getattr(params, "edge_decay_alpha", 0.002)
            beta = getattr(params, "edge_decay_beta", 0.05)
            capped_hours = min(lead_hours, 72.0)
            dynamic_min_edge = params.min_executable_edge + alpha * math.exp(beta * capped_hours)
            if executable_edge < dynamic_min_edge:
                continue
            if not exec_info.is_liquid:
                continue

            confidence = max(0.0, min(1.0, 1.0 - (agreement or 0) / 10.0))

            sig = self._make_signal(
                market_id=market.get("id", ticker),
                ticker=ticker,
                source=market.get("exchange", "kalshi"),
                city=city,
                target_date=target_date,
                market_type=market.get("market_type", "above"),
                low_f=low_f,
                high_f=high_f,
                market_price=market_price,
                fair_value=fair_value,
                executable_price=exec_info.executable_price,
                edge=raw_edge,
                executable_edge=executable_edge,
                # Side-aware effective values for correct Kelly sizing
                effective_prob=eff_model_prob,
                effective_price=exec_info.executable_price,
                effective_edge=executable_edge,
                confidence=confidence,
                consensus_f=consensus_f,
                agreement=agreement,
                n_models=n_models,
                model_temps_f=model_highs,
                side=side,
                subtitle=f"consensus={consensus_f:.1f}°F, edge={executable_edge:.3f}",
            )
            signals.append(sig)

        # Sort by executable_edge × confidence descending
        signals.sort(key=lambda s: s.executable_edge * s.confidence, reverse=True)
        return signals

    def manage_positions(
        self,
        open_positions: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []

        for pos in open_positions:
            action = self._check_exit(pos, forecasts, params)
            actions.append(action)

        return actions

    def _check_exit(
        self,
        pos: dict[str, Any],
        forecasts: list[ModelForecast],
        params: Params,
    ) -> dict[str, Any]:
        position_id = pos.get("id")
        city = pos.get("city", "")
        target_date = pos.get("target_date", "")
        entry_price = pos.get("entry_price", 0.5)
        current_price = pos.get("current_price", entry_price)
        side = pos.get("side", "YES")
        high_f = pos.get("high_f")
        low_f = pos.get("low_f")
        market_type = pos.get("market_type", "above")
        opened_at_str = pos.get("opened_at", "")

        # Compute fair value using the canonical pipeline (analytic, no MC noise)
        fair_value, consensus_f, _, n_models = compute_fair_value(
            forecasts, city, target_date, market_type, high_f, low_f, params,
            use_mc=False,
        )

        # --- Rule 1: Forecast reversal ---
        if fair_value is not None and n_models > 0:
            if side == "YES":
                updated_edge = fair_value - current_price
            else:
                updated_edge = (1 - fair_value) - (1 - current_price)
            if updated_edge < -0.05:
                return {"position_id": position_id, "action": "exit", "reason": "forecast_reversal"}

        # --- Rule 2: Convergence ---
        if fair_value is not None and abs(current_price - fair_value) < 0.03:
            return {"position_id": position_id, "action": "exit", "reason": "convergence"}

        # --- Rule 3: Stale forecast ---
        stale = _is_forecast_stale(forecasts, city, target_date, params.stale_forecast_hours)
        if stale:
            return {"position_id": position_id, "action": "exit", "reason": "stale_forecast"}

        # --- Rule 4: Pre-settlement thin edge ---
        hours_to_settlement = _hours_to_settlement(target_date)
        if hours_to_settlement is not None and hours_to_settlement < 4:
            if side == "YES":
                thin_edge = current_price - entry_price
            else:
                thin_edge = entry_price - current_price
            if thin_edge < 0.03:
                return {"position_id": position_id, "action": "exit", "reason": "pre_settlement"}

        return {"position_id": position_id, "action": "hold", "reason": ""}

    def evaluate(self, recent_trades: list[dict[str, Any]]) -> dict[str, Any]:
        if not recent_trades:
            return {"sharpe": 0.0, "brier": 0.5, "win_rate": 0.0, "trade_count": 0}

        pnls = [t.get("realized_pnl", 0.0) for t in recent_trades]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)

        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1e-9
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        brier_scores = [t.get("brier_score") for t in recent_trades if t.get("brier_score") is not None]
        avg_brier = statistics.mean(brier_scores) if brier_scores else 0.25

        return {
            "sharpe": sharpe,
            "brier": avg_brier,
            "win_rate": win_rate,
            "trade_count": len(recent_trades),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _is_forecast_stale(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    max_hours: float,
) -> bool:
    """Return True if the newest forecast for city/date is older than max_hours."""
    relevant = [
        f for f in forecasts
        if f.city == city and f.target_date == target_date and f.fetched_at
    ]
    if not relevant:
        return True  # no forecast at all = stale

    now = _now_utc()
    for f in relevant:
        try:
            fetched = datetime.fromisoformat(f.fetched_at)
            if fetched.tzinfo is None:
                from datetime import timezone as tz
                fetched = fetched.replace(tzinfo=tz.utc)
            age_hours = (now - fetched).total_seconds() / 3600
            if age_hours <= max_hours:
                return False
        except ValueError:
            continue
    return True


def _hours_to_settlement(target_date: str) -> Optional[float]:
    """
    Estimate hours until settlement (EOD of target_date in UTC).
    Returns None if target_date is unparseable.
    """
    try:
        eod = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
        delta = (eod - _now_utc()).total_seconds() / 3600
        return delta
    except ValueError:
        return None
