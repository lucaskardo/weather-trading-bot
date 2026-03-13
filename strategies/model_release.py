"""
Phase 1.3 — ModelRelease Shadow Strategy.

Watches for new GFS/ECMWF model runs (run_id changes).
Signals when the new consensus shifts > 3°F AND market hasn't repriced.

is_live = False: shadow mode only until it accumulates 50+ validated trades.
"""

from __future__ import annotations

import statistics
from typing import Any

from shared.params import Params, PARAMS
from shared.types import ModelForecast
from strategies.base import BaseStrategy, Signal
from strategies.value_entry import _build_consensus, _prob_above_threshold

_MIN_DELTA_F = 3.0       # minimum consensus shift to consider signalling
_WATCHED_MODELS = {"GFS", "ECMWF"}


class ModelReleaseStrategy(BaseStrategy):
    """
    Shadow strategy: enter when a new model run shifts consensus > 3°F
    and the market price hasn't moved to reflect it yet.
    """

    name = "model_release"
    is_live = False
    version = "1.0.0"

    def generate_signals(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
    ) -> list[Signal]:
        signals: list[Signal] = []

        for market in markets:
            city = market.get("city", "")
            target_date = market.get("target_date", "")
            ticker = market.get("ticker", "")
            market_price = market.get("market_price", 0.5)
            high_f = market.get("high_f")

            if not (city and target_date and high_f is not None):
                continue

            delta, new_consensus, old_run_ids, new_run_ids = _compute_run_delta(
                forecasts, city, target_date
            )
            if delta is None or abs(delta) < _MIN_DELTA_F:
                continue

            fair_value = _prob_above_threshold(new_consensus, high_f, params.base_std_f)
            side = "YES" if fair_value > market_price else "NO"
            raw_edge = (fair_value - market_price) if side == "YES" else ((1 - fair_value) - (1 - market_price))

            if abs(raw_edge) < params.min_executable_edge:
                continue

            _, agreement, model_highs, n_models = _build_consensus(
                forecasts, city, target_date
            )
            confidence = max(0.0, min(1.0, abs(delta) / 10.0))  # larger delta → more confident

            sig = self._make_signal(
                market_id=market.get("id", ticker),
                ticker=ticker,
                source=market.get("exchange", "kalshi"),
                city=city,
                target_date=target_date,
                market_type=market.get("market_type", "high_temp"),
                high_f=high_f,
                low_f=market.get("low_f"),
                market_price=market_price,
                fair_value=fair_value,
                executable_price=market_price,  # shadow: no exec cost calc
                edge=raw_edge,
                executable_edge=raw_edge,       # shadow: approximate
                confidence=confidence,
                consensus_f=new_consensus,
                agreement=agreement,
                n_models=n_models,
                model_temps_f=model_highs,
                side=side,
                subtitle=f"model_release delta={delta:+.1f}°F",
                is_shadow=True,
            )
            signals.append(sig)

        return signals

    def manage_positions(
        self,
        open_positions: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
    ) -> list[dict[str, Any]]:
        return [{"position_id": p.get("id"), "action": "hold", "reason": ""} for p in open_positions]

    def evaluate(self, recent_trades: list[dict[str, Any]]) -> dict[str, Any]:
        if not recent_trades:
            return {"sharpe": 0.0, "brier": 0.5, "win_rate": 0.0, "trade_count": 0}
        pnls = [t.get("realized_pnl", 0.0) for t in recent_trades]
        wins = sum(1 for p in pnls if p > 0)
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1e-9
        return {
            "sharpe": mean_pnl / std_pnl if std_pnl > 0 else 0.0,
            "brier": 0.25,
            "win_rate": wins / len(pnls),
            "trade_count": len(recent_trades),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _compute_run_delta(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
) -> tuple[float | None, float, list[str], list[str]]:
    """
    Compare the most-recent run vs the previous run for watched models.

    Returns (delta_f, new_consensus_f, old_run_ids, new_run_ids).
    delta_f = new_consensus - old_consensus; None if < 2 runs available.
    """
    relevant = [
        f for f in forecasts
        if f.city == city
        and f.target_date == target_date
        and f.model_name in _WATCHED_MODELS
        and f.run_id
    ]
    if len(relevant) < 2:
        return None, 0.0, [], []

    # Sort by run_id descending (newest first)
    relevant.sort(key=lambda f: f.run_id, reverse=True)
    newest_run_id = relevant[0].run_id

    new_run = [f for f in relevant if f.run_id == newest_run_id]
    old_run = [f for f in relevant if f.run_id != newest_run_id]

    if not old_run:
        return None, 0.0, [], []

    old_run_id = old_run[0].run_id
    old_run_forecasts = [f for f in old_run if f.run_id == old_run_id]

    new_consensus = statistics.mean(f.predicted_high_f for f in new_run)
    old_consensus = statistics.mean(f.predicted_high_f for f in old_run_forecasts)

    delta = new_consensus - old_consensus
    new_ids = list({f.run_id for f in new_run})
    old_ids = list({f.run_id for f in old_run_forecasts})

    return delta, new_consensus, old_ids, new_ids
