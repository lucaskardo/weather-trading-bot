"""
Phase 1.3 — Disagreement Shadow Strategy.

Exploits situations where models disagree heavily on temperature BUT one
model has historically been more accurate for this city/season.

Activation requires >= 20 resolved trades per model/city combo.
is_live = False always (shadow only until manually promoted).
"""

from __future__ import annotations

import statistics
from typing import Any

from shared.params import Params, PARAMS
from shared.types import ModelForecast
from strategies.base import BaseStrategy, Signal
from strategies.value_entry import _build_consensus, _prob_above_threshold

_MIN_TRADES_FOR_ACTIVATION = 20
_MIN_SPREAD_F = 4.0      # minimum std-dev across models to consider "disagreement"


class DisagreementStrategy(BaseStrategy):
    """
    Shadow strategy: bet on the historically-better model when models
    disagree widely on temperature.

    Requires a model_accuracy dict (city → {model_name → brier_score})
    to weight the best model. Until 20+ trades per combo, never signals.
    """

    name = "disagreement"
    is_live = False
    version = "1.0.0"

    def generate_signals(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
        conn: Any = None,
        model_accuracy: dict[str, dict[str, float]] | None = None,
        trade_counts: dict[str, dict[str, int]] | None = None,
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

            _, agreement, model_highs, n_models = _build_consensus(
                forecasts, city, target_date
            )
            if agreement is None or agreement < _MIN_SPREAD_F or n_models < 2:
                continue  # models agree — not a disagreement opportunity

            # Check if we have enough trade history to trust the best model
            best_model = _find_best_model(
                forecasts, city, target_date,
                model_accuracy or {},
                trade_counts or {},
            )
            if best_model is None:
                continue  # not enough history

            best_forecast = next(
                (f for f in forecasts
                 if f.city == city and f.target_date == target_date
                 and f.model_name == best_model),
                None,
            )
            if best_forecast is None:
                continue

            fair_value = _prob_above_threshold(
                best_forecast.predicted_high_f, high_f, params.base_std_f
            )
            side = "YES" if fair_value > market_price else "NO"
            raw_edge = (fair_value - market_price) if side == "YES" else (
                (1 - fair_value) - (1 - market_price)
            )

            if abs(raw_edge) < params.min_executable_edge:
                continue

            confidence = min(0.8, agreement / 20.0)  # cap at 0.8 (never fully confident)

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
                executable_price=market_price,
                edge=raw_edge,
                executable_edge=raw_edge,
                confidence=confidence,
                consensus_f=best_forecast.predicted_high_f,
                agreement=agreement,
                n_models=n_models,
                model_temps_f=model_highs,
                side=side,
                subtitle=f"disagreement spread={agreement:.1f}°F best={best_model}",
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


def _find_best_model(
    forecasts: list[ModelForecast],
    city: str,
    target_date: str,
    model_accuracy: dict[str, dict[str, float]],
    trade_counts: dict[str, dict[str, int]],
) -> str | None:
    """
    Return the model name with the lowest Brier score for this city,
    but only if it has >= MIN_TRADES_FOR_ACTIVATION resolved trades.
    """
    city_counts = trade_counts.get(city, {})
    city_accuracy = model_accuracy.get(city, {})

    available_models = {
        f.model_name for f in forecasts
        if f.city == city and f.target_date == target_date
    }

    eligible = {
        model: city_accuracy[model]
        for model in available_models
        if city_counts.get(model, 0) >= _MIN_TRADES_FOR_ACTIVATION
        and model in city_accuracy
    }

    if not eligible:
        return None

    return min(eligible, key=lambda m: eligible[m])  # lowest Brier = best
