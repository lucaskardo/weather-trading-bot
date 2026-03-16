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
_MIN_CONFIRMED_MOMENTUM_F = 1.0
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
        conn: Any = None,
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

            delta, new_consensus, old_run_ids, new_run_ids, revision_confirmed, publish_time, fetch_time = _compute_run_delta(
                forecasts, city, target_date
            )
            if delta is None or abs(delta) < _MIN_DELTA_F or not revision_confirmed:
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
                subtitle=f"model_release delta={delta:+.1f}°F confirmed",
                is_shadow=True,
                provider_publish_time=publish_time,
                model_run_time=new_run_ids[0] if new_run_ids else None,
                bot_fetch_time=fetch_time,
                parse_to_signal_time=self._stamp(),
                revision_confirmed=revision_confirmed,
                revision_delta_f=delta,
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
) -> tuple[float | None, float, list[str], list[str], bool, str | None, str | None]:
    """
    Compare the most-recent run vs the previous run for watched models.

    Returns:
      (delta_f, new_consensus_f, old_run_ids, new_run_ids, revision_confirmed,
       provider_publish_time, bot_fetch_time)

    revision_confirmed requires the latest run shift to align with the recent
    run-to-run momentum, reducing false positives from noisy single releases.
    """
    relevant = [
        f for f in forecasts
        if f.city == city
        and f.target_date == target_date
        and f.model_name in _WATCHED_MODELS
        and f.run_id
    ]
    if len(relevant) < 2:
        return None, 0.0, [], [], False, None, None

    relevant.sort(key=lambda f: f.run_id, reverse=True)
    run_ids = list(dict.fromkeys(f.run_id for f in relevant))
    newest_run_id = run_ids[0]
    new_run = [f for f in relevant if f.run_id == newest_run_id]
    old_run = [f for f in relevant if f.run_id != newest_run_id]
    if not old_run:
        return None, 0.0, [], [], False, None, None

    old_run_id = old_run[0].run_id
    old_run_forecasts = [f for f in old_run if f.run_id == old_run_id]

    new_consensus = statistics.mean(f.predicted_high_f for f in new_run)
    old_consensus = statistics.mean(f.predicted_high_f for f in old_run_forecasts)
    delta = new_consensus - old_consensus

    revision_confirmed = False
    if len(run_ids) >= 3:
        prior_run_id = run_ids[2]
        prior_run = [f for f in relevant if f.run_id == prior_run_id]
        if prior_run:
            prior_consensus = statistics.mean(f.predicted_high_f for f in prior_run)
            prev_delta = old_consensus - prior_consensus
            revision_confirmed = (abs(prev_delta) >= _MIN_CONFIRMED_MOMENTUM_F and (delta * prev_delta) > 0)
    else:
        # With only two runs, require both watched models to move in same direction.
        per_model = {}
        for nf in new_run:
            of = next((f for f in old_run_forecasts if f.model_name == nf.model_name), None)
            if of is not None:
                per_model[nf.model_name] = nf.predicted_high_f - of.predicted_high_f
        if per_model:
            signs = {1 if d > 0 else -1 for d in per_model.values() if abs(d) >= 0.5}
            revision_confirmed = len(signs) == 1 and len(per_model) >= 1

    new_ids = list({f.run_id for f in new_run})
    old_ids = list({f.run_id for f in old_run_forecasts})
    provider_publish_time = max((f.publish_time for f in new_run if f.publish_time), default=None)
    bot_fetch_time = max((f.fetched_at for f in new_run if f.fetched_at), default=None)

    return delta, new_consensus, old_ids, new_ids, revision_confirmed, provider_publish_time, bot_fetch_time
