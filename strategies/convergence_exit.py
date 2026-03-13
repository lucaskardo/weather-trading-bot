"""
Phase 1.3 — ConvergenceExit Shadow Strategy.

Monitors ALL open positions (regardless of originating strategy).
Generates exit signals when market price is within 2-3¢ of model fair value.

is_live = False: signals are logged with is_shadow=1 but never executed.
Tracks hypothetical PnL to prove/disprove early-exit profitability.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from shared.params import Params, PARAMS
from shared.types import ModelForecast
from strategies.base import BaseStrategy, Signal
from strategies.value_entry import _build_consensus, _prob_above_threshold


_CONVERGENCE_THRESHOLD = 0.03   # 3 cents


class ConvergenceExitStrategy(BaseStrategy):
    """
    Shadow strategy: log exit signal when market converges to fair value.

    Does NOT originate new entry positions.
    """

    name = "convergence_exit"
    is_live = False
    version = "1.0.0"

    def generate_signals(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
    ) -> list[Signal]:
        # Convergence exit doesn't generate entry signals
        return []

    def manage_positions(
        self,
        open_positions: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        params: Params = PARAMS,
    ) -> list[dict[str, Any]]:
        """Flag positions where market has converged within threshold of fair value."""
        actions: list[dict[str, Any]] = []

        for pos in open_positions:
            city = pos.get("city", "")
            target_date = pos.get("target_date", "")
            current_price = pos.get("current_price", pos.get("entry_price", 0.5))
            high_f = pos.get("high_f")

            consensus_f, _, _, n_models = _build_consensus(forecasts, city, target_date)
            if consensus_f is None or n_models == 0 or high_f is None:
                actions.append({"position_id": pos.get("id"), "action": "hold", "reason": ""})
                continue

            fair_value = _prob_above_threshold(consensus_f, high_f, params.base_std_f)
            gap = abs(current_price - fair_value)

            if gap <= _CONVERGENCE_THRESHOLD:
                actions.append({
                    "position_id": pos.get("id"),
                    "action": "exit",
                    "reason": "convergence",
                    "is_shadow": True,
                    "gap": gap,
                    "fair_value": fair_value,
                    "current_price": current_price,
                })
            else:
                actions.append({"position_id": pos.get("id"), "action": "hold", "reason": ""})

        return actions

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
