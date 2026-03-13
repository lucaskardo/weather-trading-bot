"""
Phase 1.1 — Base Strategy Interface.

All strategies inherit from BaseStrategy and produce Signal objects.
Strategies are isolated: they receive market data and forecasts, return
signals and position management actions, and never touch the DB directly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Signal:
    """A single tradeable signal produced by a strategy."""

    # --- Identity ---
    strategy_name: str
    market_id: str
    ticker: str
    source: str        # exchange, e.g. "kalshi" | "polymarket"
    city: str
    target_date: str   # ISO date "YYYY-MM-DD"

    # --- Market ---
    market_type: str = "high_temp"
    low_f: Optional[float] = None
    high_f: Optional[float] = None
    market_price: float = 0.5

    # --- Prediction ---
    fair_value: float = 0.5
    executable_price: float = 0.5
    edge: float = 0.0              # raw: fair_value - market_price
    executable_edge: float = 0.0   # after fees/slippage
    confidence: float = 0.0

    # --- Context ---
    consensus_f: Optional[float] = None
    agreement: Optional[float] = None   # std-dev of model temps (lower = better)
    n_models: int = 0
    model_temps_f: list[float] = field(default_factory=list)
    model_confidences: list[float] = field(default_factory=list)

    # --- Metadata ---
    created_at: str = ""
    subtitle: str = ""
    is_shadow: bool = False    # shadow signals log only — never execute
    side: str = "YES"          # "YES" | "NO"


class BaseStrategy(abc.ABC):
    """
    Abstract base for all strategies.

    Strategies are stateless computation units.  All state lives in SQLite;
    strategies receive it as arguments and return actions as plain dicts.
    """

    # Subclasses set these at class level
    name: str = "base"
    is_live: bool = False       # False = shadow mode (log only, no execution)
    version: str = "0.1.0"

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def generate_signals(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[Any],         # list[ModelForecast]
        params: Any,                  # Params
    ) -> list[Signal]:
        """
        Scan markets + forecasts and return candidate signals.

        Implementations must:
        - Use executable_edge (not raw edge) for filtering
        - Populate all Signal fields
        - Respect params.min_executable_edge threshold
        """
        ...

    @abc.abstractmethod
    def manage_positions(
        self,
        open_positions: list[dict[str, Any]],
        forecasts: list[Any],
        params: Any,
    ) -> list[dict[str, Any]]:
        """
        Review open positions and return exit/update actions.

        Each action dict must include:
          {"position_id": int, "action": "exit"|"hold", "reason": str}
        """
        ...

    @abc.abstractmethod
    def evaluate(self, recent_trades: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute performance metrics from recent resolved trades.

        Must return at minimum:
          {"sharpe": float, "brier": float, "win_rate": float, "trade_count": int}
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _stamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _make_signal(self, **kwargs) -> Signal:
        """Convenience: create a Signal with strategy_name + is_shadow pre-filled."""
        kwargs.setdefault("strategy_name", self.name)
        kwargs.setdefault("is_shadow", not self.is_live)
        kwargs.setdefault("created_at", self._stamp())
        return Signal(**kwargs)
