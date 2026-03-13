"""
Phase 2.4 — Strategy Brain (Orchestrator).

The brain coordinates one full trading cycle:
  1. Generate signals from all strategies
  2. Manage existing positions (check exits)
  3. Score strategies
  4. Allocate capital
  5. Select and size signals
  6. Execute (or dry-run)

Returns a summary dict for logging / monitoring.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from execution.lifecycle import run_lifecycle_cycle, PositionStatus
from shared.params import Params, PARAMS
from shared.types import ModelForecast
from strategies.base import Signal
from strategies.value_entry import ValueEntryStrategy
from strategies.convergence_exit import ConvergenceExitStrategy
from strategies.model_release import ModelReleaseStrategy
from strategies.disagreement import DisagreementStrategy
from strategy_router.scorecard import score_all_strategies
from strategy_router.allocator import allocate
from strategy_router.selector import select_signals

# Registry of all strategy instances
_ALL_STRATEGIES = [
    ValueEntryStrategy(),
    ConvergenceExitStrategy(),
    ModelReleaseStrategy(),
    DisagreementStrategy(),
]


class Brain:
    """
    Orchestrates one full trading cycle.

    In production, `conn` is the live SQLite connection.
    In dry-run mode, execution is skipped and orders are returned for inspection.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        params: Params = PARAMS,
        dry_run: bool = True,
    ):
        self.conn = conn
        self.params = params
        self.dry_run = dry_run
        self.strategies = list(_ALL_STRATEGIES)

    def run_cycle(
        self,
        markets: list[dict[str, Any]],
        forecasts: list[ModelForecast],
        open_positions: Optional[list[dict[str, Any]]] = None,
        settlement_results: Optional[dict[int, dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Execute one full trading cycle.

        Args:
            markets:            Current open markets from exchange.
            forecasts:          Latest model forecasts (all cities/dates).
            open_positions:     Current open positions from DB (or [] if none).
            settlement_results: Map of position_id → {won, actual_high_f}.

        Returns:
            Summary dict with counts and scores.
        """
        positions = open_positions or []
        now = datetime.now(timezone.utc).isoformat()

        # ------------------------------------------------------------------ #
        # Step 1: Generate signals from all strategies
        # ------------------------------------------------------------------ #
        all_signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                sigs = strategy.generate_signals(markets, forecasts, self.params)
                all_signals.extend(sigs)
            except Exception as exc:
                _log(f"[brain] {strategy.name}.generate_signals failed: {exc}")

        # ------------------------------------------------------------------ #
        # Step 2: Manage existing positions
        # ------------------------------------------------------------------ #
        lifecycle_actions = run_lifecycle_cycle(
            positions, forecasts, settlement_results, self.params
        )
        exits = [a for a in lifecycle_actions if a.should_execute]
        _log(f"[brain] {len(exits)} exit(s) triggered this cycle")

        # ------------------------------------------------------------------ #
        # Step 3: Score strategies
        # ------------------------------------------------------------------ #
        strategy_names = [s.name for s in self.strategies]
        scores = score_all_strategies(
            strategy_names, self.conn, self.params, lookback_days=30
        )
        _log(f"[brain] scores: {scores}")

        # ------------------------------------------------------------------ #
        # Step 4: Allocate capital
        # ------------------------------------------------------------------ #
        bankroll = self._get_bankroll()
        allocations = allocate(scores, bankroll, self.params)

        # ------------------------------------------------------------------ #
        # Step 5: Select and size signals
        # ------------------------------------------------------------------ #
        orders = select_signals(
            all_signals, allocations, positions, bankroll, self.params
        )
        executable = [o for o in orders if o.get("reason_skipped") is None]
        shadow_logged = [o for o in orders if o.get("reason_skipped") == "shadow"]
        filtered = [o for o in orders if o.get("reason_skipped") not in (None, "shadow")]

        # ------------------------------------------------------------------ #
        # Step 6: Execute (or dry-run)
        # ------------------------------------------------------------------ #
        executed = 0
        if not self.dry_run:
            for order in executable:
                try:
                    self._execute_order(order)
                    executed += 1
                except Exception as exc:
                    _log(f"[brain] execution failed for {order['signal'].ticker}: {exc}")
        else:
            executed = len(executable)  # count as "would execute"

        return {
            "cycle_at": now,
            "signals_generated": len(all_signals),
            "signals_executable": len(executable),
            "signals_shadow": len(shadow_logged),
            "signals_filtered": len(filtered),
            "exits": len(exits),
            "scores": scores,
            "allocations": allocations,
            "orders": orders,
            "executed": executed,
            "dry_run": self.dry_run,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_bankroll(self) -> float:
        row = self.conn.execute(
            "SELECT bankroll FROM portfolio WHERE id=1"
        ).fetchone()
        return float(row["bankroll"]) if row else 1000.0

    def _execute_order(self, order: dict[str, Any]) -> None:
        """
        Stub: send order to exchange API and record in DB.
        Implemented properly when exchange clients are wired up.
        """
        sig: Signal = order["signal"]
        size_usd: float = order["size_usd"]
        _log(f"[brain] EXECUTE {sig.ticker} {sig.side} ${size_usd:.2f} @ {sig.executable_price:.3f}")
        # TODO: call kalshi_client.place_order(sig.ticker, sig.side, size_usd)
        # TODO: INSERT INTO positions (...)


def _log(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr)
