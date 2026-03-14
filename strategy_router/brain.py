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

from execution.exchange_executor import ExchangeExecutor, PaperExecutor
from execution.lifecycle import run_lifecycle_cycle, PositionStatus
from risk.guards import check_daily_loss_limit, DailyLossHalt
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
        executor: ExchangeExecutor | None = None,
    ):
        self.conn = conn
        self.params = params
        self.dry_run = dry_run
        self.executor: ExchangeExecutor = executor or PaperExecutor()
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
        # Step 0: Hard risk checks — halt entire cycle if limit breached
        # ------------------------------------------------------------------ #
        bankroll_check = self._get_bankroll()
        try:
            check_daily_loss_limit(self.conn, positions, bankroll_check)
        except DailyLossHalt as exc:
            _log(f"[brain] HALT: {exc}")
            return {
                "cycle_at": now,
                "signals_generated": 0,
                "signals_executable": 0,
                "signals_shadow": 0,
                "signals_filtered": 0,
                "exits": 0,
                "scores": {},
                "allocations": {},
                "orders": [],
                "executed": 0,
                "dry_run": self.dry_run,
                "halted": True,
                "halt_reason": str(exc),
            }

        # ------------------------------------------------------------------ #
        # Step 1: Generate signals from all strategies
        # ------------------------------------------------------------------ #
        all_signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                sigs = strategy.generate_signals(markets, forecasts, self.params, conn=self.conn)
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
        # Step 2b: Persist exits — update status, compute PnL, return capital
        # ------------------------------------------------------------------ #
        for action in exits:
            self._process_exit(action, positions)

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

    def _process_exit(self, action, positions: list[dict[str, Any]]) -> None:
        """
        Persist an exit lifecycle action:
          - Update position status and exit fields
          - Compute realized PnL
          - Return capital to bankroll
          - Record in daily_pnl
        """
        now = datetime.now(timezone.utc).isoformat()
        today = now[:10]

        # Find the position dict for PnL calculation
        pos = next((p for p in positions if p.get("id") == action.position_id), None)
        if pos is None:
            return

        exit_price = action.exit_price or pos.get("current_price") or pos.get("entry_price", 0.5)
        entry_price = pos.get("entry_price", 0.5)
        size_usd = pos.get("size_usd", 0.0)
        side = pos.get("side", "YES")

        if action.next_status.value in ("WON", "LOST"):
            # Settlement: WON returns $1/contract (full notional), LOST returns $0
            if action.next_status.value == "WON":
                realized_pnl = size_usd * (1.0 / max(entry_price, 0.01) - 1.0)
            else:
                realized_pnl = -size_usd
        else:
            # Mid-trade exit: PnL based on price move
            if side == "YES":
                realized_pnl = (exit_price - entry_price) * size_usd / max(entry_price, 0.01)
            else:
                realized_pnl = (entry_price - exit_price) * size_usd / max(1.0 - entry_price, 0.01)

        opened_at = pos.get("opened_at", now)
        try:
            hold_hours = (
                datetime.fromisoformat(now) - datetime.fromisoformat(opened_at)
            ).total_seconds() / 3600
        except Exception:
            hold_hours = 0.0

        self.conn.execute(
            """UPDATE positions
               SET status=?, exit_price=?, exit_reason=?,
                   realized_pnl=?, hold_time_hours=?, closed_at=?, updated_at=?
               WHERE id=?""",
            (
                action.next_status.value, exit_price, action.reason,
                realized_pnl, hold_hours, now, now,
                action.position_id,
            ),
        )

        # Return capital (size_usd) + PnL to bankroll
        returned = size_usd + realized_pnl
        self.conn.execute(
            "UPDATE portfolio SET bankroll = bankroll + ?, updated_at=? WHERE id=1",
            (returned, now),
        )

        # Log to daily_pnl
        self.conn.execute(
            """INSERT INTO daily_pnl (date, realized_pnl, num_trades, num_wins)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(date) DO UPDATE SET
                 realized_pnl = realized_pnl + excluded.realized_pnl,
                 num_trades   = num_trades + 1,
                 num_wins     = num_wins + excluded.num_wins""",
            (today, realized_pnl, 1 if realized_pnl > 0 else 0),
        )
        self.conn.commit()

        _log(
            f"[brain] EXIT {action.position_id} → {action.next_status.value} "
            f"({action.reason}) pnl=${realized_pnl:.2f}"
        )

    def _execute_order(self, order: dict[str, Any]) -> None:
        """
        Paper execution: log trade to predictions + positions tables and
        deduct from portfolio bankroll. In live mode this would also call
        the exchange API before writing to the DB.
        """
        from datetime import datetime, timezone
        sig: Signal = order["signal"]
        size_usd: float = order["size_usd"]
        now = datetime.now(timezone.utc).isoformat()

        fill = self.executor.place_order(
            ticker=sig.ticker,
            side=sig.side,
            size_usd=size_usd,
            price=sig.executable_price,
            dry_run=self.dry_run,
        )
        _log(
            f"[brain] ORDER {fill['status']}  {sig.ticker}  {sig.side}  "
            f"${size_usd:.2f} @ {fill['fill_price']:.3f}  "
            f"edge={sig.executable_edge:.3f}"
        )

        # Record prediction
        self.conn.execute(
            """INSERT OR IGNORE INTO predictions
               (strategy_name, ticker, city, target_date,
                fair_value, market_price, executable_price,
                edge, executable_edge, confidence,
                consensus_f, agreement, n_models,
                is_shadow, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (
                sig.strategy_name, sig.ticker, sig.city, sig.target_date,
                sig.fair_value, sig.market_price, sig.executable_price,
                sig.edge, sig.executable_edge, sig.confidence,
                sig.consensus_f, sig.agreement, sig.n_models,
                now,
            ),
        )

        # Build full entry snapshot for post-trade analysis
        import json as _json
        entry_reason = _json.dumps({
            "high_f": sig.high_f,
            "low_f": sig.low_f,
            "market_type": sig.market_type,
            "consensus_f": sig.consensus_f,
            "agreement": sig.agreement,
            "n_models": sig.n_models,
            "edge": sig.edge,
            "executable_edge": sig.executable_edge,
            "market_price": sig.market_price,
            "fair_value": sig.fair_value,
        })

        # Open position with full snapshot columns for lifecycle engine
        self.conn.execute(
            """INSERT INTO positions
               (strategy_name, market_id, ticker, city, target_date,
                high_f, low_f, market_type, exchange,
                side, entry_price, size_usd, status, entry_reason, opened_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'OPENED',?,?)""",
            (
                sig.strategy_name, sig.market_id, sig.ticker, sig.city, sig.target_date,
                sig.high_f, sig.low_f, sig.market_type, sig.source,
                sig.side, sig.executable_price, size_usd, entry_reason, now,
            ),
        )

        # Deduct from bankroll
        self.conn.execute(
            "UPDATE portfolio SET bankroll = bankroll - ?, updated_at=? WHERE id=1",
            (size_usd, now),
        )
        self.conn.commit()


def _log(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr)
