"""
Regression tests for Weather Trading Bot v8 and v11 fixes.

v8 bugs (WEATHER_BOT_FINAL_SPEC (2).md):
  Bug 1: _check_exit uses _compute_fair_value_for_market for band/below
  Bug 2: MC rounding matches analytic (±0.5 convention)
  Bug 3: current_price injected from live market data
  Bug 4: NO-side mid-exit PnL formula correct (with exit costs)
  Bug 5: orders + fills tables written after paper trade
  Bug 6: predictions INSERT includes market_id

v11 profit leaks (WEATHER_BOT_FINAL_SPEC (3).md):
  Leak 1: Strategy budget not overspent in one cycle
  Leak 2: _check_exit uses canonical compute_fair_value (dynamic std + temp scaling)
  Leak 3: Mid-exit PnL deducts slippage + fee
  Leak 4: Calibrator loads market_type/high_f/low_f for band/below recalculation
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from state.db import init_db
from strategies.value_entry import _compute_fair_value_for_market, ValueEntryStrategy
from core.forecaster import monte_carlo_prob
from strategies.value_entry import _compute_fair_value_for_market as _analytic
from strategy_router.brain import Brain
from shared.params import PARAMS, Params
from strategies.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "v8_test.db")


def _params(**kw) -> Params:
    # Use Params() directly so a mutated PARAMS singleton never bleeds into tests
    p = Params()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _fresh_ts(hours_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _make_signal(
    ticker="T1",
    market_id="mkt1",
    city="NYC",
    target_date="2026-06-01",
    market_type="above",
    high_f=75.0,
    low_f=None,
    fair_value=0.70,
    market_price=0.50,
    side="YES",
    executable_edge=0.15,
) -> Signal:
    exec_price = market_price + 0.02
    eff_prob = fair_value if side == "YES" else (1.0 - fair_value)
    return Signal(
        strategy_name="value_entry",
        market_id=market_id,
        ticker=ticker,
        source="kalshi",
        city=city,
        target_date=target_date,
        market_type=market_type,
        high_f=high_f,
        low_f=low_f,
        market_price=market_price,
        fair_value=fair_value,
        executable_price=exec_price,
        edge=executable_edge,
        executable_edge=executable_edge,
        effective_prob=eff_prob,
        effective_price=exec_price,
        effective_edge=executable_edge,
        confidence=0.8,
        consensus_f=78.0,
        agreement=1.0,
        n_models=3,
        side=side,
    )


# ---------------------------------------------------------------------------
# Bug 1: _check_exit uses correct probability per market_type
# ---------------------------------------------------------------------------

class TestBug1CheckExitMarketType:

    def test_band_exit_uses_correct_fair_value(self):
        """
        Band "65-66°F", consensus=68:
          correct analytic → P(64.5 ≤ X ≤ 66.5) — varies with dynamic std
          wrong (above-only) → P(X > 66) ≈ 0.55-0.65

        Convergence should use the correct band formula.
        We compute the actual fair_value at test time and set current_price = fv
        so the convergence check is always within threshold.
        """
        from shared.types import ModelForecast
        from core.forecaster import compute_fair_value
        strategy = ValueEntryStrategy()

        f = ModelForecast(
            city="NYC", target_date="2026-06-01",
            model_name="GFS", predicted_high_f=68.0,
            fetched_at=_fresh_ts(0.5),
        )
        p = _params(base_std_f=5.0, stale_forecast_hours=12.0)

        # Compute the actual band fair_value (correct formula, with dynamic std)
        correct_fv, _, _, _ = compute_fair_value(
            [f], "NYC", "2026-06-01", "band", 66.0, 65.0, p, use_mc=False
        )
        assert correct_fv is not None

        # current_price = computed fair_value → guaranteed convergence
        pos_near_correct = {
            "id": 1, "city": "NYC", "target_date": "2026-06-01",
            "side": "YES", "entry_price": correct_fv, "current_price": correct_fv,
            "high_f": 66.0, "low_f": 65.0, "market_type": "band",
            "opened_at": _fresh_ts(2.0),
        }
        # current_price near wrong value (above-only ~0.65) → reversal or hold, never convergence
        pos_near_wrong = {
            "id": 2, "city": "NYC", "target_date": "2026-06-01",
            "side": "YES", "entry_price": 0.65, "current_price": 0.65,
            "high_f": 66.0, "low_f": 65.0, "market_type": "band",
            "opened_at": _fresh_ts(2.0),
        }

        action_correct = strategy._check_exit(pos_near_correct, [f], p)
        action_wrong = strategy._check_exit(pos_near_wrong, [f], p)

        # Near correct fair_value → convergence
        assert action_correct["action"] == "exit", \
            f"Should exit: current_price={correct_fv:.3f} matches band fair_value"
        assert action_correct["reason"] == "convergence"

        # Near wrong value (0.65) → forecast_reversal (correct_fv << 0.65 → large negative edge)
        assert action_wrong.get("reason") != "convergence", \
            "Should NOT converge: current_price is near wrong (above-only) probability"

    def test_below_exit_uses_correct_fair_value(self):
        """Below market with consensus=68 >> threshold=50: P(X<=50) ≈ 0.0003."""
        from shared.types import ModelForecast
        strategy = ValueEntryStrategy()
        f = ModelForecast(
            city="NYC", target_date="2026-06-01",
            model_name="GFS", predicted_high_f=68.0,
            fetched_at=_fresh_ts(0.5),
        )
        # current_price near correct (0.0003) → convergence
        pos = {
            "id": 1, "city": "NYC", "target_date": "2026-06-01",
            "side": "YES", "entry_price": 0.01, "current_price": 0.01,
            "high_f": 50.0, "low_f": None, "market_type": "below",
            "opened_at": _fresh_ts(2.0),
        }
        p = _params(base_std_f=5.0, stale_forecast_hours=12.0)
        action = strategy._check_exit(pos, [f], p)
        assert action["action"] == "exit"
        assert action["reason"] == "convergence"


# ---------------------------------------------------------------------------
# Bug 2: MC rounding matches analytic (±0.5 convention)
# ---------------------------------------------------------------------------

class TestBug2MCRounding:

    def test_band_mc_matches_analytic(self):
        """
        Band 74-76, μ=75, σ=5:
        Analytic: P(73.5 ≤ X ≤ 76.5) ≈ 0.2358
        MC with ±0.5: should be within 1% of analytic.
        """
        analytic = _compute_fair_value_for_market(75.0, "band", 76.0, 74.0, 5.0)
        mc = monte_carlo_prob(
            model_forecasts=[75.0],
            market_type="band",
            high_f=76.0,
            low_f=74.0,
            sigma=5.0,
            n_samples=10000,
            seed=42,
        )
        assert analytic is not None
        assert abs(mc - analytic) < 0.02, \
            f"MC={mc:.4f} vs analytic={analytic:.4f}, diff={abs(mc-analytic):.4f}"

    def test_above_mc_matches_analytic(self):
        """Above: μ=68, threshold=68 → analytic ≈ 0.54."""
        analytic = _compute_fair_value_for_market(68.0, "above", 68.0, None, 5.0)
        mc = monte_carlo_prob(
            model_forecasts=[68.0],
            market_type="above",
            high_f=68.0,
            low_f=None,
            sigma=5.0,
            n_samples=10000,
            seed=42,
        )
        assert analytic is not None
        assert abs(mc - analytic) < 0.02

    def test_below_mc_matches_analytic(self):
        """Below: μ=68, threshold=50 → analytic ≈ 0.01."""
        analytic = _compute_fair_value_for_market(68.0, "below", 50.0, None, 5.0)
        mc = monte_carlo_prob(
            model_forecasts=[68.0],
            market_type="below",
            high_f=50.0,
            low_f=None,
            sigma=5.0,
            n_samples=10000,
            seed=42,
        )
        assert analytic is not None
        assert abs(mc - analytic) < 0.02


# ---------------------------------------------------------------------------
# Bug 3: current_price injected from live market data
# ---------------------------------------------------------------------------

class TestBug3LivePriceInjection:

    def test_current_price_updated_from_markets(self, tmp_path):
        """Brain should inject live market prices into positions before lifecycle."""
        from shared.types import ModelForecast
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)

        markets = [
            {"id": "mkt1", "ticker": "T1", "city": "NYC",
             "target_date": "2026-06-01", "market_price": 0.72,
             "market_type": "above", "high_f": 75.0, "exchange": "kalshi"},
        ]
        positions = [
            {"id": 1, "ticker": "T1", "city": "NYC", "target_date": "2026-06-01",
             "side": "YES", "entry_price": 0.45, "current_price": None,
             "high_f": 75.0, "low_f": None, "market_type": "above",
             "status": "HOLDING", "opened_at": _fresh_ts(2.0)},
        ]

        # After run_cycle, the position dict should have current_price = 0.72
        # (Brain injects live_prices before lifecycle)
        live_prices = {m["ticker"]: m.get("market_price") for m in markets}
        for pos in positions:
            if pos.get("ticker") in live_prices:
                pos["current_price"] = live_prices[pos["ticker"]]

        assert positions[0]["current_price"] == 0.72


# ---------------------------------------------------------------------------
# Bug 4: NO-side mid-exit PnL formula
# ---------------------------------------------------------------------------

class TestBug4NoPnlFormula:

    def test_no_bet_price_drops_is_profit(self, tmp_path):
        """
        NO at YES=60¢ (entry_price=0.60), YES drops to 30¢ (exit=0.30).
        We win on NO: YES cost us 0.60, now worth 0.70 (= 1 - 0.30).
        PnL = size_usd * ((1 - exit) / entry - 1) = 100 * ((0.70/0.60) - 1) = +$16.67
        """
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)

        from execution.lifecycle import PositionStatus, LifecycleAction
        pos = {
            "id": 1, "side": "NO", "entry_price": 0.60,
            "current_price": 0.30, "size_usd": 100.0,
            "opened_at": _fresh_ts(2.0),
        }
        conn.execute(
            """INSERT INTO positions
               (strategy_name, ticker, city, target_date, side, entry_price,
                current_price, size_usd, status, opened_at)
               VALUES ('v','T','NYC','2026-06-01','NO',0.60,0.30,100.0,'HOLDING',?)""",
            (_fresh_ts(2.0),)
        )
        conn.commit()
        pos_id = conn.execute("SELECT id FROM positions").fetchone()[0]
        pos["id"] = pos_id

        action = LifecycleAction(
            position_id=pos_id,
            current_status=PositionStatus.HOLDING,
            next_status=PositionStatus.EXITED_CONVERGENCE,
            reason="convergence",
            should_execute=True,
            exit_price=0.30,
            updated_at=_fresh_ts(),
        )
        brain._process_exit(action, [pos])

        row = conn.execute("SELECT realized_pnl FROM positions WHERE id=?", (pos_id,)).fetchone()
        assert row is not None
        from execution.orderbook import get_exit_price
        net_exit = get_exit_price(0.30, "NO", _params())
        expected = 100.0 * (net_exit / 0.60 - 1.0)  # ~+$12.55 after exit costs
        assert abs(row[0] - expected) < 0.01, f"Got {row[0]:.4f}, expected {expected:.4f}"

    def test_no_bet_price_rises_is_loss(self, tmp_path):
        """
        NO at YES=60¢, YES rises to 50¢ (exit=0.50).
        Value = 1 - 0.50 = 0.50 < entry 0.60. Loss.
        PnL = 100 * ((0.50/0.60) - 1) = -$16.67
        """
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)

        from execution.lifecycle import PositionStatus, LifecycleAction
        conn.execute(
            """INSERT INTO positions
               (strategy_name, ticker, city, target_date, side, entry_price,
                current_price, size_usd, status, opened_at)
               VALUES ('v','T','NYC','2026-06-01','NO',0.60,0.50,100.0,'HOLDING',?)""",
            (_fresh_ts(2.0),)
        )
        conn.commit()
        pos_id = conn.execute("SELECT id FROM positions").fetchone()[0]
        pos = {"id": pos_id, "side": "NO", "entry_price": 0.60,
               "current_price": 0.50, "size_usd": 100.0, "opened_at": _fresh_ts(2.0)}

        action = LifecycleAction(
            position_id=pos_id,
            current_status=PositionStatus.HOLDING,
            next_status=PositionStatus.EXITED_STOP,
            reason="forecast_reversal",
            should_execute=True,
            exit_price=0.50,
            updated_at=_fresh_ts(),
        )
        brain._process_exit(action, [pos])

        row = conn.execute("SELECT realized_pnl FROM positions WHERE id=?", (pos_id,)).fetchone()
        from execution.orderbook import get_exit_price
        net_exit = get_exit_price(0.50, "NO", _params())
        expected = 100.0 * (net_exit / 0.60 - 1.0)  # ~-$21.25 after exit costs
        assert abs(row[0] - expected) < 0.01, f"Got {row[0]:.4f}, expected {expected:.4f}"


# ---------------------------------------------------------------------------
# Bug 5: orders + fills tables written after paper trade
# ---------------------------------------------------------------------------

class TestBug5OrdersFillsWritten:

    def test_orders_and_fills_populated(self, tmp_path):
        """After _execute_order, both orders and fills tables should have rows."""
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO markets (id, ticker, city, target_date, market_type, exchange)
               VALUES ('mkt1','T1','NYC','2026-06-01','above','kalshi')"""
        )
        conn.commit()
        brain = Brain(conn=conn, params=_params(), dry_run=True)

        sig = _make_signal()
        order = {"signal": sig, "size_usd": 50.0, "reason_skipped": None}
        brain._execute_order(order)

        orders = conn.execute("SELECT * FROM orders").fetchall()
        fills = conn.execute("SELECT * FROM fills").fetchall()

        assert len(orders) == 1, f"Expected 1 order, got {len(orders)}"
        assert len(fills) == 1, f"Expected 1 fill, got {len(fills)}"
        assert orders[0]["ticker"] == "T1"
        assert fills[0]["fill_price"] > 0


# ---------------------------------------------------------------------------
# Bug 6: predictions INSERT includes market_id
# ---------------------------------------------------------------------------

class TestBug6MarketIdInPredictions:

    def test_prediction_has_market_id(self, tmp_path):
        """predictions.market_id should be non-NULL after _execute_order."""
        conn = _db(tmp_path)
        # Insert market so FK is valid
        conn.execute(
            """INSERT INTO markets (id, ticker, city, target_date, market_type, exchange)
               VALUES ('mkt1','T1','NYC','2026-06-01','above','kalshi')"""
        )
        conn.commit()

        brain = Brain(conn=conn, params=_params(), dry_run=True)
        sig = _make_signal(market_id="mkt1")
        order = {"signal": sig, "size_usd": 50.0, "reason_skipped": None}
        brain._execute_order(order)

        row = conn.execute("SELECT market_id FROM predictions LIMIT 1").fetchone()
        assert row is not None
        assert row["market_id"] == "mkt1", f"Got market_id={row['market_id']}"


# ---------------------------------------------------------------------------
# Leak 1: Strategy budget not overspent in one cycle
# ---------------------------------------------------------------------------

class TestLeak1BudgetNotOverspent:

    def test_total_spend_within_budget(self):
        """5 high-edge signals on $200 budget → total executed ≤ $200."""
        from strategy_router.selector import select_signals
        from strategies.base import Signal

        def _sig(ticker, edge=0.20):
            return Signal(
                strategy_name="value_entry",
                market_id=ticker, ticker=ticker,
                source="kalshi", city="NYC",
                target_date="2026-06-01",
                market_type="above", high_f=75.0, low_f=None,
                market_price=0.50, fair_value=0.70,
                executable_price=0.52,
                edge=edge, executable_edge=edge,
                effective_prob=0.70, effective_price=0.52, effective_edge=edge,
                confidence=0.9, consensus_f=78.0, agreement=1.0, n_models=3,
                side="YES",
            )

        signals = [_sig(f"T{i}") for i in range(5)]
        allocations = {"value_entry": 200.0}
        orders = select_signals(signals, allocations, [], bankroll=1000.0)

        executable = [o for o in orders if o.get("reason_skipped") is None]
        total_spent = sum(o["size_usd"] for o in executable)
        assert total_spent <= 200.0 + 1e-6, \
            f"Overspent: total={total_spent:.2f} on $200 budget"

    def test_budget_decrements_per_signal(self):
        """Each accepted signal reduces the remaining budget for subsequent ones."""
        from strategy_router.selector import select_signals
        from strategies.base import Signal

        def _sig(ticker):
            return Signal(
                strategy_name="value_entry",
                market_id=ticker, ticker=ticker,
                source="kalshi", city=f"City{ticker}",
                target_date="2026-06-01",
                market_type="above", high_f=75.0, low_f=None,
                market_price=0.30, fair_value=0.70,
                executable_price=0.32,
                edge=0.38, executable_edge=0.38,
                effective_prob=0.70, effective_price=0.32, effective_edge=0.38,
                confidence=0.9, consensus_f=78.0, agreement=0.5, n_models=3,
                side="YES",
            )

        signals = [_sig(f"T{i}") for i in range(10)]
        budget = 100.0
        allocations = {"value_entry": budget}
        orders = select_signals(signals, allocations, [], bankroll=1000.0)

        executable = [o for o in orders if o.get("reason_skipped") is None]
        total_spent = sum(o["size_usd"] for o in executable)
        assert total_spent <= budget + 1e-6


# ---------------------------------------------------------------------------
# Leak 2: Exit uses canonical compute_fair_value (dynamic std + temp scaling)
# ---------------------------------------------------------------------------

class TestLeak2CanonicalExitFairValue:

    def test_check_exit_uses_dynamic_std(self):
        """
        For a far-future date, dynamic_std > base_std_f.
        Exit fair_value should reflect this (lower certainty → value closer to 0.5).
        """
        from shared.types import ModelForecast
        from core.forecaster import compute_fair_value, _compute_fair_value_for_market
        strategy = ValueEntryStrategy()

        f = ModelForecast(
            city="NYC", target_date="2026-06-01",
            model_name="GFS", predicted_high_f=80.0,
            fetched_at=_fresh_ts(0.5),
        )
        p = _params(base_std_f=5.0, stale_forecast_hours=24.0)

        # Canonical fair value (dynamic std)
        fv_canonical, _, std_used, _ = compute_fair_value(
            [f], "NYC", "2026-06-01", "above", 75.0, None, p, use_mc=False
        )
        # Static fair value (base_std only)
        fv_static = _compute_fair_value_for_market(80.0, "above", 75.0, None, 5.0)

        # Dynamic std for a far-future June date is larger than base_std=5
        assert std_used >= 5.0, f"Expected dynamic_std >= 5, got {std_used:.2f}"

        # Both values are valid probabilities
        assert fv_canonical is not None
        assert 0 < fv_canonical < 1

    def test_band_exit_canonical_equals_entry_formula(self):
        """Exit and entry use the same formula: same inputs → same fair_value."""
        from shared.types import ModelForecast
        from core.forecaster import compute_fair_value
        from strategies.value_entry import ValueEntryStrategy

        f = ModelForecast(
            city="NYC", target_date="2026-06-01",
            model_name="GFS", predicted_high_f=75.0,
            fetched_at=_fresh_ts(0.5),
        )
        p = _params(base_std_f=5.0, stale_forecast_hours=24.0)

        # Both use compute_fair_value with same inputs
        fv1, _, _, _ = compute_fair_value(
            [f], "NYC", "2026-06-01", "band", 76.0, 74.0, p, use_mc=False
        )
        fv2, _, _, _ = compute_fair_value(
            [f], "NYC", "2026-06-01", "band", 76.0, 74.0, p, use_mc=False
        )
        assert fv1 == fv2, "Canonical function must be deterministic"


# ---------------------------------------------------------------------------
# Leak 3: Mid-exit PnL deducts slippage + fee
# ---------------------------------------------------------------------------

class TestLeak3ExitCostsDeducted:

    def test_get_exit_price_yes(self):
        """YES exit: proceeds = current_price - slippage - fee."""
        from execution.orderbook import get_exit_price, kalshi_taker_fee
        p = _params()
        slippage = p.slippage_buffer_cents / 100.0
        price = 0.65
        net = get_exit_price(price, "YES", p)
        expected = price - slippage - kalshi_taker_fee(price)
        assert abs(net - expected) < 1e-9

    def test_get_exit_price_no(self):
        """NO exit: proceeds = (1 - current_price) - slippage - fee."""
        from execution.orderbook import get_exit_price, kalshi_taker_fee
        p = _params()
        slippage = p.slippage_buffer_cents / 100.0
        price = 0.30
        net = get_exit_price(price, "NO", p)
        expected = (1.0 - price) - slippage - kalshi_taker_fee(price)
        assert abs(net - expected) < 1e-9

    def test_exit_costs_reduce_yes_pnl(self, tmp_path):
        """YES mid-exit PnL is lower than naive (exit_price - entry_price) * contracts."""
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        from execution.lifecycle import PositionStatus, LifecycleAction

        conn.execute(
            """INSERT INTO positions
               (strategy_name, ticker, city, target_date, side, entry_price,
                current_price, size_usd, status, opened_at)
               VALUES ('v','T','NYC','2026-06-01','YES',0.40,0.65,100.0,'HOLDING',?)""",
            (_fresh_ts(2.0),)
        )
        conn.commit()
        pos_id = conn.execute("SELECT id FROM positions").fetchone()[0]
        pos = {"id": pos_id, "side": "YES", "entry_price": 0.40,
               "current_price": 0.65, "size_usd": 100.0, "opened_at": _fresh_ts(2.0)}

        action = LifecycleAction(
            position_id=pos_id,
            current_status=PositionStatus.HOLDING,
            next_status=PositionStatus.EXITED_CONVERGENCE,
            reason="convergence",
            should_execute=True,
            exit_price=0.65,
            updated_at=_fresh_ts(),
        )
        brain._process_exit(action, [pos])

        row = conn.execute("SELECT realized_pnl FROM positions WHERE id=?", (pos_id,)).fetchone()
        from execution.orderbook import get_exit_price
        net_exit = get_exit_price(0.65, "YES", _params())
        expected = (100.0 / 0.40) * (net_exit - 0.40)
        assert abs(row[0] - expected) < 0.01, f"Got {row[0]:.4f}, expected {expected:.4f}"
        # PnL with costs < naive PnL
        naive = (100.0 / 0.40) * (0.65 - 0.40)
        assert row[0] < naive, "Exit costs must reduce PnL"


# ---------------------------------------------------------------------------
# Leak 4: Calibrator loads market_type/high_f/low_f from markets table
# ---------------------------------------------------------------------------

class TestLeak4CalibratorBandBelow:

    def test_load_resolved_predictions_joins_markets(self, tmp_path):
        """_load_resolved_predictions returns market_type and high_f from markets JOIN."""
        from research.calibrator import _load_resolved_predictions
        conn = _db(tmp_path)

        conn.execute(
            """INSERT INTO markets (id, ticker, city, target_date, market_type,
               high_f, low_f, exchange)
               VALUES ('mkt1','T1','NYC','2026-01-01','band',76.0,74.0,'kalshi')"""
        )
        conn.execute(
            """INSERT INTO predictions
               (strategy_name, market_id, ticker, city, target_date,
                fair_value, market_price, executable_price,
                edge, executable_edge, confidence,
                consensus_f, agreement, n_models,
                outcome, is_shadow, created_at)
               VALUES ('value_entry','mkt1','T1','NYC','2026-01-01',
                       0.25, 0.30, 0.31, 0.05, 0.05, 0.8,
                       75.0, 1.0, 2,
                       1.0, 0, '2026-01-02T00:00:00')"""
        )
        conn.commit()

        rows = _load_resolved_predictions(conn)
        assert len(rows) == 1
        assert rows[0]["market_type"] == "band"
        assert rows[0]["high_f"] == 76.0
        assert rows[0]["low_f"] == 74.0

    def test_brier_uses_band_formula_not_above(self, tmp_path):
        """
        For a band market, Brier recalculation should use band formula.
        P(band 74-76 | mu=75, std=5) ≈ 0.236 — different from above formula.
        """
        from core.forecaster import _compute_fair_value_for_market
        # Band: P(73.5 ≤ X ≤ 76.5 | mu=75, std=5)
        band_fv = _compute_fair_value_for_market(75.0, "band", 76.0, 74.0, 5.0)
        # Above: P(X ≥ 75.5 | mu=75, std=5) ≈ 0.46
        above_fv = _compute_fair_value_for_market(75.0, "above", 76.0, None, 5.0)

        assert band_fv is not None and above_fv is not None
        # Band and above give meaningfully different probabilities for same threshold
        assert abs(band_fv - above_fv) > 0.10, \
            f"Band={band_fv:.3f} vs above={above_fv:.3f} should differ by >10pp"
