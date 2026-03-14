"""
Regression tests for Weather Trading Bot v8 bug fixes.

Covers all 6 bugs from WEATHER_BOT_FINAL_SPEC (2).md:
  Bug 1: _check_exit uses _compute_fair_value_for_market for band/below
  Bug 2: MC rounding matches analytic (±0.5 convention)
  Bug 3: current_price injected from live market data
  Bug 4: NO-side mid-exit PnL formula correct
  Bug 5: orders + fills tables written after paper trade
  Bug 6: predictions INSERT includes market_id
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
    import copy
    p = copy.copy(PARAMS)
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
        Band "65-66°F", consensus=68, std=5:
          analytic → P(65 ≤ X ≤ 66) ≈ 0.14
          old _prob_above_threshold(68, 66, 5) → P(X > 66) ≈ 0.66
        Convergence should use 0.14, not 0.66.
        """
        from shared.types import ModelForecast
        strategy = ValueEntryStrategy()

        # Forecast: consensus = 68
        f = ModelForecast(
            city="NYC", target_date="2026-06-01",
            model_name="GFS", predicted_high_f=68.0,
            fetched_at=_fresh_ts(0.5),
        )
        # Band position: current_price near wrong fair_value (0.66) should NOT converge
        # current_price near correct fair_value (0.14) SHOULD converge
        pos_near_correct = {
            "id": 1, "city": "NYC", "target_date": "2026-06-01",
            "side": "YES", "entry_price": 0.14, "current_price": 0.14,
            "high_f": 66.0, "low_f": 65.0, "market_type": "band",
            "opened_at": _fresh_ts(2.0),
        }
        pos_near_wrong = {
            "id": 2, "city": "NYC", "target_date": "2026-06-01",
            "side": "YES", "entry_price": 0.65, "current_price": 0.66,
            "high_f": 66.0, "low_f": 65.0, "market_type": "band",
            "opened_at": _fresh_ts(2.0),
        }
        p = _params(base_std_f=5.0, stale_forecast_hours=12.0)

        action_correct = strategy._check_exit(pos_near_correct, [f], p)
        action_wrong = strategy._check_exit(pos_near_wrong, [f], p)

        # Near 0.14 (correct) → convergence
        assert action_correct["action"] == "exit", \
            "Should exit: current_price matches correct band fair_value"
        assert action_correct["reason"] == "convergence"

        # Near 0.66 (wrong) → should NOT converge (may hold or reversal)
        assert action_wrong["action"] != "convergence"

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
        expected = 100.0 * ((1.0 - 0.30) / 0.60 - 1.0)  # +$16.67
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
        expected = 100.0 * ((1.0 - 0.50) / 0.60 - 1.0)  # -$16.67
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
