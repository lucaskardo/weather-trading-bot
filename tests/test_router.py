"""Tests for Phase 2: Strategy Router (Scorecard, Allocator, Selector, Brain)."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.params import Params, PARAMS
from shared.types import ModelForecast
from state.db import init_db
from strategies.base import Signal
from strategy_router.scorecard import (
    MIN_TRADES_FOR_SCORE,
    _compute_score,
    _max_drawdown_penalty,
    _sharpe,
    compute_score_from_trades,
    score_all_strategies,
    score_strategy,
)
from strategy_router.allocator import (
    _apply_caps,
    _softmax_fracs,
    allocate,
)
from strategy_router.selector import (
    _cluster_exposure,
    _kelly_size,
    select_signals,
)
from strategy_router.brain import Brain


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _params(**kw) -> Params:
    p = Params()
    p.min_depth_usd = 0.0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _make_trade(pnl: float = 5.0, brier: float = 0.10, edge: float = 0.08) -> dict:
    return {"realized_pnl": pnl, "brier_score": brier, "executable_edge": edge, "outcome": 1.0, "fair_value": 0.65}


def _trades(n: int, pnl: float = 5.0, brier: float = 0.10, edge: float = 0.08) -> list[dict]:
    return [_make_trade(pnl, brier, edge) for _ in range(n)]


def _signal(
    ticker="TKR",
    city="NYC",
    strategy="value_entry",
    edge=0.10,
    confidence=0.8,
    fair_value=0.70,
    market_price=0.55,
    is_shadow=False,
    side="YES",
) -> Signal:
    return Signal(
        strategy_name=strategy,
        market_id=ticker,
        ticker=ticker,
        source="kalshi",
        city=city,
        target_date="2026-06-01",
        market_price=market_price,
        fair_value=fair_value,
        executable_price=market_price + 0.02,
        edge=edge,
        executable_edge=edge,
        confidence=confidence,
        is_shadow=is_shadow,
        side=side,
    )


def _open_position(city="BOS", size_usd=100.0) -> dict:
    return {"id": 1, "city": city, "size_usd": size_usd, "status": "HOLDING"}


def _db(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "router_test.db")


# --------------------------------------------------------------------------- #
# Phase 2.1 — Scorecard
# --------------------------------------------------------------------------- #

class TestScorecardHelpers:
    def test_sharpe_positive_pnl(self):
        pnls = [5.0, 6.0, 4.0, 5.5]
        assert _sharpe(pnls) > 0

    def test_sharpe_zero_std(self):
        assert _sharpe([5.0]) == 0.0

    def test_max_drawdown_no_loss(self):
        assert _max_drawdown_penalty([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_max_drawdown_with_loss(self):
        # peak=10, then drops to 5 → dd=0.5
        pnls = [5.0, 5.0, -5.0]
        dd = _max_drawdown_penalty(pnls)
        assert dd == pytest.approx(0.5, abs=0.01)

    def test_max_drawdown_clamped_to_one(self):
        pnls = [1.0, -100.0]
        assert _max_drawdown_penalty(pnls) <= 1.0


class TestComputeScoreFromTrades:
    def test_returns_none_below_min_trades(self):
        result = compute_score_from_trades(_trades(5), min_trades=10)
        assert result is None

    def test_returns_score_at_min_trades(self):
        result = compute_score_from_trades(_trades(10), min_trades=10)
        assert result is not None

    def test_score_in_range(self):
        result = compute_score_from_trades(_trades(15))
        assert 0.0 <= result <= 100.0

    def test_good_trades_score_higher_than_bad(self):
        good = compute_score_from_trades(_trades(15, pnl=10.0, brier=0.05, edge=0.12))
        bad = compute_score_from_trades(_trades(15, pnl=-5.0, brier=0.35, edge=0.01))
        assert good > bad

    def test_perfect_calibration_boosts_score(self):
        good_brier = compute_score_from_trades(_trades(15, brier=0.01))
        bad_brier = compute_score_from_trades(_trades(15, brier=0.40))
        assert good_brier > bad_brier

    def test_high_drawdown_penalises_score(self):
        # Alternating big wins then big loss
        trades_dd = (
            [_make_trade(pnl=10.0)] * 7
            + [_make_trade(pnl=-50.0)]
            + [_make_trade(pnl=2.0)] * 7
        )
        trades_ok = _trades(15, pnl=5.0)
        score_dd = compute_score_from_trades(trades_dd, min_trades=10)
        score_ok = compute_score_from_trades(trades_ok, min_trades=10)
        assert score_dd < score_ok


class TestScoreStrategyFromDb:
    def test_returns_none_with_empty_db(self, tmp_path):
        conn = _db(tmp_path)
        result = score_strategy("value_entry", conn, min_trades=10)
        assert result is None

    def test_returns_score_with_sufficient_data(self, tmp_path):
        conn = _db(tmp_path)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        for i in range(12):
            conn.execute(
                """INSERT INTO predictions
                   (strategy_name, ticker, city, target_date, fair_value, market_price,
                    edge, is_shadow, outcome, brier_score, executable_edge, realized_pnl,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("value_entry", f"TKR{i}", "NYC", "2026-06-01",
                 0.65, 0.50, 0.10, 0, 1.0, 0.12, 0.08, 5.0, cutoff),
            )
        conn.commit()
        result = score_strategy("value_entry", conn, min_trades=10)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_score_all_strategies(self, tmp_path):
        conn = _db(tmp_path)
        result = score_all_strategies(["value_entry", "model_release"], conn, min_trades=10)
        assert "value_entry" in result
        assert "model_release" in result
        # Both should be None (no data)
        assert result["value_entry"] is None
        assert result["model_release"] is None


# --------------------------------------------------------------------------- #
# Phase 2.2 — Allocator
# --------------------------------------------------------------------------- #

class TestSoftmaxFracs:
    def test_equal_scores_equal_fracs(self):
        fracs = _softmax_fracs({"A": 50.0, "B": 50.0}, temperature=10.0)
        assert fracs["A"] == pytest.approx(0.5, abs=0.01)
        assert fracs["B"] == pytest.approx(0.5, abs=0.01)

    def test_higher_score_gets_more(self):
        fracs = _softmax_fracs({"A": 80.0, "B": 20.0}, temperature=10.0)
        assert fracs["A"] > fracs["B"]

    def test_fracs_sum_to_one(self):
        fracs = _softmax_fracs({"A": 70.0, "B": 60.0, "C": 50.0}, temperature=10.0)
        assert sum(fracs.values()) == pytest.approx(1.0)

    def test_single_strategy_gets_all(self):
        fracs = _softmax_fracs({"A": 75.0}, temperature=10.0)
        assert fracs["A"] == pytest.approx(1.0)


class TestApplyCaps:
    def test_caps_max_at_40_pct(self):
        # One strategy dominates before capping
        fracs = _apply_caps({"A": 0.90, "B": 0.10})
        assert fracs["A"] <= 0.40 + 1e-9

    def test_floors_min_at_5_pct(self):
        fracs = _apply_caps({"A": 0.97, "B": 0.03})
        assert fracs["B"] >= 0.05 - 1e-9

    def test_fracs_sum_to_one_after_capping(self):
        fracs = _apply_caps({"A": 0.60, "B": 0.30, "C": 0.10})
        assert sum(fracs.values()) == pytest.approx(1.0, abs=1e-6)


class TestAllocate:
    def test_total_deployment_within_80_pct(self):
        scores = {"A": 70.0, "B": 60.0}
        allocs = allocate(scores, bankroll=1000.0)
        assert sum(allocs.values()) <= 800.0 + 1e-6

    def test_no_strategy_exceeds_40_pct_of_available(self):
        scores = {"A": 100.0, "B": 10.0}
        allocs = allocate(scores, bankroll=1000.0)
        available = 800.0
        for usd in allocs.values():
            assert usd / available <= 0.40 + 1e-6

    def test_each_strategy_gets_at_least_5_pct(self):
        scores = {"A": 100.0, "B": 1.0}
        allocs = allocate(scores, bankroll=1000.0)
        available = 800.0
        for usd in allocs.values():
            assert usd / available >= 0.05 - 1e-6

    def test_unscored_strategy_gets_min_floor(self):
        scores = {"A": 70.0, "B": None}
        allocs = allocate(scores, bankroll=1000.0)
        available = 800.0
        # B is unscored → gets exactly min floor
        assert allocs["B"] / available == pytest.approx(0.05, abs=0.01)

    def test_empty_scores_returns_empty(self):
        assert allocate({}, bankroll=1000.0) == {}

    def test_zero_bankroll_returns_empty(self):
        assert allocate({"A": 70.0}, bankroll=0.0) == {}

    def test_higher_score_gets_more_capital(self):
        # Use moderate, close scores so neither strategy hits the 40% cap
        # (extreme score gaps cause both to reach the max cap and equalize)
        scores = {"A": 60.0, "B": 50.0, "C": 40.0}
        allocs = allocate(scores, bankroll=1000.0)
        assert allocs["A"] >= allocs["B"] >= allocs["C"]


# --------------------------------------------------------------------------- #
# Phase 2.3 — Selector
# --------------------------------------------------------------------------- #

class TestKellySize:
    def test_positive_edge_gives_positive_size(self):
        sig = _signal(edge=0.10, fair_value=0.70, market_price=0.55)
        size = _kelly_size(sig, budget=500.0, params=_params())
        assert size > 0

    def test_zero_edge_gives_zero_size(self):
        sig = _signal(edge=0.0, fair_value=0.55, market_price=0.55)
        size = _kelly_size(sig, budget=500.0, params=_params())
        assert size == 0.0

    def test_capped_at_max_kelly(self):
        sig = _signal(edge=0.30, fair_value=0.95, market_price=0.55)
        p = _params(max_kelly_fraction=0.25)
        size = _kelly_size(sig, budget=1000.0, params=p)
        assert size <= 0.25 * 1000.0

    def test_floored_at_min_kelly(self):
        sig = _signal(edge=0.06, fair_value=0.58, market_price=0.52)
        p = _params(min_kelly_fraction=0.01)
        size = _kelly_size(sig, budget=1000.0, params=p)
        assert size >= 0.01 * 1000.0


class TestClusterExposure:
    def test_computes_cluster_exposure(self):
        positions = [
            {"city": "NYC", "size_usd": 200.0},
            {"city": "BOS", "size_usd": 150.0},
        ]
        exposure = _cluster_exposure(positions, bankroll=1000.0)
        assert exposure.get("northeast", 0.0) == pytest.approx(350.0)

    def test_unknown_city_not_counted(self):
        positions = [{"city": "UNKNOWN", "size_usd": 500.0}]
        exposure = _cluster_exposure(positions, bankroll=1000.0)
        assert sum(exposure.values()) == pytest.approx(0.0)


class TestSelectSignals:
    def test_shadow_signal_skipped(self):
        sig = _signal(is_shadow=True)
        orders = select_signals([sig], {"value_entry": 500.0}, [], 1000.0, _params())
        assert orders[0]["reason_skipped"] == "shadow"
        assert orders[0]["size_usd"] == 0.0

    def test_live_signal_with_budget_executes(self):
        sig = _signal(is_shadow=False, strategy="value_entry")
        orders = select_signals([sig], {"value_entry": 500.0}, [], 1000.0, _params())
        assert orders[0]["reason_skipped"] is None
        assert orders[0]["size_usd"] > 0

    def test_cluster_cap_enforced(self):
        # Fill northeast cluster to 15%+ of bankroll
        bankroll = 1000.0
        positions = [
            {"city": "NYC", "size_usd": 80.0},
            {"city": "BOS", "size_usd": 70.0},
        ]  # 150/1000 = 15% → at cap
        sig = _signal(city="DC", strategy="value_entry")  # DC is also northeast
        p = _params(max_cluster_exposure_pct=0.15)
        orders = select_signals([sig], {"value_entry": 500.0}, positions, bankroll, p)
        assert orders[0]["reason_skipped"] == "cluster_cap"

    def test_city_limit_enforced(self):
        positions = [
            {"city": "NYC", "size_usd": 50.0, "status": "HOLDING"},
            {"city": "NYC", "size_usd": 50.0, "status": "HOLDING"},
            {"city": "NYC", "size_usd": 50.0, "status": "HOLDING"},
        ]
        sig = _signal(city="NYC", strategy="value_entry")
        p = _params(max_positions_per_city=3)
        orders = select_signals([sig], {"value_entry": 500.0}, positions, 1000.0, p)
        assert orders[0]["reason_skipped"] == "city_limit"

    def test_no_budget_skipped(self):
        sig = _signal()
        orders = select_signals([sig], {}, [], 1000.0, _params())
        assert orders[0]["reason_skipped"] == "no_budget"

    def test_sorted_by_edge_times_confidence(self):
        s1 = _signal(ticker="T1", edge=0.05, confidence=0.5)   # score 0.025
        s2 = _signal(ticker="T2", edge=0.15, confidence=0.8)   # score 0.12
        orders = select_signals(
            [s1, s2], {"value_entry": 1000.0}, [], 2000.0, _params()
        )
        tickers = [o["signal"].ticker for o in orders]
        assert tickers[0] == "T2"  # higher score first

    def test_shadow_strategy_name_is_skipped(self):
        sig = _signal(strategy="convergence_exit", is_shadow=True)
        orders = select_signals([sig], {"convergence_exit": 500.0}, [], 1000.0, _params())
        assert orders[0]["reason_skipped"] == "shadow"


# --------------------------------------------------------------------------- #
# Phase 2.4 — Brain
# --------------------------------------------------------------------------- #

class TestBrain:
    def test_run_cycle_returns_summary(self, tmp_path):
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[])
        assert "signals_generated" in result
        assert "exits" in result
        assert "scores" in result
        assert "allocations" in result
        assert "executed" in result
        assert result["dry_run"] is True

    def test_zero_signals_with_no_markets(self, tmp_path):
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[])
        assert result["signals_generated"] == 0

    def test_cycle_does_not_raise_with_empty_positions(self, tmp_path):
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[], open_positions=[])
        assert result["exits"] == 0

    def test_bankroll_read_from_db(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute("UPDATE portfolio SET bankroll=5000.0 WHERE id=1")
        conn.commit()
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        assert brain._get_bankroll() == pytest.approx(5000.0)

    def test_scores_all_registered_strategies(self, tmp_path):
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[])
        expected = {"value_entry", "convergence_exit", "model_release", "disagreement"}
        assert expected.issubset(set(result["scores"].keys()))

    def test_shadow_signals_not_counted_as_executed(self, tmp_path):
        """Shadow signals should never increment executed count."""
        conn = _db(tmp_path)
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[])
        # With no markets there are no signals, so executed must be 0
        assert result["signals_shadow"] == 0

    def test_allocations_respect_bankroll(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute("UPDATE portfolio SET bankroll=2000.0 WHERE id=1")
        conn.commit()
        brain = Brain(conn=conn, params=_params(), dry_run=True)
        result = brain.run_cycle(markets=[], forecasts=[])
        total_alloc = sum(result["allocations"].values())
        # Must not exceed 80% of bankroll
        assert total_alloc <= 2000.0 * 0.80 + 1e-6


# --------------------------------------------------------------------------- #
# Phase 2.5 — main.py startup checks
# --------------------------------------------------------------------------- #

class TestStartupChecks:
    def test_params_sanity_passes_defaults(self):
        from main import _check_params_sanity
        _check_params_sanity(PARAMS)  # should not raise

    def test_params_sanity_fails_bad_std(self):
        from main import _check_params_sanity
        p = _params(base_std_f=0.5)  # below minimum of 1
        with pytest.raises(AssertionError):
            _check_params_sanity(p)

    def test_params_sanity_fails_bad_weights(self):
        from main import _check_params_sanity
        p = _params(router_w_sharpe=0.99)  # weights no longer sum to 1
        with pytest.raises(AssertionError):
            _check_params_sanity(p)
