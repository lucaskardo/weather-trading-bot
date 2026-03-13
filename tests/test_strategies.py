"""Tests for Phase 1: Strategy Framework + Active Management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.params import Params
from shared.types import ModelForecast
from strategies.base import BaseStrategy, Signal
from strategies.value_entry import (
    ValueEntryStrategy,
    _build_consensus,
    _is_forecast_stale,
    _hours_to_settlement,
    _prob_above_threshold,
)
from strategies.convergence_exit import ConvergenceExitStrategy
from strategies.model_release import ModelReleaseStrategy, _compute_run_delta
from strategies.disagreement import DisagreementStrategy, _find_best_model
from execution.lifecycle import (
    LifecycleAction,
    PositionStatus,
    TERMINAL_STATES,
    is_valid_transition,
    process_position,
    run_lifecycle_cycle,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _params(**overrides) -> Params:
    p = Params()
    p.min_executable_edge = 0.05
    p.base_std_f = 5.0
    p.stale_forecast_hours = 6.0
    p.min_depth_usd = 0.0   # disable liquidity filter for most tests
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _fresh_forecast(
    city="NYC",
    target_date="2026-06-01",
    model="GFS",
    high_f=80.0,
    run_id="2026060112",
    hours_ago: float = 1.0,
) -> ModelForecast:
    fetched = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return ModelForecast(
        model_name=model,
        city=city,
        target_date=target_date,
        predicted_high_f=high_f,
        run_id=run_id,
        publish_time=fetched,
        source_url="http://fake",
        fetched_at=fetched,
    )


def _market(
    ticker="KXNYC-26-80",
    city="NYC",
    target_date="2026-06-01",
    market_price=0.45,
    high_f=80.0,
) -> dict[str, Any]:
    return {
        "id": ticker,
        "ticker": ticker,
        "city": city,
        "target_date": target_date,
        "market_price": market_price,
        "high_f": high_f,
        "low_f": 60.0,
        "exchange": "kalshi",
        "market_type": "high_temp",
        "target_size_usd": 100.0,
    }


def _position(
    pos_id=1,
    city="NYC",
    target_date="2026-06-01",
    side="YES",
    entry_price=0.45,
    current_price=0.52,
    high_f=80.0,
    status="HOLDING",
    hours_ago: float = 2.0,
) -> dict[str, Any]:
    opened = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "id": pos_id,
        "city": city,
        "target_date": target_date,
        "side": side,
        "entry_price": entry_price,
        "current_price": current_price,
        "high_f": high_f,
        "status": status,
        "opened_at": opened,
        "strategy_name": "value_entry",
    }


# --------------------------------------------------------------------------- #
# Phase 1.1 — BaseStrategy interface
# --------------------------------------------------------------------------- #

class TestBaseStrategyInterface:
    def test_signal_dataclass_has_required_fields(self):
        sig = Signal(
            strategy_name="test",
            market_id="mkt1",
            ticker="TKR",
            source="kalshi",
            city="NYC",
            target_date="2026-06-01",
        )
        # Identity
        assert sig.strategy_name == "test"
        assert sig.ticker == "TKR"
        # Market
        assert sig.market_price == 0.5
        # Prediction
        assert sig.fair_value == 0.5
        assert sig.edge == 0.0
        assert sig.executable_edge == 0.0
        # Context
        assert sig.n_models == 0
        assert sig.model_temps_f == []
        # Metadata
        assert sig.is_shadow is False

    def test_base_strategy_is_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()  # cannot instantiate abstract class

    def test_concrete_strategy_must_implement_all_methods(self):
        class Incomplete(BaseStrategy):
            name = "incomplete"
            def generate_signals(self, *a, **kw): return []
            # missing manage_positions and evaluate

        with pytest.raises(TypeError):
            Incomplete()


# --------------------------------------------------------------------------- #
# Phase 1.2 — ValueEntryStrategy
# --------------------------------------------------------------------------- #

class TestValueEntrySignals:
    def test_generates_signal_with_sufficient_edge(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=88.0)]   # consensus 88 > threshold 80 → YES
        markets = [_market(market_price=0.40, high_f=80.0)]
        p = _params(min_executable_edge=0.05)

        signals = strategy.generate_signals(markets, forecasts, p)
        assert len(signals) == 1
        assert signals[0].executable_edge >= 0.05

    def test_no_signal_below_edge_threshold(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=80.5)]  # barely above threshold
        markets = [_market(market_price=0.50, high_f=80.0)]
        p = _params(min_executable_edge=0.20)  # very high threshold

        signals = strategy.generate_signals(markets, forecasts, p)
        assert len(signals) == 0

    def test_signal_has_executable_edge_not_raw(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=90.0)]
        markets = [_market(market_price=0.30, high_f=80.0)]
        p = _params(taker_fee_pct=0.01, slippage_buffer_cents=1.0, min_executable_edge=0.0)

        signals = strategy.generate_signals(markets, forecasts, p)
        assert len(signals) == 1
        # executable_edge must be strictly less than raw edge
        assert signals[0].executable_edge < signals[0].edge

    def test_signal_side_yes_when_model_above_market(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=90.0)]  # high consensus → high prob
        markets = [_market(market_price=0.35, high_f=80.0)]
        p = _params(min_executable_edge=0.0)

        signals = strategy.generate_signals(markets, forecasts, p)
        assert signals[0].side == "YES"

    def test_signal_sorted_by_edge_times_confidence(self):
        strategy = ValueEntryStrategy()
        forecasts = [
            _fresh_forecast(city="NYC", high_f=90.0),
            _fresh_forecast(city="CHI", high_f=88.0, model="ECMWF"),
        ]
        markets = [
            _market(ticker="TKR_A", city="NYC", market_price=0.30, high_f=80.0),
            _market(ticker="TKR_B", city="CHI", market_price=0.35, high_f=80.0),
        ]
        p = _params(min_executable_edge=0.0)
        signals = strategy.generate_signals(markets, forecasts, p)
        if len(signals) >= 2:
            scores = [s.executable_edge * s.confidence for s in signals]
            assert scores == sorted(scores, reverse=True)

    def test_no_signal_without_matching_forecast(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(city="CHI")]   # different city
        markets = [_market(city="NYC")]
        signals = strategy.generate_signals(markets, forecasts, _params())
        assert len(signals) == 0

    def test_is_live_true(self):
        assert ValueEntryStrategy.is_live is True

    def test_signal_is_not_shadow(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=90.0)]
        markets = [_market(market_price=0.30)]
        p = _params(min_executable_edge=0.0)
        signals = strategy.generate_signals(markets, forecasts, p)
        if signals:
            assert signals[0].is_shadow is False


class TestValueEntryExits:
    def test_exit_on_forecast_reversal(self):
        strategy = ValueEntryStrategy()
        # Forecast now shows low temp → prob reverses → edge goes negative
        forecasts = [_fresh_forecast(high_f=65.0)]  # way below threshold → prob ~ 0
        pos = _position(side="YES", entry_price=0.60, current_price=0.60, high_f=80.0)
        p = _params()

        actions = strategy.manage_positions([pos], forecasts, p)
        assert actions[0]["action"] == "exit"
        assert actions[0]["reason"] == "forecast_reversal"

    def test_exit_on_convergence(self):
        strategy = ValueEntryStrategy()
        # Consensus exactly AT threshold → fair_value ≈ 0.50; current_price ≈ 0.50
        forecasts = [_fresh_forecast(high_f=80.0)]  # consensus = threshold
        pos = _position(side="YES", current_price=0.50, high_f=80.0)
        p = _params()

        actions = strategy.manage_positions([pos], forecasts, p)
        assert actions[0]["action"] == "exit"
        assert actions[0]["reason"] == "convergence"

    def test_exit_on_stale_forecast(self):
        strategy = ValueEntryStrategy()
        # Forecast is 10 hours old
        forecasts = [_fresh_forecast(high_f=85.0, hours_ago=10.0)]
        pos = _position()
        p = _params(stale_forecast_hours=6.0)

        actions = strategy.manage_positions([pos], forecasts, p)
        assert actions[0]["action"] == "exit"
        assert actions[0]["reason"] == "stale_forecast"

    def test_hold_when_no_exit_condition(self):
        strategy = ValueEntryStrategy()
        forecasts = [_fresh_forecast(high_f=85.0, hours_ago=1.0)]
        # Entry 0.45, current 0.52, fair ~0.84 → good edge, fresh forecast
        pos = _position(entry_price=0.45, current_price=0.52)
        p = _params()

        actions = strategy.manage_positions([pos], forecasts, p)
        assert actions[0]["action"] == "hold"

    def test_evaluate_returns_required_keys(self):
        strategy = ValueEntryStrategy()
        trades = [
            {"realized_pnl": 10.0, "brier_score": 0.10},
            {"realized_pnl": -5.0, "brier_score": 0.20},
            {"realized_pnl": 8.0, "brier_score": 0.15},
        ]
        result = strategy.evaluate(trades)
        assert "sharpe" in result
        assert "brier" in result
        assert "win_rate" in result
        assert "trade_count" in result
        assert result["trade_count"] == 3

    def test_evaluate_empty_trades(self):
        strategy = ValueEntryStrategy()
        result = strategy.evaluate([])
        assert result["trade_count"] == 0


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

class TestBuildConsensus:
    def test_single_model(self):
        forecasts = [_fresh_forecast(high_f=82.0)]
        c, a, highs, n = _build_consensus(forecasts, "NYC", "2026-06-01")
        assert c == pytest.approx(82.0)
        assert n == 1

    def test_multiple_models(self):
        forecasts = [
            _fresh_forecast(model="GFS", high_f=80.0),
            _fresh_forecast(model="ECMWF", high_f=84.0),
        ]
        c, a, highs, n = _build_consensus(forecasts, "NYC", "2026-06-01")
        assert c == pytest.approx(82.0)
        assert n == 2
        assert a > 0

    def test_mismatched_city_returns_none(self):
        forecasts = [_fresh_forecast(city="CHI")]
        c, a, highs, n = _build_consensus(forecasts, "NYC", "2026-06-01")
        assert c is None
        assert n == 0


class TestProbAboveThreshold:
    def test_consensus_above_threshold(self):
        p = _prob_above_threshold(85.0, 80.0, 5.0)
        assert p > 0.5

    def test_consensus_below_threshold(self):
        p = _prob_above_threshold(75.0, 80.0, 5.0)
        assert p < 0.5

    def test_consensus_equals_threshold(self):
        p = _prob_above_threshold(80.0, 80.0, 5.0)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_zero_std(self):
        assert _prob_above_threshold(85.0, 80.0, 0.0) == 1.0
        assert _prob_above_threshold(75.0, 80.0, 0.0) == 0.0

    def test_probability_in_range(self):
        for consensus in range(60, 100, 5):
            p = _prob_above_threshold(float(consensus), 80.0, 5.0)
            assert 0.0 <= p <= 1.0


class TestForecastStaleness:
    def test_fresh_forecast_not_stale(self):
        forecasts = [_fresh_forecast(hours_ago=1.0)]
        assert _is_forecast_stale(forecasts, "NYC", "2026-06-01", max_hours=6.0) is False

    def test_old_forecast_is_stale(self):
        forecasts = [_fresh_forecast(hours_ago=8.0)]
        assert _is_forecast_stale(forecasts, "NYC", "2026-06-01", max_hours=6.0) is True

    def test_empty_forecasts_is_stale(self):
        assert _is_forecast_stale([], "NYC", "2026-06-01", max_hours=6.0) is True

    def test_wrong_city_is_stale(self):
        forecasts = [_fresh_forecast(city="CHI", hours_ago=1.0)]
        assert _is_forecast_stale(forecasts, "NYC", "2026-06-01", max_hours=6.0) is True


# --------------------------------------------------------------------------- #
# Phase 1.3 — Shadow strategies
# --------------------------------------------------------------------------- #

class TestConvergenceExitStrategy:
    def test_is_shadow(self):
        assert ConvergenceExitStrategy.is_live is False

    def test_no_entry_signals(self):
        s = ConvergenceExitStrategy()
        signals = s.generate_signals([_market()], [_fresh_forecast()], _params())
        assert signals == []

    def test_exit_when_converged(self):
        s = ConvergenceExitStrategy()
        # fair_value ≈ 0.5 (consensus = threshold), current_price = 0.51 → gap < 0.03
        forecasts = [_fresh_forecast(high_f=80.0)]
        pos = _position(current_price=0.51, high_f=80.0)
        actions = s.manage_positions([pos], forecasts, _params())
        assert actions[0]["action"] == "exit"
        assert actions[0].get("is_shadow") is True

    def test_hold_when_not_converged(self):
        s = ConvergenceExitStrategy()
        forecasts = [_fresh_forecast(high_f=90.0)]  # fair_value ≈ 0.97
        pos = _position(current_price=0.50, high_f=80.0)  # gap >> 0.03
        actions = s.manage_positions([pos], forecasts, _params())
        assert actions[0]["action"] == "hold"


class TestModelReleaseStrategy:
    def test_is_shadow(self):
        assert ModelReleaseStrategy.is_live is False

    def test_signals_on_large_run_delta(self):
        s = ModelReleaseStrategy()
        # Two runs: old = 75°F, new = 82°F → delta = 7°F > 3°F
        forecasts = [
            _fresh_forecast(model="GFS", high_f=75.0, run_id="2026060100"),
            _fresh_forecast(model="GFS", high_f=82.0, run_id="2026060112"),
        ]
        markets = [_market(market_price=0.40, high_f=80.0)]
        signals = s.generate_signals(markets, forecasts, _params(min_executable_edge=0.0))
        assert len(signals) == 1
        assert signals[0].is_shadow is True

    def test_no_signal_when_delta_small(self):
        s = ModelReleaseStrategy()
        forecasts = [
            _fresh_forecast(model="GFS", high_f=80.0, run_id="2026060100"),
            _fresh_forecast(model="GFS", high_f=81.0, run_id="2026060112"),
        ]
        markets = [_market()]
        signals = s.generate_signals(markets, forecasts, _params())
        assert len(signals) == 0

    def test_no_signal_with_single_run(self):
        s = ModelReleaseStrategy()
        forecasts = [_fresh_forecast(model="GFS", high_f=85.0, run_id="2026060112")]
        markets = [_market()]
        signals = s.generate_signals(markets, forecasts, _params(min_executable_edge=0.0))
        assert len(signals) == 0


class TestDisagreementStrategy:
    def test_is_shadow(self):
        assert DisagreementStrategy.is_live is False

    def test_no_signal_without_trade_history(self):
        s = DisagreementStrategy()
        forecasts = [
            _fresh_forecast(model="GFS", high_f=72.0),
            _fresh_forecast(model="ECMWF", high_f=85.0),
        ]
        markets = [_market()]
        # No trade history → should not signal
        signals = s.generate_signals(markets, forecasts, _params(min_executable_edge=0.0))
        assert len(signals) == 0

    def test_signals_with_sufficient_history(self):
        s = DisagreementStrategy()
        forecasts = [
            _fresh_forecast(model="GFS", high_f=72.0),
            _fresh_forecast(model="ECMWF", high_f=86.0),  # large spread
        ]
        markets = [_market(market_price=0.40)]
        # GFS has 25 trades and lower Brier → best model
        model_accuracy = {"NYC": {"GFS": 0.10, "ECMWF": 0.18}}
        trade_counts = {"NYC": {"GFS": 25, "ECMWF": 25}}

        signals = s.generate_signals(
            markets, forecasts, _params(min_executable_edge=0.0),
            model_accuracy=model_accuracy,
            trade_counts=trade_counts,
        )
        assert len(signals) == 1
        assert signals[0].is_shadow is True

    def test_no_signal_when_models_agree(self):
        s = DisagreementStrategy()
        # Spread < 4°F
        forecasts = [
            _fresh_forecast(model="GFS", high_f=80.0),
            _fresh_forecast(model="ECMWF", high_f=81.0),
        ]
        markets = [_market()]
        model_accuracy = {"NYC": {"GFS": 0.10, "ECMWF": 0.18}}
        trade_counts = {"NYC": {"GFS": 25, "ECMWF": 25}}
        signals = s.generate_signals(
            markets, forecasts, _params(min_executable_edge=0.0),
            model_accuracy=model_accuracy, trade_counts=trade_counts,
        )
        assert len(signals) == 0


# --------------------------------------------------------------------------- #
# Phase 1.4 — Position Lifecycle Engine
# --------------------------------------------------------------------------- #

class TestPositionStatus:
    def test_terminal_states_defined(self):
        assert PositionStatus.WON in TERMINAL_STATES
        assert PositionStatus.LOST in TERMINAL_STATES
        assert PositionStatus.EXITED_CONVERGENCE in TERMINAL_STATES
        assert PositionStatus.EXITED_STOP in TERMINAL_STATES
        assert PositionStatus.EXITED_PRE_SETTLEMENT in TERMINAL_STATES

    def test_non_terminal_states(self):
        assert PositionStatus.OPENED not in TERMINAL_STATES
        assert PositionStatus.HOLDING not in TERMINAL_STATES


class TestIsValidTransition:
    def test_opened_to_holding(self):
        assert is_valid_transition(PositionStatus.OPENED, PositionStatus.HOLDING)

    def test_holding_to_terminal(self):
        for terminal in TERMINAL_STATES:
            assert is_valid_transition(PositionStatus.HOLDING, terminal)

    def test_terminal_to_anything_invalid(self):
        for terminal in TERMINAL_STATES:
            assert not is_valid_transition(terminal, PositionStatus.HOLDING)

    def test_self_transition_always_valid(self):
        for status in PositionStatus:
            assert is_valid_transition(status, status)


class TestProcessPosition:
    def test_opened_transitions_to_holding(self):
        pos = _position(status="OPENED")
        forecasts = [_fresh_forecast()]
        action = process_position(pos, forecasts, params=_params())
        assert action.next_status == PositionStatus.HOLDING
        assert action.should_execute is False

    def test_terminal_position_unchanged(self):
        pos = _position(status="WON")
        action = process_position(pos, [], params=_params())
        assert action.next_status == PositionStatus.WON
        assert action.should_execute is False

    def test_settlement_won(self):
        pos = _position(status="HOLDING")
        action = process_position(
            pos, [], settlement_result={"won": True, "actual_high_f": 85.0}, params=_params()
        )
        assert action.next_status == PositionStatus.WON

    def test_settlement_lost(self):
        pos = _position(status="HOLDING")
        action = process_position(
            pos, [], settlement_result={"won": False, "actual_high_f": 75.0}, params=_params()
        )
        assert action.next_status == PositionStatus.LOST

    def test_exit_on_forecast_reversal(self):
        # Forecast flips: was 88°F, now 65°F → prob collapses, edge < -0.05
        pos = _position(status="HOLDING", side="YES", current_price=0.70, high_f=80.0)
        forecasts = [_fresh_forecast(high_f=65.0)]  # big reversal
        action = process_position(pos, forecasts, params=_params())
        assert action.next_status == PositionStatus.EXITED_STOP
        assert action.reason == "forecast_reversal"
        assert action.should_execute is True

    def test_exit_on_stale_forecast(self):
        pos = _position(status="HOLDING")
        forecasts = [_fresh_forecast(hours_ago=10.0)]
        action = process_position(pos, forecasts, params=_params(stale_forecast_hours=6.0))
        assert action.next_status == PositionStatus.EXITED_STOP
        assert action.reason == "stale_forecast"

    def test_hold_when_all_good(self):
        pos = _position(status="HOLDING", current_price=0.52, entry_price=0.45)
        forecasts = [_fresh_forecast(high_f=85.0, hours_ago=1.0)]
        action = process_position(pos, forecasts, params=_params())
        assert action.next_status == PositionStatus.HOLDING
        assert action.should_execute is False

    def test_returns_lifecycle_action_type(self):
        pos = _position(status="HOLDING")
        forecasts = [_fresh_forecast(hours_ago=1.0)]
        action = process_position(pos, forecasts, params=_params())
        assert isinstance(action, LifecycleAction)


class TestRunLifecycleCycle:
    def test_processes_all_positions(self):
        positions = [_position(pos_id=i, status="HOLDING") for i in range(3)]
        forecasts = [_fresh_forecast(hours_ago=1.0)]
        actions = run_lifecycle_cycle(positions, forecasts, params=_params())
        assert len(actions) == 3

    def test_applies_settlement_results(self):
        positions = [_position(pos_id=1, status="HOLDING")]
        sr = {1: {"won": True, "actual_high_f": 85.0}}
        actions = run_lifecycle_cycle(positions, [], settlement_results=sr, params=_params())
        assert actions[0].next_status == PositionStatus.WON

    def test_empty_positions_returns_empty(self):
        actions = run_lifecycle_cycle([], [], params=_params())
        assert actions == []


# --------------------------------------------------------------------------- #
# Shadow strategy execution gating
# --------------------------------------------------------------------------- #

class TestShadowExecutionGating:
    """Shadow strategies must never set should_execute=True in lifecycle engine,
    and their signals must always have is_shadow=True."""

    def test_convergence_exit_signals_marked_shadow(self):
        s = ConvergenceExitStrategy()
        forecasts = [_fresh_forecast(high_f=80.0)]
        pos = _position(current_price=0.51)
        actions = s.manage_positions([pos], forecasts, _params())
        for a in actions:
            if a["action"] == "exit":
                assert a.get("is_shadow") is True

    def test_model_release_signals_are_shadow(self):
        s = ModelReleaseStrategy()
        forecasts = [
            _fresh_forecast(model="GFS", high_f=74.0, run_id="2026060100"),
            _fresh_forecast(model="GFS", high_f=82.0, run_id="2026060112"),
        ]
        markets = [_market(market_price=0.35)]
        signals = s.generate_signals(markets, forecasts, _params(min_executable_edge=0.0))
        for sig in signals:
            assert sig.is_shadow is True

    def test_disagreement_signals_are_shadow(self):
        s = DisagreementStrategy()
        forecasts = [
            _fresh_forecast(model="GFS", high_f=72.0),
            _fresh_forecast(model="ECMWF", high_f=86.0),
        ]
        markets = [_market(market_price=0.40)]
        accuracy = {"NYC": {"GFS": 0.10, "ECMWF": 0.18}}
        counts = {"NYC": {"GFS": 25, "ECMWF": 25}}
        signals = s.generate_signals(
            markets, forecasts, _params(min_executable_edge=0.0),
            model_accuracy=accuracy, trade_counts=counts,
        )
        for sig in signals:
            assert sig.is_shadow is True
