"""
Tests for Phase B profitability upgrades:
  - B1: research/bias_correction.py
  - B2: Kalshi fee schedule (execution/orderbook.py)
  - B3: clients/hrrr.py
  - B4: research/autoresearch.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from state.db import init_db
from research.bias_correction import apply_bias, get_bias_summary, learn_biases, MIN_OBS
from execution.orderbook import kalshi_fee_rate, get_executable_price
from clients.hrrr import fetch_hrrr_nowcast, hrrr_signal_edge, _compute_nowcast
from research.autoresearch import ExperimentRegistry, PROMOTION_THRESHOLD
from shared.params import Params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "test.db")


def _seed_forecasts_and_settlements(conn, city, model, pairs):
    """pairs = list of (predicted_high_f, actual_high_f)"""
    target_date = "2025-07-01"
    for i, (pred, actual) in enumerate(pairs):
        conn.execute(
            """INSERT INTO forecasts
               (city, target_date, model_name, predicted_high_f)
               VALUES (?,?,?,?)""",
            (city, f"2025-{7+i:02d}-01", model, pred),
        )
        conn.execute(
            """INSERT OR IGNORE INTO settlement_cache
               (city, target_date, actual_high_f, station)
               VALUES (?,?,?,?)""",
            (city, f"2025-{7+i:02d}-01", actual, "KLGA"),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# B1: Bias correction
# ---------------------------------------------------------------------------

class TestLearnBiases:
    def test_returns_empty_with_no_data(self, conn):
        result = learn_biases(conn)
        assert result == {}

    def test_below_min_obs_excluded(self, conn):
        # Only 5 observations — below MIN_OBS=10
        _seed_forecasts_and_settlements(conn, "NYC", "GFS",
                                        [(72.0, 70.0)] * 5)
        result = learn_biases(conn, min_obs=10)
        assert "NYC" not in result

    def test_bias_computed_correctly(self, conn):
        # Predicted always 2°F above actual → bias = +2.0
        _seed_forecasts_and_settlements(conn, "NYC", "GFS",
                                        [(72.0, 70.0)] * 12)
        result = learn_biases(conn, min_obs=10)
        assert "NYC" in result
        assert abs(result["NYC"]["GFS"] - 2.0) < 1e-6

    def test_cold_bias(self, conn):
        # Predicted 1.5°F below actual → bias = -1.5
        _seed_forecasts_and_settlements(conn, "CHI", "ECMWF",
                                        [(68.5, 70.0)] * 12)
        result = learn_biases(conn, min_obs=10)
        assert abs(result["CHI"]["ECMWF"] - (-1.5)) < 1e-6

    def test_multiple_models(self, conn):
        _seed_forecasts_and_settlements(conn, "NYC", "GFS",
                                        [(72.0, 70.0)] * 12)
        _seed_forecasts_and_settlements(conn, "NYC", "ECMWF",
                                        [(69.0, 70.0)] * 12)
        result = learn_biases(conn, min_obs=10)
        assert abs(result["NYC"]["GFS"] - 2.0) < 1e-6
        assert abs(result["NYC"]["ECMWF"] - (-1.0)) < 1e-6


class TestApplyBias:
    def test_no_bias_entry_returns_raw(self):
        result = apply_bias(75.0, "NYC", "GFS", {})
        assert result == 75.0

    def test_applies_positive_bias(self):
        table = {"NYC": {"GFS": 2.0}}
        result = apply_bias(72.0, "NYC", "GFS", table)
        assert abs(result - 70.0) < 1e-9

    def test_applies_negative_bias(self):
        table = {"CHI": {"ECMWF": -1.5}}
        result = apply_bias(68.5, "CHI", "ECMWF", table)
        assert abs(result - 70.0) < 1e-9

    def test_missing_model_returns_raw(self):
        table = {"NYC": {"GFS": 2.0}}
        result = apply_bias(75.0, "NYC", "ECMWF", table)
        assert result == 75.0

    def test_missing_city_returns_raw(self):
        table = {"NYC": {"GFS": 2.0}}
        result = apply_bias(75.0, "CHI", "GFS", table)
        assert result == 75.0


class TestGetBiasSummary:
    def test_returns_list(self, conn):
        result = get_bias_summary(conn)
        assert isinstance(result, list)

    def test_includes_required_fields(self, conn):
        _seed_forecasts_and_settlements(conn, "NYC", "GFS",
                                        [(72.0, 70.0)] * 12)
        result = get_bias_summary(conn, min_obs=10)
        assert len(result) == 1
        assert {"city", "model_name", "n_obs", "bias_f", "mae_f"}.issubset(result[0].keys())


# ---------------------------------------------------------------------------
# B2: Kalshi fee schedule
# ---------------------------------------------------------------------------

class TestKalshiFeeRate:
    def test_near_50_cents_highest(self):
        fee_50 = kalshi_fee_rate(0.50)
        fee_10 = kalshi_fee_rate(0.10)
        fee_90 = kalshi_fee_rate(0.90)
        # Near 50¢ should cost more than extremes
        assert fee_50 > fee_10
        assert fee_50 > fee_90

    def test_symmetric_around_50(self):
        # Fee for 0.30 and 0.70 should be equal (symmetric)
        assert abs(kalshi_fee_rate(0.30) - kalshi_fee_rate(0.70)) < 1e-9

    def test_extreme_prices_lowest(self):
        fee_05 = kalshi_fee_rate(0.05)
        fee_95 = kalshi_fee_rate(0.95)
        fee_50 = kalshi_fee_rate(0.50)
        assert fee_05 < fee_50
        assert fee_95 < fee_50

    def test_returns_positive(self):
        for price in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
            assert kalshi_fee_rate(price) > 0

    def test_fee_applied_in_get_executable_price(self):
        # At 50¢ market, fee should be higher than flat 1%
        from execution.orderbook import OrderbookLevel
        info = get_executable_price(
            "TICKER", "YES", 100.0, 0.50, orderbook=None
        )
        # Kalshi fee at 50¢ is 0.035 (effective_cents=50, rate=7% → 50/100*7%=3.5%)
        assert info.fees_est > 0.01  # must be higher than old flat 1%


# ---------------------------------------------------------------------------
# B3: HRRR
# ---------------------------------------------------------------------------

class TestHrrrSignalEdge:
    def test_above_market_positive_edge(self):
        # HRRR says 75°F, threshold is 70°F → high prob → positive edge vs 0.5
        edge = hrrr_signal_edge(
            hrrr_high_f=75.0, threshold_f=70.0,
            base_std_f=5.0, market_price=0.50
        )
        assert edge > 0

    def test_below_market_negative_edge(self):
        # HRRR says 65°F, threshold is 70°F → low prob → negative edge vs 0.5
        edge = hrrr_signal_edge(
            hrrr_high_f=65.0, threshold_f=70.0,
            base_std_f=5.0, market_price=0.50
        )
        assert edge < 0

    def test_zero_edge_when_aligned(self):
        # HRRR prob ≈ market price → edge ≈ 0
        import math
        # When HRRR exactly agrees with market (70°F vs 70°F threshold, std=5 → prob≈0.5)
        edge = hrrr_signal_edge(70.0, 70.0, 5.0, 0.5)
        assert abs(edge) < 0.01


class TestComputeNowcast:
    def _make_raw(self, target_date, temps):
        hours = [f"{target_date}T{h:02d}:00" for h in range(24)]
        return {
            "hourly": {
                "time": hours,
                "temperature_2m": temps,
            }
        }

    def test_future_day_returns_max(self):
        future = "2099-07-01"  # guaranteed future
        temps = [60.0 + i for i in range(24)]
        raw = self._make_raw(future, temps)
        result = _compute_nowcast(raw, future)
        assert result is not None
        assert result["hrrr_high_f"] == max(temps)
        assert result["source"] == "hrrr_forecast_only"

    def test_empty_temps_returns_none(self):
        result = _compute_nowcast({"hourly": {"time": [], "temperature_2m": []}}, "2025-07-01")
        assert result is None

    def test_required_fields_present(self):
        future = "2099-07-01"
        raw = self._make_raw(future, [70.0] * 24)
        result = _compute_nowcast(raw, future)
        assert {"hrrr_high_f", "observed_max_f", "hours_remaining", "source"}.issubset(
            result.keys()
        )


class TestFetchHrrrNowcast:
    def test_unknown_city_returns_none(self):
        result = fetch_hrrr_nowcast("UNKNOWN_CITY", "2025-07-01")
        assert result is None

    def test_api_error_returns_none(self):
        session = MagicMock()
        session.get.side_effect = Exception("network error")
        result = fetch_hrrr_nowcast("NYC", "2025-07-01", session=session)
        assert result is None

    def test_valid_response_returns_dict(self):
        session = MagicMock()
        resp = MagicMock()
        future = "2099-07-01"
        resp.json.return_value = {
            "hourly": {
                "time": [f"{future}T{h:02d}:00" for h in range(24)],
                "temperature_2m": [70.0 + h * 0.5 for h in range(24)],
            }
        }
        session.get.return_value = resp
        result = fetch_hrrr_nowcast("NYC", future, session=session)
        assert result is not None
        assert "hrrr_high_f" in result


# ---------------------------------------------------------------------------
# B4: Autoresearch
# ---------------------------------------------------------------------------

def _make_trade_log(n=30, brier=0.15):
    """Create synthetic trade log with approximate target Brier score."""
    import random
    rng = random.Random(42)
    trades = []
    for _ in range(n):
        outcome = rng.choice([0.0, 1.0])
        fair_value = outcome + rng.gauss(0, 0.3)
        fair_value = max(0.01, min(0.99, fair_value))
        trades.append({
            "fair_value": fair_value,
            "outcome": outcome,
            "threshold_f": 70.0,
            "city": "NYC",
            "target_date": "2025-07-01",
        })
    return trades


class TestExperimentRegistry:
    def test_propose_creates_db_row(self, conn):
        registry = ExperimentRegistry(conn)
        exp_id = registry.propose_experiment("test experiment")
        row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["description"] == "test experiment"

    def test_params_stored_as_json(self, conn):
        registry = ExperimentRegistry(conn)
        exp_id = registry.propose_experiment()
        row = conn.execute("SELECT params_json FROM experiments WHERE id=?", (exp_id,)).fetchone()
        params = json.loads(row["params_json"])
        assert len(params) > 0
        assert all(isinstance(v, (int, float)) for v in params.values())

    def test_run_experiment_updates_row(self, conn):
        registry = ExperimentRegistry(conn)
        trade_log = _make_trade_log(30)
        exp_id = registry.propose_experiment()
        result = registry.run_experiment(exp_id, trade_log)
        row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        assert row["status"] == "completed"
        assert row["baseline_brier"] is not None
        assert row["candidate_brier"] is not None
        assert row["trade_count"] == 30

    def test_compare_to_baseline(self, conn):
        registry = ExperimentRegistry(conn)
        trade_log = _make_trade_log(30)
        exp_id = registry.propose_experiment()
        registry.run_experiment(exp_id, trade_log)
        comparison = registry.compare_to_baseline(exp_id)
        assert "improvement_pct" in comparison
        assert "better" in comparison
        assert "promotable" in comparison

    def test_promote_if_better_true_when_improvement(self, conn):
        registry = ExperimentRegistry(conn)
        exp_id = str(__import__("uuid").uuid4())[:8]
        # Manually insert a clearly-winning experiment
        conn.execute(
            """INSERT INTO experiments
               (id, description, params_json, baseline_brier, candidate_brier,
                improvement_pct, trade_count, status)
               VALUES (?,?,?,?,?,?,?,'completed')""",
            (exp_id, "test", json.dumps({"base_std_f": 5.5}),
             0.20, 0.15, 25.0, 30),
        )
        conn.commit()
        promoted = registry.promote_if_better(exp_id)
        assert promoted is True
        row = conn.execute("SELECT status FROM experiments WHERE id=?", (exp_id,)).fetchone()
        assert row["status"] == "promoted"

    def test_promote_if_better_false_when_no_improvement(self, conn):
        registry = ExperimentRegistry(conn)
        exp_id = str(__import__("uuid").uuid4())[:8]
        conn.execute(
            """INSERT INTO experiments
               (id, description, params_json, baseline_brier, candidate_brier,
                improvement_pct, trade_count, status)
               VALUES (?,?,?,?,?,?,?,'completed')""",
            (exp_id, "test", json.dumps({"base_std_f": 5.5}),
             0.20, 0.21, -5.0, 30),
        )
        conn.commit()
        promoted = registry.promote_if_better(exp_id)
        assert promoted is False

    def test_list_experiments(self, conn):
        registry = ExperimentRegistry(conn)
        registry.propose_experiment("exp1")
        registry.propose_experiment("exp2")
        listing = registry.list_experiments()
        assert len(listing) >= 2
        assert all("id" in e for e in listing)

    def test_run_cycle_returns_result(self, conn):
        registry = ExperimentRegistry(conn)
        trade_log = _make_trade_log(30)
        result = registry.run_cycle(trade_log)
        assert "experiment_id" in result
        assert "baseline_brier" in result
        assert "promoted" in result

    def test_compare_nonexistent_returns_error(self, conn):
        registry = ExperimentRegistry(conn)
        result = registry.compare_to_baseline("nonexistent")
        assert "error" in result

    def test_experiments_table_exists(self, conn):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
        ).fetchall()
        assert len(rows) == 1
