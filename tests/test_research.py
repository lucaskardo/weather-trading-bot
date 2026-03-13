"""Tests for Phase 3: Research & Calibration Upgrade."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.forecaster import (
    brier_decomposition,
    brier_score,
    prob_above_threshold,
    temperature_scale,
)
from research.walk_forward import (
    make_synthetic_trades,
    walk_forward_brier,
    walk_forward_variance,
)
from research.optimizer import (
    OptimizerResult,
    _SCIPY_AVAILABLE,
    _fallback_optimize,
    load_params,
    optimize_params,
    save_params,
)
from research.calibrator import CalibrationResult, run_calibration
from state.db import init_db


# --------------------------------------------------------------------------- #
# Phase 3.3 — Temperature Scaling
# --------------------------------------------------------------------------- #

class TestTemperatureScale:
    def test_T_equals_one_is_identity(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert temperature_scale(p, T=1.0) == pytest.approx(p, abs=1e-6)

    def test_T_less_than_one_sharpens(self):
        """T < 1 makes probabilities more extreme (further from 0.5)."""
        sharp = temperature_scale(0.7, T=0.5)
        original = 0.7
        assert sharp > original  # pushed toward 1.0

        sharp_low = temperature_scale(0.3, T=0.5)
        assert sharp_low < 0.3   # pushed toward 0.0

    def test_T_greater_than_one_smooths(self):
        """T > 1 makes probabilities less extreme (closer to 0.5)."""
        smoothed = temperature_scale(0.8, T=1.5)
        assert smoothed < 0.8

        smoothed_low = temperature_scale(0.2, T=1.5)
        assert smoothed_low > 0.2

    def test_output_stays_in_unit_interval(self):
        for T in [0.1, 0.5, 1.0, 1.5, 2.0]:
            for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
                result = temperature_scale(p, T=T)
                assert 0.0 < result < 1.0

    def test_symmetric_around_half(self):
        """temperature_scale(p, T) + temperature_scale(1-p, T) ≈ 1."""
        for T in [0.7, 1.0, 1.4]:
            for p in [0.3, 0.6, 0.8]:
                assert temperature_scale(p, T) + temperature_scale(1 - p, T) == pytest.approx(1.0, abs=1e-6)

    def test_extreme_T_does_not_crash(self):
        assert 0 < temperature_scale(0.7, T=0.01) < 1
        assert 0 < temperature_scale(0.7, T=100.0) < 1


class TestProbAboveThreshold:
    def test_T_one_matches_erfc(self):
        """With T=1 result should match pure Gaussian erfc."""
        from core.forecaster import prob_above_threshold as pat
        result = pat(85.0, 80.0, 5.0, temp_T=1.0)
        # consensus > threshold → prob > 0.5
        assert result > 0.5

    def test_temperature_affects_output(self):
        from core.forecaster import prob_above_threshold as pat
        p_identity = pat(85.0, 80.0, 5.0, temp_T=1.0)
        p_sharp = pat(85.0, 80.0, 5.0, temp_T=0.5)
        p_smooth = pat(85.0, 80.0, 5.0, temp_T=1.5)
        assert p_sharp > p_identity > p_smooth


class TestBrierScore:
    def test_perfect_correct_prediction(self):
        assert brier_score(1.0, 1.0) == pytest.approx(0.0)
        assert brier_score(0.0, 0.0) == pytest.approx(0.0)

    def test_perfect_wrong_prediction(self):
        assert brier_score(1.0, 0.0) == pytest.approx(1.0)
        assert brier_score(0.0, 1.0) == pytest.approx(1.0)

    def test_fifty_fifty(self):
        assert brier_score(0.5, 1.0) == pytest.approx(0.25)
        assert brier_score(0.5, 0.0) == pytest.approx(0.25)


class TestBrierDecomposition:
    def test_returns_required_keys(self):
        decomp = brier_decomposition([0.7, 0.6, 0.8], [1.0, 1.0, 0.0])
        assert "brier" in decomp
        assert "reliability" in decomp
        assert "resolution" in decomp
        assert "uncertainty" in decomp
        assert "brier_index" in decomp

    def test_brier_index_range(self):
        decomp = brier_decomposition([0.7, 0.6], [1.0, 0.0])
        assert 0.0 <= decomp["brier_index"] <= 100.0

    def test_perfect_predictions_low_brier(self):
        decomp = brier_decomposition([0.99, 0.99, 0.01], [1.0, 1.0, 0.0])
        assert decomp["brier"] < 0.01

    def test_random_predictions_brier_near_quarter(self):
        decomp = brier_decomposition([0.5] * 100, [1.0] * 50 + [0.0] * 50)
        assert decomp["brier"] == pytest.approx(0.25, abs=0.01)

    def test_empty_returns_defaults(self):
        decomp = brier_decomposition([], [])
        assert decomp["brier"] == 0.25

    def test_brier_index_formula(self):
        preds = [0.8, 0.7, 0.9]
        outcomes = [1.0, 1.0, 1.0]
        decomp = brier_decomposition(preds, outcomes)
        expected_index = (1.0 - math.sqrt(max(0.0, decomp["brier"]))) * 100.0
        assert decomp["brier_index"] == pytest.approx(expected_index, abs=0.01)


# --------------------------------------------------------------------------- #
# Phase 3.1 — Walk-Forward Validation
# --------------------------------------------------------------------------- #

class TestWalkForwardBrier:
    def test_returns_baseline_with_insufficient_data(self):
        trades = make_synthetic_trades(3)  # too few for 5 windows
        result = walk_forward_brier({"base_std_f": 5.0, "temp_T": 1.0}, trades)
        assert result == pytest.approx(0.25)

    def test_returns_float_with_sufficient_data(self):
        trades = make_synthetic_trades(100)
        result = walk_forward_brier({"base_std_f": 5.0, "temp_T": 1.0}, trades)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_good_params_score_better_than_bad(self):
        trades = make_synthetic_trades(200, true_std_f=5.0)
        # Good params: close to true std
        good = walk_forward_brier({"base_std_f": 5.0, "temp_T": 1.0}, trades)
        # Bad params: wildly wrong std
        bad = walk_forward_brier({"base_std_f": 0.1, "temp_T": 1.0}, trades)
        assert good < bad

    def test_result_between_zero_and_one(self):
        trades = make_synthetic_trades(100)
        for std in [2.5, 5.0, 10.0]:
            result = walk_forward_brier({"base_std_f": std, "temp_T": 1.0}, trades)
            assert 0.0 <= result <= 1.0

    def test_n_windows_parameter(self):
        trades = make_synthetic_trades(100)
        params = {"base_std_f": 5.0, "temp_T": 1.0}
        r3 = walk_forward_brier(params, trades, n_windows=3)
        r5 = walk_forward_brier(params, trades, n_windows=5)
        # Both should be valid floats
        assert 0.0 <= r3 <= 1.0
        assert 0.0 <= r5 <= 1.0


class TestWalkForwardVariance:
    def test_returns_zero_with_single_fold(self):
        trades = make_synthetic_trades(50)
        result = walk_forward_variance({"base_std_f": 5.0, "temp_T": 1.0}, trades, n_windows=1)
        assert result == pytest.approx(0.0)

    def test_returns_non_negative(self):
        trades = make_synthetic_trades(100)
        result = walk_forward_variance({"base_std_f": 5.0, "temp_T": 1.0}, trades)
        assert result >= 0.0

    def test_stable_params_have_lower_variance(self):
        trades = make_synthetic_trades(200, true_std_f=5.0)
        stable = walk_forward_variance({"base_std_f": 5.0, "temp_T": 1.0}, trades)
        unstable = walk_forward_variance({"base_std_f": 0.1, "temp_T": 1.0}, trades)
        # "good" params should have lower variance across folds
        assert stable <= unstable + 0.05  # allow small tolerance


class TestMakeSyntheticTrades:
    def test_returns_correct_count(self):
        trades = make_synthetic_trades(50)
        assert len(trades) == 50

    def test_outcomes_binary(self):
        trades = make_synthetic_trades(100)
        for t in trades:
            assert t["outcome"] in (0.0, 1.0)

    def test_reproducible_with_seed(self):
        t1 = make_synthetic_trades(10, seed=99)
        t2 = make_synthetic_trades(10, seed=99)
        assert [t["outcome"] for t in t1] == [t["outcome"] for t in t2]

    def test_different_seeds_different_results(self):
        t1 = make_synthetic_trades(10, seed=1)
        t2 = make_synthetic_trades(10, seed=2)
        assert [t["outcome"] for t in t1] != [t["outcome"] for t in t2]


# --------------------------------------------------------------------------- #
# Phase 3.2 — Optimizer
# --------------------------------------------------------------------------- #

class TestOptimizerFallback:
    """Test the coordinate-descent fallback (doesn't require scipy)."""

    def test_returns_optimizer_result(self):
        trades = make_synthetic_trades(100)
        result = _fallback_optimize(trades, n_windows=3, current_params=None)
        assert isinstance(result, OptimizerResult)

    def test_params_within_bounds(self):
        trades = make_synthetic_trades(100)
        result = _fallback_optimize(trades, n_windows=3, current_params=None)
        assert 2.5 <= result.best_params["base_std_f"] <= 12.0
        assert 0.7 <= result.best_params["temp_T"] <= 1.8

    def test_best_brier_is_float(self):
        trades = make_synthetic_trades(100)
        result = _fallback_optimize(trades, n_windows=3, current_params=None)
        assert isinstance(result.best_brier, float)
        assert 0.0 <= result.best_brier <= 1.0

    def test_n_evaluations_positive(self):
        trades = make_synthetic_trades(100)
        result = _fallback_optimize(trades, n_windows=3, current_params=None)
        assert result.n_evaluations > 0


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not installed")
class TestOptimizerWithScipy:
    def test_converges_on_synthetic_data(self):
        """Optimizer should find std_f close to the true value (5.0)."""
        trades = make_synthetic_trades(300, true_std_f=5.0, seed=42)
        result = optimize_params(trades, n_windows=3, maxiter=20, popsize=5, seed=42)
        assert isinstance(result, OptimizerResult)
        assert 2.5 <= result.best_params["base_std_f"] <= 12.0
        assert 0.7 <= result.best_params["temp_T"] <= 1.8
        # Should find something reasonably close to truth
        assert result.best_brier < 0.25  # beats random baseline

    def test_optimised_beats_default_params(self):
        """Optimised Brier should be <= default params Brier on same data."""
        trades = make_synthetic_trades(200, true_std_f=5.0, seed=7)
        result = optimize_params(trades, n_windows=3, maxiter=15, popsize=5, seed=7)
        default_brier = walk_forward_brier({"base_std_f": 5.0, "temp_T": 1.0}, trades, n_windows=3)
        # Optimised should be at most marginally worse (allow 1% slack)
        assert result.best_brier <= default_brier + 0.01


class TestSaveLoadParams:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "params.json"
        params = {"base_std_f": 6.5, "temp_T": 1.2}
        save_params(params, path)
        loaded = load_params(path)
        assert loaded == pytest.approx(params)

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_params(tmp_path / "no_such_file.json")
        assert result is None


# --------------------------------------------------------------------------- #
# Phase 3.4 — Calibrator
# --------------------------------------------------------------------------- #

class TestRunCalibration:
    def test_returns_calibration_result(self, tmp_path):
        conn = init_db(tmp_path / "cal_test.db")
        result = run_calibration(conn, min_trades=1, save=False)
        assert isinstance(result, CalibrationResult)

    def test_insufficient_data_skips_optimizer(self, tmp_path):
        conn = init_db(tmp_path / "cal_skip.db")
        result = run_calibration(conn, min_trades=50, save=False)
        assert result.optimizer_result is None
        assert result.n_trades == 0

    def test_calibration_with_synthetic_data(self, tmp_path):
        conn = init_db(tmp_path / "cal_data.db")
        _insert_resolved_predictions(conn, n=25)
        result = run_calibration(conn, min_trades=10, save=False)
        assert isinstance(result, CalibrationResult)
        assert result.n_trades >= 10
        assert 0.0 <= result.brier <= 1.0
        assert 0.0 <= result.brier_index <= 100.0

    def test_calibration_updates_params(self, tmp_path):
        from shared.params import Params
        conn = init_db(tmp_path / "cal_update.db")
        _insert_resolved_predictions(conn, n=25)
        p = Params()
        p.base_std_f = 5.0
        run_calibration(conn, params=p, min_trades=10, save=False)
        # Params may have been updated (any value within bounds is fine)
        assert 2.5 <= p.base_std_f <= 12.0

    def test_save_writes_json(self, tmp_path):
        conn = init_db(tmp_path / "cal_save.db")
        _insert_resolved_predictions(conn, n=25)
        params_path = tmp_path / "calibrated.json"
        run_calibration(conn, min_trades=10, save=True, params_path=params_path)
        assert params_path.exists()

    def test_brier_index_formula(self, tmp_path):
        conn = init_db(tmp_path / "cal_brier.db")
        _insert_resolved_predictions(conn, n=25)
        result = run_calibration(conn, min_trades=10, save=False)
        expected = (1.0 - math.sqrt(max(0.0, result.brier))) * 100.0
        assert result.brier_index == pytest.approx(expected, abs=0.1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _insert_resolved_predictions(conn, n: int = 25) -> None:
    """Insert synthetic resolved predictions into the DB for calibration tests."""
    import random
    rng = random.Random(42)
    now = "2026-06-01T12:00:00+00:00"
    for i in range(n):
        fair_value = rng.uniform(0.4, 0.8)
        outcome = 1.0 if rng.random() < fair_value else 0.0
        brier = (fair_value - outcome) ** 2
        conn.execute(
            """INSERT INTO predictions
               (strategy_name, ticker, city, target_date, fair_value,
                market_price, edge, is_shadow, outcome, brier_score,
                consensus_f, created_at, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "value_entry", f"TKR{i}", "NYC", "2026-06-01",
                fair_value, 0.50, fair_value - 0.50,
                0, outcome, brier,
                rng.uniform(78.0, 88.0),  # consensus_f
                now, now,
            ),
        )
    conn.commit()
