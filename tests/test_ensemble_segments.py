from __future__ import annotations

from dataclasses import replace

import pytest

from core.forecaster import compute_fair_value, empirical_prob_for_market
from research.calibrator import run_calibration
from shared.params import PARAMS
from shared.types import ModelForecast
from state.db import init_db


def test_empirical_prob_for_market_band():
    samples = [74.0, 74.4, 75.2, 76.4, 77.0]
    prob = empirical_prob_for_market(samples, "band", 76.0, 74.0)
    assert prob == pytest.approx(4 / 5)


def test_compute_fair_value_prefers_true_ensemble_members():
    params = replace(PARAMS, use_empirical_ensemble=True, use_monte_carlo=True)
    forecasts = [
        ModelForecast(
            model_name="ENS", city="NYC", target_date="2099-07-01",
            predicted_high_f=75.0, ensemble_members_f=[70.0, 71.0, 72.0, 80.0],
        ),
        ModelForecast(
            model_name="ENS2", city="NYC", target_date="2099-07-01",
            predicted_high_f=75.0, ensemble_members_f=[81.0, 82.0, 83.0, 84.0],
        ),
    ]
    fv, consensus, std_f, n_models = compute_fair_value(
        forecasts, "NYC", "2099-07-01", "above", 80.0, None, params, conn=None, use_mc=True
    )
    assert n_models == 2
    assert consensus == pytest.approx(75.0)
    # 5 of 8 members are >= 79.5 after threshold rounding
    assert fv == pytest.approx(5 / 8, abs=1e-6)
    assert std_f > 0


def test_calibration_persists_segment_metrics(tmp_path):
    conn = init_db(tmp_path / "segmented.db")
    now = "2026-06-01T12:00:00+00:00"
    for i, (mtype, high_f, low_f) in enumerate([("above", 80.0, None), ("below", 70.0, None), ("band", 76.0, 74.0)]):
        market_id = f"m{i}"
        conn.execute(
            "INSERT INTO markets (id, ticker, city, target_date, market_type, high_f, low_f, exchange, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (market_id, f"TKR{i}", "NYC", "2026-06-01", mtype, high_f, low_f, "kalshi", "open"),
        )
        for j in range(5):
            fair_value = 0.4 + 0.1 * ((i + j) % 3)
            outcome = 1.0 if fair_value >= 0.5 else 0.0
            conn.execute(
                """INSERT INTO predictions
                   (strategy_name, market_id, ticker, city, target_date, fair_value, market_price, edge, is_shadow, outcome, brier_score, consensus_f, agreement, created_at, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("value_entry", market_id, f"TKR{i}", "NYC", "2026-06-01", fair_value, 0.5, fair_value - 0.5, 0, outcome, (fair_value - outcome) ** 2, 75.0 + i, 1.0 + i, now, now),
            )
    conn.commit()
    result = run_calibration(conn, min_trades=1, save=True, params_path=tmp_path / "params.json")
    assert result.segment_metrics
    rows = conn.execute("SELECT segment_kind, segment_value, trade_count FROM calibration_segments").fetchall()
    assert rows
    kinds = {r[0] for r in rows}
    assert {"market_type", "lead_bucket", "regime", "city"}.issubset(kinds)


from clients.weather import _parse_open_meteo


def test_parse_open_meteo_extracts_member_keys():
    data = {
        "daily": {
            "time": ["2026-06-01"],
            "temperature_2m_max": [80.0],
            "temperature_2m_min": [68.0],
            "temperature_2m_max_member_001": [79.0],
            "temperature_2m_max_member_002": [81.5],
        }
    }
    f = _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake")
    assert f.ensemble_members_f == [79.0, 81.5]


def test_segment_calibration_profiles_adjust_fair_value(tmp_path):
    conn = init_db(tmp_path / "profiles.db")
    conn.execute(
        "INSERT INTO calibration_profiles (segment_kind, segment_value, trade_count, avg_brier, avg_outcome, avg_prediction, prob_adjustment) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("market_type", "above", 12, 0.12, 0.7, 0.5, 0.10),
    )
    conn.commit()
    params = replace(PARAMS, use_empirical_ensemble=False, use_monte_carlo=False)
    forecasts = [
        ModelForecast(model_name="GFS", city="NYC", target_date="2099-07-01", predicted_high_f=80.0),
        ModelForecast(model_name="ECMWF", city="NYC", target_date="2099-07-01", predicted_high_f=80.0),
    ]
    fv_no_conn, *_ = compute_fair_value(forecasts, "NYC", "2099-07-01", "above", 80.0, None, params, conn=None, use_mc=False)
    fv_conn, *_ = compute_fair_value(forecasts, "NYC", "2099-07-01", "above", 80.0, None, params, conn=conn, use_mc=False)
    assert fv_conn > fv_no_conn
    assert fv_conn == pytest.approx(min(0.99, fv_no_conn + 0.10), rel=1e-6)


def test_city_segment_profile_adjusts_fair_value(tmp_path):
    conn = init_db(tmp_path / "city_profiles.db")
    conn.execute(
        "INSERT INTO calibration_profiles (segment_kind, segment_value, trade_count, avg_brier, avg_outcome, avg_prediction, prob_adjustment) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("city", "NYC", 20, 0.10, 0.65, 0.50, 0.08),
    )
    conn.commit()
    params = replace(PARAMS, use_empirical_ensemble=False, use_monte_carlo=False)
    forecasts = [
        ModelForecast(model_name="GFS", city="NYC", target_date="2099-07-01", predicted_high_f=80.0),
        ModelForecast(model_name="ECMWF", city="NYC", target_date="2099-07-01", predicted_high_f=80.0),
    ]
    fv_no_conn, *_ = compute_fair_value(forecasts, "NYC", "2099-07-01", "above", 80.0, None, params, conn=None, use_mc=False)
    fv_conn, *_ = compute_fair_value(forecasts, "NYC", "2099-07-01", "above", 80.0, None, params, conn=conn, use_mc=False)
    assert fv_conn > fv_no_conn
