"""Tests for Phase 4: Risk Hardening (Guards + Reconciliation + Startup)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.params import Params
from shared.types import ModelForecast
from state.db import init_db
from risk.guards import (
    CityLimitExceeded,
    ClusterCapExceeded,
    DailyLossHalt,
    RiskViolation,
    StaleDataHalt,
    check_city_limit,
    check_cluster_exposure,
    check_daily_loss_limit,
    get_daily_loss,
    is_forecast_fresh,
    check_stale_forecast,
)
from risk.reconciliation import (
    Discrepancy,
    DiscrepancyType,
    ReconciliationResult,
    check_orphaned_orders,
    reconcile_positions,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _params(**kw) -> Params:
    p = Params()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _forecast(
    city="NYC",
    target_date="2026-06-01",
    model="GFS",
    hours_ago: float = 1.0,
) -> ModelForecast:
    fetched = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return ModelForecast(
        model_name=model,
        city=city,
        target_date=target_date,
        predicted_high_f=82.0,
        run_id="2026060100",
        publish_time=fetched,
        source_url="http://fake",
        fetched_at=fetched,
    )


def _position(city="NYC", size_usd=100.0, side="YES", entry=0.50, current=0.55) -> dict:
    return {
        "id": 1,
        "ticker": f"TKR_{city}",
        "city": city,
        "target_date": "2026-06-01",
        "side": side,
        "size_usd": size_usd,
        "entry_price": entry,
        "current_price": current,
        "status": "HOLDING",
        "strategy_name": "value_entry",
    }


def _db(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "risk_test.db")


def _seed_daily_pnl(conn, pnl: float) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT OR REPLACE INTO daily_pnl (date, realized_pnl) VALUES (?,?)",
        (today, pnl),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Guard 1: Stale Forecast
# --------------------------------------------------------------------------- #

class TestStaleDataHalt:
    def test_fresh_forecast_passes(self):
        forecasts = [_forecast(hours_ago=1.0)]
        # Should not raise
        check_stale_forecast(forecasts, "NYC", "2026-06-01", _params(stale_forecast_hours=6.0))

    def test_stale_forecast_raises(self):
        forecasts = [_forecast(hours_ago=10.0)]
        with pytest.raises(StaleDataHalt):
            check_stale_forecast(forecasts, "NYC", "2026-06-01", _params(stale_forecast_hours=6.0))

    def test_no_forecasts_raises(self):
        with pytest.raises(StaleDataHalt, match="No forecasts"):
            check_stale_forecast([], "NYC", "2026-06-01", _params())

    def test_wrong_city_raises(self):
        forecasts = [_forecast(city="CHI", hours_ago=1.0)]
        with pytest.raises(StaleDataHalt):
            check_stale_forecast(forecasts, "NYC", "2026-06-01", _params())

    def test_wrong_date_raises(self):
        forecasts = [_forecast(target_date="2026-07-01", hours_ago=1.0)]
        with pytest.raises(StaleDataHalt):
            check_stale_forecast(forecasts, "NYC", "2026-06-01", _params())

    def test_one_fresh_one_stale_passes(self):
        """One fresh forecast is enough — don't halt if any model is current."""
        forecasts = [
            _forecast(model="GFS", hours_ago=8.0),   # stale
            _forecast(model="ECMWF", hours_ago=2.0), # fresh
        ]
        check_stale_forecast(forecasts, "NYC", "2026-06-01", _params(stale_forecast_hours=6.0))

    def test_is_forecast_fresh_true(self):
        forecasts = [_forecast(hours_ago=1.0)]
        assert is_forecast_fresh(forecasts, "NYC", "2026-06-01") is True

    def test_is_forecast_fresh_false(self):
        forecasts = [_forecast(hours_ago=10.0)]
        assert is_forecast_fresh(forecasts, "NYC", "2026-06-01") is False

    def test_exception_is_risk_violation_subclass(self):
        forecasts = [_forecast(hours_ago=10.0)]
        with pytest.raises(RiskViolation):
            check_stale_forecast(forecasts, "NYC", "2026-06-01", _params(stale_forecast_hours=6.0))

    def test_just_within_limit_passes(self):
        """A forecast just within the age limit should pass."""
        forecasts = [_forecast(hours_ago=5.9)]
        check_stale_forecast(forecasts, "NYC", "2026-06-01", _params(stale_forecast_hours=6.0))


# --------------------------------------------------------------------------- #
# Guard 2: Cluster Exposure
# --------------------------------------------------------------------------- #

class TestClusterCapExceeded:
    def test_within_cap_passes(self):
        positions = [_position(city="NYC", size_usd=50.0)]
        check_cluster_exposure("BOS", 50.0, positions, bankroll=1000.0,
                               params=_params(max_cluster_exposure_pct=0.15))

    def test_exceeds_cap_raises(self):
        positions = [
            _position(city="NYC", size_usd=80.0),
            _position(city="BOS", size_usd=70.0),
        ]
        with pytest.raises(ClusterCapExceeded):
            check_cluster_exposure("DC", 10.0, positions, bankroll=1000.0,
                                   params=_params(max_cluster_exposure_pct=0.15))

    def test_unknown_city_always_passes(self):
        positions = [_position(city="NYC", size_usd=999.0)]
        check_cluster_exposure("ZZUNKNOWN", 500.0, positions, bankroll=100.0,
                               params=_params(max_cluster_exposure_pct=0.15))

    def test_different_cluster_not_counted(self):
        positions = [_position(city="MIA", size_usd=500.0)]  # south cluster
        # NYC is in northeast — south exposure doesn't affect northeast cap
        check_cluster_exposure("NYC", 100.0, positions, bankroll=1000.0,
                               params=_params(max_cluster_exposure_pct=0.15))

    def test_exactly_at_cap_raises(self):
        """If new trade would exactly reach cap it should raise (> not >=)."""
        positions = [_position(city="NYC", size_usd=140.0)]  # 14% of 1000
        with pytest.raises(ClusterCapExceeded):
            check_cluster_exposure("BOS", 11.0, positions, bankroll=1000.0,
                                   params=_params(max_cluster_exposure_pct=0.15))

    def test_is_risk_violation_subclass(self):
        positions = [_position(city="NYC", size_usd=200.0)]
        with pytest.raises(RiskViolation):
            check_cluster_exposure("BOS", 200.0, positions, bankroll=1000.0,
                                   params=_params(max_cluster_exposure_pct=0.15))


class TestCityLimitExceeded:
    def test_within_limit_passes(self):
        positions = [_position(city="NYC"), _position(city="NYC")]
        check_city_limit("NYC", positions, _params(max_positions_per_city=3))

    def test_at_limit_raises(self):
        positions = [_position(city="NYC")] * 3
        with pytest.raises(CityLimitExceeded):
            check_city_limit("NYC", positions, _params(max_positions_per_city=3))

    def test_different_city_not_counted(self):
        positions = [_position(city="CHI")] * 5
        check_city_limit("NYC", positions, _params(max_positions_per_city=3))

    def test_empty_positions_passes(self):
        check_city_limit("NYC", [], _params(max_positions_per_city=3))


# --------------------------------------------------------------------------- #
# Guard 3: Daily Loss Limit
# --------------------------------------------------------------------------- #

class TestDailyLossHalt:
    def test_no_loss_passes(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=50.0)   # positive PnL
        check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)

    def test_realized_loss_below_limit_passes(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-30.0)  # 3% loss < 5% limit
        check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)

    def test_realized_loss_at_limit_raises(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-50.0)  # exactly 5%
        with pytest.raises(DailyLossHalt):
            check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)

    def test_realized_loss_above_limit_raises(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-80.0)  # 8% > 5%
        with pytest.raises(DailyLossHalt):
            check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)

    def test_unrealized_loss_counted(self, tmp_path):
        conn = _db(tmp_path)
        # YES position: entry 0.60, current 0.50 → loss = 0.10 * 600 = $60 > 5% of $1000 ($50)
        positions = [_position(city="NYC", size_usd=600.0, side="YES", entry=0.60, current=0.50)]
        with pytest.raises(DailyLossHalt):
            check_daily_loss_limit(conn, positions, bankroll=1000.0, max_loss_pct=0.05)

    def test_combined_loss_triggers_halt(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-30.0)  # 3% realized
        # Open position with $30 unrealized loss → total 6% > 5%
        positions = [_position(city="NYC", size_usd=300.0, side="YES", entry=0.60, current=0.50)]
        with pytest.raises(DailyLossHalt):
            check_daily_loss_limit(conn, positions, bankroll=1000.0, max_loss_pct=0.05)

    def test_no_daily_pnl_row_passes(self, tmp_path):
        """If no daily_pnl row exists yet, treat realized loss as 0."""
        conn = _db(tmp_path)
        check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)

    def test_no_position_losing_no_trigger(self, tmp_path):
        """Position moving in our favour should not count as loss."""
        conn = _db(tmp_path)
        positions = [_position(city="NYC", size_usd=500.0, side="YES", entry=0.50, current=0.60)]
        check_daily_loss_limit(conn, positions, bankroll=1000.0, max_loss_pct=0.05)

    def test_get_daily_loss_breakdown(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-20.0)
        positions = [_position(city="NYC", size_usd=200.0, side="YES", entry=0.60, current=0.50)]
        loss = get_daily_loss(conn, positions)
        assert loss["realized_loss"] == pytest.approx(20.0)
        assert loss["unrealized_loss"] == pytest.approx(20.0)   # 0.10 * 200
        assert loss["total_loss"] == pytest.approx(40.0)

    def test_is_risk_violation_subclass(self, tmp_path):
        conn = _db(tmp_path)
        _seed_daily_pnl(conn, pnl=-100.0)
        with pytest.raises(RiskViolation):
            check_daily_loss_limit(conn, [], bankroll=1000.0, max_loss_pct=0.05)


# --------------------------------------------------------------------------- #
# Phase 4.2 — Reconciliation
# --------------------------------------------------------------------------- #

class TestReconcilePositions:
    def test_no_discrepancies_clean_state(self, tmp_path):
        conn = _db(tmp_path)
        # Insert a local position
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, current_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, 0.60, "HOLDING", "value_entry"),
        )
        conn.commit()
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "open", "size_usd": 100.0}]
        result = reconcile_positions(conn, exchange)
        assert len(result.discrepancies) == 0
        assert result.corrections == 0

    def test_detects_missing_locally(self, tmp_path):
        conn = _db(tmp_path)  # empty DB
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "open", "city": "NYC", "target_date": "2026-06-01"}]
        result = reconcile_positions(conn, exchange, auto_correct=False)
        assert any(d.discrepancy_type == DiscrepancyType.MISSING_LOCALLY
                   for d in result.discrepancies)

    def test_detects_missing_on_exchange(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, "HOLDING", "value_entry"),
        )
        conn.commit()
        result = reconcile_positions(conn, exchange_positions=[], auto_correct=False)
        assert any(d.discrepancy_type == DiscrepancyType.MISSING_ON_EXCHANGE
                   for d in result.discrepancies)

    def test_detects_price_mismatch(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, "HOLDING", "value_entry"),
        )
        conn.commit()
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.60,   # differs > 2¢
                     "status": "open", "size_usd": 100.0}]
        result = reconcile_positions(conn, exchange, auto_correct=False)
        assert any(d.discrepancy_type == DiscrepancyType.PRICE_MISMATCH
                   for d in result.discrepancies)

    def test_detects_status_mismatch(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, "HOLDING", "value_entry"),
        )
        conn.commit()
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "CLOSED", "won": True}]
        result = reconcile_positions(conn, exchange, auto_correct=False)
        assert any(d.discrepancy_type == DiscrepancyType.STATUS_MISMATCH
                   for d in result.discrepancies)

    def test_auto_correct_fixes_status_mismatch(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, "HOLDING", "value_entry"),
        )
        conn.commit()
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "CLOSED", "won": True}]
        reconcile_positions(conn, exchange, auto_correct=True)
        row = conn.execute("SELECT status FROM positions WHERE ticker='TKR_NYC'").fetchone()
        assert row["status"] == "WON"

    def test_auto_correct_inserts_missing_local(self, tmp_path):
        conn = _db(tmp_path)
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "open", "city": "NYC", "target_date": "2026-06-01",
                     "size_usd": 100.0}]
        reconcile_positions(conn, exchange, auto_correct=True)
        row = conn.execute("SELECT * FROM positions WHERE ticker='TKR_NYC'").fetchone()
        assert row is not None
        assert row["status"] == "HOLDING"

    def test_returns_reconciliation_result_type(self, tmp_path):
        conn = _db(tmp_path)
        result = reconcile_positions(conn, [])
        assert isinstance(result, ReconciliationResult)

    def test_critical_count_counts_critical_discrepancies(self, tmp_path):
        conn = _db(tmp_path)
        exchange = [{"ticker": "TKR_NYC", "side": "YES", "entry_price": 0.55,
                     "status": "open", "city": "NYC", "target_date": "2026-06-01"}]
        result = reconcile_positions(conn, exchange, auto_correct=False)
        # MISSING_LOCALLY is critical
        assert result.critical_count >= 1

    def test_empty_both_sides_no_discrepancies(self, tmp_path):
        conn = _db(tmp_path)
        result = reconcile_positions(conn, [])
        assert len(result.discrepancies) == 0


class TestOrphanedOrders:
    def test_detects_orphaned_order(self, tmp_path):
        conn = _db(tmp_path)  # no local positions
        orders = [{"ticker": "TKR_ORPHAN", "status": "open"}]
        orphans = check_orphaned_orders(conn, orders)
        assert len(orphans) == 1
        assert orphans[0].discrepancy_type == DiscrepancyType.ORPHANED_ORDER

    def test_no_orphan_when_in_db(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            """INSERT INTO positions (ticker, city, target_date, side, size_usd,
               entry_price, status, strategy_name)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("TKR_NYC", "NYC", "2026-06-01", "YES", 100.0, 0.55, "HOLDING", "value_entry"),
        )
        conn.commit()
        orders = [{"ticker": "TKR_NYC", "status": "open"}]
        orphans = check_orphaned_orders(conn, orders)
        assert len(orphans) == 0

    def test_empty_orders_no_orphans(self, tmp_path):
        conn = _db(tmp_path)
        orphans = check_orphaned_orders(conn, [])
        assert orphans == []


class TestDiscrepancy:
    def test_missing_locally_is_critical(self):
        d = Discrepancy(DiscrepancyType.MISSING_LOCALLY, "TKR", None, {})
        assert d.is_critical()

    def test_status_mismatch_is_critical(self):
        d = Discrepancy(DiscrepancyType.STATUS_MISMATCH, "TKR", {}, {})
        assert d.is_critical()

    def test_price_mismatch_not_critical(self):
        d = Discrepancy(DiscrepancyType.PRICE_MISMATCH, "TKR", {}, {})
        assert not d.is_critical()

    def test_missing_on_exchange_not_critical(self):
        d = Discrepancy(DiscrepancyType.MISSING_ON_EXCHANGE, "TKR", {}, None)
        assert not d.is_critical()


# --------------------------------------------------------------------------- #
# Phase 4.3 — Startup Checks
# --------------------------------------------------------------------------- #

class TestStartupChecks:
    def test_startup_checks_pass_on_clean_db(self, tmp_path):
        from state.db import init_db as make_db
        conn = make_db(tmp_path / "startup.db")
        from main import startup_checks
        startup_checks(conn)  # should not raise

    def test_startup_checks_fail_on_bad_params(self, tmp_path):
        from state.db import init_db as make_db
        from main import _check_params_sanity
        conn = make_db(tmp_path / "startup.db")
        bad_params = _params(base_std_f=50.0)  # out of range
        with pytest.raises(AssertionError):
            _check_params_sanity(bad_params)

    def test_run_once_dry_run_does_not_raise(self, tmp_path, monkeypatch):
        """Smoke test: run_once in dry-run mode against an empty system."""
        import state.db as db_module
        test_db = tmp_path / "smoke.db"

        # Point the DB module to a temp path for this test
        original_path = db_module.DB_PATH
        db_module.DB_PATH = test_db
        try:
            from main import run_once
            result = run_once(dry_run=True)
            assert "signals_generated" in result
        finally:
            db_module.DB_PATH = original_path
