"""
Tests for Phase 5: Shadow Competition & Auto-Promotion.

Covers:
  - shadow_logger: log_shadow_signal, log_shadow_signals_batch, settle_shadow_predictions,
                   get_shadow_performance
  - promotion: scan_for_candidates (score gate, trade-count gate)
  - analytics: compute_position_analytics (MFE/MAE, hold time, PnL),
               record_position_analytics, get_strategy_analytics,
               get_liquidity_bucket
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from state.db import init_db
from strategies.base import Signal
from strategies.shadow_logger import (
    get_shadow_performance,
    log_shadow_signal,
    log_shadow_signals_batch,
    settle_shadow_predictions,
)
from strategies.analytics import (
    compute_position_analytics,
    get_liquidity_bucket,
    get_strategy_analytics,
    record_position_analytics,
)
from strategies.promotion import (
    MIN_TRADES_FOR_PROMOTION,
    PROMOTION_SCORE_THRESHOLD,
    scan_for_candidates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    # Add market rows with explicit TEXT ids so FK references work
    c.execute(
        "INSERT INTO markets (id, ticker, city, target_date, high_f, market_type) "
        "VALUES ('mkt1','NYC_H_70','NYC','2025-07-01',70.0,'high')"
    )
    c.execute(
        "INSERT INTO markets (id, ticker, city, target_date, high_f, market_type) "
        "VALUES ('mkt2','CHI_H_80','CHI','2025-08-01',80.0,'high')"
    )
    c.commit()
    yield c
    c.close()


def _make_signal(
    strategy_name="shadow_strat",
    city="NYC",
    target_date="2025-07-01",
    ticker="NYC_H_70",
    fair_value=0.65,
    market_price=0.55,
    executable_price=0.56,
    edge=0.10,
    executable_edge=0.09,
    is_shadow=True,
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        market_id=None,
        ticker=ticker,
        source="test",
        city=city,
        target_date=target_date,
        market_type="high",
        fair_value=fair_value,
        market_price=market_price,
        executable_price=executable_price,
        edge=edge,
        executable_edge=executable_edge,
        confidence=0.70,
        consensus_f=72.0,
        agreement=0.80,
        n_models=3,
        side="YES",
        is_shadow=is_shadow,
    )


# ---------------------------------------------------------------------------
# shadow_logger tests
# ---------------------------------------------------------------------------

class TestLogShadowSignal:
    def test_inserts_row(self, conn):
        sig = _make_signal()
        row_id = log_shadow_signal(conn, sig)
        assert row_id > 0
        row = conn.execute("SELECT * FROM predictions WHERE id=?", (row_id,)).fetchone()
        assert row["is_shadow"] == 1
        assert row["strategy_name"] == "shadow_strat"
        assert row["city"] == "NYC"
        assert abs(row["fair_value"] - 0.65) < 1e-9

    def test_market_id_override(self, conn):
        sig = _make_signal()
        row_id = log_shadow_signal(conn, sig, market_id="mkt1")
        row = conn.execute("SELECT market_id FROM predictions WHERE id=?", (row_id,)).fetchone()
        assert row["market_id"] == "mkt1"

    def test_edge_fields_stored(self, conn):
        sig = _make_signal(executable_edge=0.09)
        row_id = log_shadow_signal(conn, sig)
        row = conn.execute("SELECT executable_edge FROM predictions WHERE id=?", (row_id,)).fetchone()
        assert abs(row["executable_edge"] - 0.09) < 1e-9


class TestLogShadowSignalsBatch:
    def test_skips_non_shadow(self, conn):
        live_sig = _make_signal(is_shadow=False)
        shadow_sig = _make_signal(is_shadow=True)
        ids = log_shadow_signals_batch(conn, [live_sig, shadow_sig])
        assert len(ids) == 1

    def test_batch_all_shadow(self, conn):
        sigs = [_make_signal(city="NYC"), _make_signal(city="CHI")]
        ids = log_shadow_signals_batch(conn, sigs)
        assert len(ids) == 2


class TestSettleShadowPredictions:
    def _insert_shadow_pred(self, conn, city, target_date, market_price, fair_value=0.6):
        cursor = conn.execute(
            """INSERT INTO predictions
               (strategy_name, ticker, city, target_date,
                fair_value, market_price, executable_price,
                edge, executable_edge, confidence,
                consensus_f, agreement, n_models, is_shadow)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("s", "T", city, target_date,
             fair_value, market_price, market_price,
             0.1, 0.09, 0.7, 72.0, 0.8, 3, 1),
        )
        conn.commit()
        return cursor.lastrowid

    def test_yes_outcome_positive_pnl(self, conn):
        # market_price=0.4, outcome YES → pnl = (1-0.4)*100 = 60
        pid = self._insert_shadow_pred(conn, "NYC", "2025-07-01", market_price=0.40)
        conn.execute("UPDATE predictions SET market_id='mkt1' WHERE id=?", (pid,))
        conn.commit()

        settled = settle_shadow_predictions(conn, "NYC", "2025-07-01", 75.0, "KLGA")
        assert settled == 1
        row = conn.execute("SELECT outcome, realized_pnl, brier_score FROM predictions WHERE id=?", (pid,)).fetchone()
        assert row["outcome"] == 1.0
        assert abs(row["realized_pnl"] - 60.0) < 1e-6

    def test_no_outcome_negative_pnl(self, conn):
        # market_price=0.6, outcome NO → pnl = -0.6*100 = -60
        pid = self._insert_shadow_pred(conn, "NYC", "2025-07-01", market_price=0.60)
        conn.execute("UPDATE predictions SET market_id='mkt1' WHERE id=?", (pid,))
        conn.commit()

        # actual_high_f=65 < high_f=70 → outcome NO
        settled = settle_shadow_predictions(conn, "NYC", "2025-07-01", 65.0, "KLGA")
        assert settled == 1
        row = conn.execute("SELECT outcome, realized_pnl FROM predictions WHERE id=?", (pid,)).fetchone()
        assert row["outcome"] == 0.0
        assert abs(row["realized_pnl"] - (-60.0)) < 1e-6

    def test_brier_score_computed(self, conn):
        pid = self._insert_shadow_pred(conn, "NYC", "2025-07-01", market_price=0.6, fair_value=0.6)
        conn.execute("UPDATE predictions SET market_id='mkt1' WHERE id=?", (pid,))
        conn.commit()
        settle_shadow_predictions(conn, "NYC", "2025-07-01", 75.0, "KLGA")
        row = conn.execute("SELECT brier_score, fair_value FROM predictions WHERE id=?", (pid,)).fetchone()
        expected_brier = (row["fair_value"] - 1.0) ** 2
        assert abs(row["brier_score"] - expected_brier) < 1e-9

    def test_already_settled_not_resettled(self, conn):
        pid = self._insert_shadow_pred(conn, "NYC", "2025-07-01", market_price=0.5)
        conn.execute("UPDATE predictions SET market_id='mkt1', outcome=1.0 WHERE id=?", (pid,))
        conn.commit()
        settled = settle_shadow_predictions(conn, "NYC", "2025-07-01", 75.0, "KLGA")
        assert settled == 0

    def test_returns_count(self, conn):
        for _ in range(3):
            pid = self._insert_shadow_pred(conn, "CHI", "2025-08-01", market_price=0.5)
            conn.execute("UPDATE predictions SET market_id='mkt2' WHERE id=?", (pid,))
        conn.commit()
        settled = settle_shadow_predictions(conn, "CHI", "2025-08-01", 85.0, "KORD")
        assert settled == 3


class TestGetShadowPerformance:
    def _seed_shadow_trades(self, conn, strategy_name, n_wins, n_losses):
        for _ in range(n_wins):
            conn.execute(
                """INSERT INTO predictions
                   (strategy_name, ticker, city, target_date,
                    fair_value, market_price, executable_price,
                    edge, executable_edge, confidence,
                    consensus_f, agreement, n_models,
                    is_shadow, outcome, brier_score, realized_pnl)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (strategy_name, "T", "NYC", "2025-07-01",
                 0.7, 0.5, 0.5, 0.2, 0.19, 0.8, 72.0, 0.9, 3,
                 1, 1.0, 0.09, 50.0),
            )
        for _ in range(n_losses):
            conn.execute(
                """INSERT INTO predictions
                   (strategy_name, ticker, city, target_date,
                    fair_value, market_price, executable_price,
                    edge, executable_edge, confidence,
                    consensus_f, agreement, n_models,
                    is_shadow, outcome, brier_score, realized_pnl)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (strategy_name, "T", "NYC", "2025-07-01",
                 0.3, 0.5, 0.5, -0.2, -0.21, 0.8, 72.0, 0.9, 3,
                 1, 0.0, 0.49, -50.0),
            )
        conn.commit()

    def test_returns_none_below_min_trades(self, conn):
        self._seed_shadow_trades(conn, "sparse", 5, 0)
        result = get_shadow_performance(conn, "sparse", min_trades=10)
        assert result is None

    def test_win_rate(self, conn):
        self._seed_shadow_trades(conn, "balanced", 7, 3)
        result = get_shadow_performance(conn, "balanced", min_trades=10)
        assert result is not None
        assert abs(result["win_rate"] - 0.7) < 1e-9

    def test_sharpe_positive_for_net_winner(self, conn):
        self._seed_shadow_trades(conn, "winner", 8, 2)
        result = get_shadow_performance(conn, "winner", min_trades=10)
        assert result["sharpe"] > 0

    def test_trade_count(self, conn):
        self._seed_shadow_trades(conn, "counter", 6, 4)
        result = get_shadow_performance(conn, "counter", min_trades=10)
        assert result["trade_count"] == 10


# ---------------------------------------------------------------------------
# analytics tests
# ---------------------------------------------------------------------------

class TestComputePositionAnalytics:
    def _make_position(self, side="YES", entry_price=0.50, size_usd=100.0, city="NYC"):
        now = datetime.now(timezone.utc)
        opened = (now - timedelta(hours=2)).isoformat()
        return {
            "side": side,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "city": city,
            "opened_at": opened,
            "executable_edge": 0.10,
        }

    def test_hold_time_positive(self):
        pos = self._make_position()
        closed_at = datetime.now(timezone.utc).isoformat()
        result = compute_position_analytics(pos, [0.51, 0.52], 0.55, closed_at)
        assert result["hold_time_hours"] > 0

    def test_mfe_yes_side(self):
        # YES: MFE = max(all_prices) - entry
        pos = self._make_position(side="YES", entry_price=0.50)
        result = compute_position_analytics(pos, [0.55, 0.60], 0.58, datetime.now(timezone.utc).isoformat())
        assert abs(result["max_favorable_excursion"] - 0.10) < 1e-9  # 0.60 - 0.50

    def test_mae_yes_side(self):
        # YES: MAE = entry - min(all_prices)
        pos = self._make_position(side="YES", entry_price=0.50)
        result = compute_position_analytics(pos, [0.45, 0.48], 0.52, datetime.now(timezone.utc).isoformat())
        assert abs(result["max_adverse_excursion"] - 0.05) < 1e-9  # 0.50 - 0.45

    def test_mfe_no_side(self):
        # NO: MFE = entry - min(all_prices)
        pos = self._make_position(side="NO", entry_price=0.50)
        result = compute_position_analytics(pos, [0.40, 0.45], 0.42, datetime.now(timezone.utc).isoformat())
        assert abs(result["max_favorable_excursion"] - 0.10) < 1e-9  # 0.50 - 0.40

    def test_mae_no_side(self):
        # NO: MAE = max(all_prices) - entry
        pos = self._make_position(side="NO", entry_price=0.50)
        result = compute_position_analytics(pos, [0.55, 0.60], 0.58, datetime.now(timezone.utc).isoformat())
        assert abs(result["max_adverse_excursion"] - 0.10) < 1e-9  # 0.60 - 0.50

    def test_realized_pnl_yes_win(self):
        pos = self._make_position(side="YES", entry_price=0.50, size_usd=100.0)
        result = compute_position_analytics(pos, [], 0.70, datetime.now(timezone.utc).isoformat())
        assert abs(result["realized_pnl"] - 20.0) < 1e-9  # (0.70-0.50)*100

    def test_realized_pnl_yes_loss(self):
        pos = self._make_position(side="YES", entry_price=0.60, size_usd=100.0)
        result = compute_position_analytics(pos, [], 0.40, datetime.now(timezone.utc).isoformat())
        assert abs(result["realized_pnl"] - (-20.0)) < 1e-9

    def test_realized_pnl_no_win(self):
        # NO win: price fell from 0.60 to 0.40
        pos = self._make_position(side="NO", entry_price=0.60, size_usd=100.0)
        result = compute_position_analytics(pos, [], 0.40, datetime.now(timezone.utc).isoformat())
        assert abs(result["realized_pnl"] - 20.0) < 1e-9  # (0.60-0.40)*100

    def test_exit_reason_contains_cluster(self):
        pos = self._make_position(city="NYC")
        result = compute_position_analytics(pos, [], 0.55, datetime.now(timezone.utc).isoformat())
        payload = json.loads(result["exit_reason"])
        assert payload["cluster"] == "northeast"

    def test_exit_reason_contains_forecast_run_ids(self):
        pos = self._make_position()
        result = compute_position_analytics(
            pos, [], 0.55, datetime.now(timezone.utc).isoformat(),
            forecast_run_ids=["run1", "run2"]
        )
        payload = json.loads(result["exit_reason"])
        assert "run1" in payload["forecast_run_ids"]
        assert "run2" in payload["forecast_run_ids"]

    def test_status_won(self):
        pos = self._make_position(side="YES", entry_price=0.50, size_usd=100.0)
        result = compute_position_analytics(pos, [], 0.70, datetime.now(timezone.utc).isoformat())
        assert result["status"] == "WON"

    def test_status_lost(self):
        pos = self._make_position(side="YES", entry_price=0.70, size_usd=100.0)
        result = compute_position_analytics(pos, [], 0.50, datetime.now(timezone.utc).isoformat())
        assert result["status"] == "LOST"

    def test_empty_price_history(self):
        pos = self._make_position(side="YES", entry_price=0.50)
        result = compute_position_analytics(pos, [], 0.60, datetime.now(timezone.utc).isoformat())
        assert result["max_favorable_excursion"] >= 0
        assert result["max_adverse_excursion"] >= 0


class TestRecordPositionAnalytics:
    def _insert_position(self, conn):
        cursor = conn.execute(
            """INSERT INTO positions
               (strategy_name, ticker, city, target_date,
                side, entry_price, size_usd, status)
               VALUES ('s','T','NYC','2025-07-01','YES',0.5,100.0,'HOLDING')"""
        )
        conn.commit()
        return cursor.lastrowid

    def test_updates_fields(self, conn):
        pid = self._insert_position(conn)
        record_position_analytics(conn, pid, {
            "hold_time_hours": 3.5,
            "max_favorable_excursion": 0.10,
            "max_adverse_excursion": 0.05,
            "realized_pnl": 20.0,
            "status": "WON",
        })
        row = conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
        assert abs(row["hold_time_hours"] - 3.5) < 1e-9
        assert abs(row["max_favorable_excursion"] - 0.10) < 1e-9
        assert row["status"] == "WON"

    def test_ignores_unknown_fields(self, conn):
        pid = self._insert_position(conn)
        # Should not raise
        record_position_analytics(conn, pid, {
            "realized_pnl": 10.0,
            "unknown_field": "ignored",
        })
        row = conn.execute("SELECT realized_pnl FROM positions WHERE id=?", (pid,)).fetchone()
        assert abs(row["realized_pnl"] - 10.0) < 1e-9

    def test_empty_dict_is_noop(self, conn):
        pid = self._insert_position(conn)
        # Should not raise and not change anything
        record_position_analytics(conn, pid, {})


class TestGetStrategyAnalytics:
    def _insert_closed_position(self, conn, strategy, pnl, city="NYC", hold_hours=2.0, mfe=0.1, mae=0.05):
        conn.execute(
            """INSERT INTO positions
               (strategy_name, ticker, city, target_date,
                side, entry_price, size_usd,
                status, realized_pnl, hold_time_hours,
                max_favorable_excursion, max_adverse_excursion)
               VALUES (?,?,?,?, 'YES',0.5,100.0, 'WON',?,?,?,?)""",
            (strategy, "T", city, "2025-07-01", pnl, hold_hours, mfe, mae),
        )
        conn.commit()

    def test_empty_returns_zero_count(self, conn):
        result = get_strategy_analytics(conn, "nonexistent")
        assert result["trade_count"] == 0

    def test_win_rate(self, conn):
        self._insert_closed_position(conn, "s1", pnl=10.0)
        self._insert_closed_position(conn, "s1", pnl=-5.0)
        self._insert_closed_position(conn, "s1", pnl=8.0)
        result = get_strategy_analytics(conn, "s1")
        assert abs(result["win_rate"] - 2/3) < 1e-9

    def test_total_pnl(self, conn):
        self._insert_closed_position(conn, "s2", pnl=10.0)
        self._insert_closed_position(conn, "s2", pnl=20.0)
        result = get_strategy_analytics(conn, "s2")
        assert abs(result["total_pnl"] - 30.0) < 1e-9

    def test_mfe_mae_ratio(self, conn):
        self._insert_closed_position(conn, "s3", pnl=10.0, mfe=0.20, mae=0.10)
        self._insert_closed_position(conn, "s3", pnl=5.0, mfe=0.20, mae=0.10)
        result = get_strategy_analytics(conn, "s3")
        assert abs(result["mfe_to_mae_ratio"] - 2.0) < 1e-9

    def test_cluster_breakdown(self, conn):
        self._insert_closed_position(conn, "s4", pnl=10.0, city="NYC")
        self._insert_closed_position(conn, "s4", pnl=5.0, city="BOS")
        self._insert_closed_position(conn, "s4", pnl=3.0, city="CHI")
        result = get_strategy_analytics(conn, "s4")
        assert result["cluster_breakdown"].get("northeast", 0) == 2
        assert result["cluster_breakdown"].get("midwest", 0) == 1


class TestGetLiquidityBucket:
    def test_micro(self):
        assert get_liquidity_bucket(10.0) == "micro"
        assert get_liquidity_bucket(24.99) == "micro"

    def test_small(self):
        assert get_liquidity_bucket(25.0) == "small"
        assert get_liquidity_bucket(99.99) == "small"

    def test_medium(self):
        assert get_liquidity_bucket(100.0) == "medium"
        assert get_liquidity_bucket(499.99) == "medium"

    def test_large(self):
        assert get_liquidity_bucket(500.0) == "large"
        assert get_liquidity_bucket(10000.0) == "large"


# ---------------------------------------------------------------------------
# promotion tests
# ---------------------------------------------------------------------------

class TestScanForCandidates:
    def _seed_shadow_resolved(self, conn, strategy_name, n, pnl=50.0, brier=0.09):
        for _ in range(n):
            conn.execute(
                """INSERT INTO predictions
                   (strategy_name, ticker, city, target_date,
                    fair_value, market_price, executable_price,
                    edge, executable_edge, confidence,
                    consensus_f, agreement, n_models,
                    is_shadow, outcome, brier_score, realized_pnl)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (strategy_name, "T", "NYC", "2025-07-01",
                 0.7, 0.5, 0.5, 0.2, 0.19, 0.8, 72.0, 0.9, 3,
                 1, 1.0, brier, pnl),
            )
        conn.commit()

    def test_below_min_trades_not_candidate(self, conn):
        self._seed_shadow_resolved(conn, "few_trades", MIN_TRADES_FOR_PROMOTION - 1)
        with patch("strategies.promotion.score_strategy", return_value=90.0):
            candidates = scan_for_candidates(conn, ["few_trades"])
        assert len(candidates) == 0

    def test_below_score_threshold_not_candidate(self, conn):
        self._seed_shadow_resolved(conn, "low_score", MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=PROMOTION_SCORE_THRESHOLD - 1):
            candidates = scan_for_candidates(conn, ["low_score"])
        assert len(candidates) == 0

    def test_meets_criteria_is_candidate(self, conn):
        self._seed_shadow_resolved(conn, "good_strat", MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=PROMOTION_SCORE_THRESHOLD + 1):
            candidates = scan_for_candidates(conn, ["good_strat"])
        assert len(candidates) == 1
        assert candidates[0].strategy_name == "good_strat"
        assert candidates[0].score > PROMOTION_SCORE_THRESHOLD

    def test_exact_threshold_not_candidate(self, conn):
        # score must be strictly > threshold
        self._seed_shadow_resolved(conn, "exact", MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=PROMOTION_SCORE_THRESHOLD):
            candidates = scan_for_candidates(conn, ["exact"])
        assert len(candidates) == 0

    def test_candidate_logged_to_strategy_metrics(self, conn):
        self._seed_shadow_resolved(conn, "log_me", MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=80.0):
            scan_for_candidates(conn, ["log_me"])
        row = conn.execute(
            "SELECT * FROM strategy_metrics WHERE strategy_name='log_me'"
        ).fetchone()
        assert row is not None
        assert row["is_live"] == 0

    def test_multiple_candidates(self, conn):
        for name in ["alpha", "beta"]:
            self._seed_shadow_resolved(conn, name, MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=80.0):
            candidates = scan_for_candidates(conn, ["alpha", "beta"])
        assert len(candidates) == 2

    def test_score_none_not_candidate(self, conn):
        self._seed_shadow_resolved(conn, "no_score", MIN_TRADES_FOR_PROMOTION)
        with patch("strategies.promotion.score_strategy", return_value=None):
            candidates = scan_for_candidates(conn, ["no_score"])
        assert len(candidates) == 0
