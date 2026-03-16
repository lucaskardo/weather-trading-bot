from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.db import init_db

pytest.importorskip("flask")


def seed_dashboard_data(conn):
    conn.execute(
        """INSERT INTO markets
           (id, ticker, city, target_date, market_type, low_f, high_f, exchange, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("m1", "KXNYC-80", "NYC", "2026-07-01", "temp_high", 79, 80, "kalshi", "open"),
    )
    conn.execute(
        """INSERT INTO forecasts
           (market_id, city, target_date, model_name, predicted_high_f, confidence,
            run_id, publish_time, source_url, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '-2 hours'), ?, datetime('now', '-1 hours'))""",
        ("m1", "NYC", "2026-07-01", "ecmwf", 81.2, 0.8, "run-1", "https://example.com/ecmwf"),
    )
    conn.execute(
        """INSERT INTO settlement_cache
           (city, target_date, actual_high_f, station, source_url)
           VALUES (?, ?, ?, ?, ?)""",
        ("NYC", "2026-07-01", 80.0, "KNYC", "https://example.com/settlement"),
    )
    conn.execute(
        """INSERT INTO predictions
           (strategy_name, market_id, ticker, city, target_date, fair_value, market_price,
            executable_price, edge, executable_edge, confidence, brier_score, outcome, realized_pnl, is_shadow)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        ("value_entry", "m1", "KXNYC-80", "NYC", "2026-07-01", 0.62, 0.51, 0.52, 0.11, 0.10, 0.75, 0.1444, 1.0, 7.5),
    )
    conn.execute(
        """INSERT INTO decision_audit
           (strategy_name, ticker, city, target_date, provider_publish_time, model_run_time,
            bot_fetch_time, parse_to_signal_time, market_snapshot_time, order_sent_time,
            fill_received_time, revision_confirmed, revision_delta_f)
           VALUES (?, ?, ?, ?, datetime('now','-3 hours'), ?, datetime('now','-2 hours'),
                   datetime('now','-90 minutes'), datetime('now','-80 minutes'),
                   datetime('now','-79 minutes'), datetime('now','-78 minutes'), 1, 4.2)""",
        ("model_release", "KXNYC-80", "NYC", "2026-07-01", "2026070106"),
    )
    conn.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "bot.db"
    conn = init_db(db_path)
    seed_dashboard_data(conn)
    monkeypatch.setenv("BOT_DB_PATH", str(db_path))
    import dashboard.api as api
    importlib.reload(api)
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_decisions_endpoint(client):
    r = client.get("/api/decisions")
    data = r.get_json()
    assert r.status_code == 200
    assert data and data[0]["contract"] == "KXNYC-80"
    assert data[0]["recommended_action"] == "BUY"


def test_truth_metrics_endpoint(client):
    r = client.get("/api/truth_metrics")
    data = r.get_json()
    assert r.status_code == 200
    assert data["resolved_predictions"] >= 1
    assert data["avg_brier"] == pytest.approx(0.1444)
    assert data["by_model"][0]["model_name"] == "ecmwf"


def test_source_health_endpoint(client):
    r = client.get("/api/source_health")
    data = r.get_json()
    assert r.status_code == 200
    assert data[0]["model_name"] == "ecmwf"
    assert "source_url" in data[0]


def test_market_explorer_endpoint(client):
    r = client.get("/api/market_explorer?exchange=kalshi&q=NYC")
    data = r.get_json()
    assert r.status_code == 200
    assert len(data) == 1
    assert data[0]["ticker"] == "KXNYC-80"


def test_quick_stats_and_last_cycle(client):
    r = client.post("/api/last_cycle", json={"executed": 1, "cycle_mode": "paper"})
    assert r.status_code == 200
    r = client.get("/api/quick_stats")
    data = r.get_json()
    assert data["last_cycle"]["executed"] == 1
    assert "bankroll" in data


def test_revision_audit_endpoint(client):
    r = client.get("/api/revision_audit")
    data = r.get_json()
    assert r.status_code == 200
    assert data
    assert data[0]["revision_confirmed"] == 1
    assert data[0]["ticker"] == "KXNYC-80"


def test_calibration_segments_endpoint(client):
    # endpoint should exist even before calibration has populated rows
    r = client.get("/api/calibration_segments")
    data = r.get_json()
    assert r.status_code == 200
    assert isinstance(data, list)


def test_portfolio_risk_endpoint(client):
    r = client.get("/api/portfolio_risk")
    data = r.get_json()
    assert r.status_code == 200
    assert "var95_usd" in data
    assert "limit_pct" in data


def test_truth_alias_and_refresh(client):
    r = client.get("/api/truth")
    data = r.get_json()
    assert r.status_code == 200
    assert "overall_brier" in data
    r2 = client.post("/api/refresh")
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True


def test_autoresearch_status_endpoint(client):
    r = client.get("/api/autoresearch_status")
    data = r.get_json()
    assert r.status_code == 200
    assert "promoted_params" in data
    assert "avg_brier" in data
