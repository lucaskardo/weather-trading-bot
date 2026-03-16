from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.db import init_db
from strategy_router.brain import Brain
from strategies.base import Signal
from strategies.model_release import _compute_run_delta
from shared.params import Params
from shared.types import ModelForecast


def test_model_release_requires_confirmation():
    forecasts = [
        ModelForecast("GFS", "NYC", "2026-07-01", 80.0, run_id="2026070100", publish_time="2026-07-01T00:30:00+00:00", fetched_at="2026-07-01T00:40:00+00:00"),
        ModelForecast("ECMWF", "NYC", "2026-07-01", 81.0, run_id="2026070100", publish_time="2026-07-01T00:30:00+00:00", fetched_at="2026-07-01T00:40:00+00:00"),
        ModelForecast("GFS", "NYC", "2026-07-01", 76.0, run_id="2026063018", publish_time="2026-06-30T18:30:00+00:00", fetched_at="2026-06-30T18:40:00+00:00"),
        ModelForecast("ECMWF", "NYC", "2026-07-01", 77.0, run_id="2026063018", publish_time="2026-06-30T18:30:00+00:00", fetched_at="2026-06-30T18:40:00+00:00"),
        ModelForecast("GFS", "NYC", "2026-07-01", 73.5, run_id="2026063012", publish_time="2026-06-30T12:30:00+00:00", fetched_at="2026-06-30T12:40:00+00:00"),
        ModelForecast("ECMWF", "NYC", "2026-07-01", 74.5, run_id="2026063012", publish_time="2026-06-30T12:30:00+00:00", fetched_at="2026-06-30T12:40:00+00:00"),
    ]
    delta, _, _, new_ids, confirmed, publish_time, fetch_time = _compute_run_delta(forecasts, "NYC", "2026-07-01")
    assert delta is not None and delta > 0
    assert confirmed is True
    assert new_ids[0] == "2026070100"
    assert publish_time is not None
    assert fetch_time is not None


def test_brain_persists_decision_audit(tmp_path):
    conn = init_db(tmp_path / "ts.db")
    conn.execute(
        """INSERT INTO markets (id, ticker, city, target_date, market_type, high_f, exchange, status)
           VALUES ('m1','KXNYC-80','NYC','2026-07-01','above',80.0,'kalshi','open')"""
    )
    conn.commit()

    sig = Signal(
        strategy_name="model_release",
        market_id="m1",
        ticker="KXNYC-80",
        source="kalshi",
        city="NYC",
        target_date="2026-07-01",
        market_type="above",
        high_f=80.0,
        market_price=0.45,
        fair_value=0.60,
        executable_price=0.46,
        edge=0.15,
        executable_edge=0.14,
        confidence=0.8,
        side="YES",
        provider_publish_time="2026-07-01T00:30:00+00:00",
        model_run_time="2026070100",
        bot_fetch_time="2026-07-01T00:40:00+00:00",
        parse_to_signal_time="2026-07-01T00:41:00+00:00",
        market_snapshot_time="2026-07-01T00:41:05+00:00",
        revision_confirmed=True,
        revision_delta_f=4.2,
    )

    brain = Brain(conn, params=Params(), dry_run=False)
    brain._execute_order({"signal": sig, "size_usd": 25.0})

    row = conn.execute("SELECT * FROM decision_audit ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["ticker"] == "KXNYC-80"
    assert row["revision_confirmed"] == 1
    assert abs(row["revision_delta_f"] - 4.2) < 1e-9
    assert row["market_snapshot_time"] is not None
    assert row["order_sent_time"] is not None
    assert row["fill_received_time"] is not None
