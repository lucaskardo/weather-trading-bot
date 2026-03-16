from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path

from execution.exchange_executor import PaperExecutor, MakerFirstExecutor
from shared.params import Params
from state.db import init_db
from strategies.base import Signal
from strategy_router.brain import Brain


def _db(tmp_path: Path):
    return init_db(tmp_path / "paper_exec.db")


def _params(**kw) -> Params:
    p = Params()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _signal(**overrides) -> Signal:
    payload = dict(
        strategy_name="value_entry",
        market_id="mkt1",
        ticker="T1",
        source="kalshi",
        city="NYC",
        target_date="2026-06-01",
        market_type="above",
        high_f=75.0,
        market_price=0.50,
        fair_value=0.65,
        executable_price=0.52,
        edge=0.15,
        executable_edge=0.10,
        effective_prob=0.65,
        effective_price=0.52,
        effective_edge=0.13,
        confidence=0.8,
        consensus_f=76.0,
        agreement=2.0,
        n_models=3,
        side="YES",
    )
    payload.update(overrides)
    return Signal(**payload)


def test_paper_executor_partial_fill_when_depth_thin():
    ex = PaperExecutor()
    result = ex.place_order("T1", "YES", size_usd=25.0, price=0.52, dry_run=False, depth_usd=10.0)
    assert result["status"] == "partial"
    assert round(result["fill_ratio"], 4) == 0.4
    assert round(result["fill_size_usd"], 4) == 10.0


def test_maker_executor_uses_more_conservative_depth_in_dry_run():
    ex = MakerFirstExecutor(PaperExecutor())
    result = ex.place_order("T1", "YES", size_usd=25.0, price=0.52, dry_run=True, depth_usd=10.0)
    assert result["execution_type"] == "maker"
    assert result["fill_size_usd"] < 10.0  # maker depth haircut


def test_brain_persists_partial_fill_size(tmp_path):
    conn = _db(tmp_path)
    conn.execute(
        """INSERT INTO markets (id, ticker, city, target_date, market_type, exchange)
           VALUES ('mkt1','T1','NYC','2026-06-01','above','kalshi')"""
    )
    conn.commit()
    brain = Brain(conn=conn, params=_params(), dry_run=False, executor=PaperExecutor())
    sig = _signal(execution_depth_usd=10.0)
    brain._execute_order({"signal": sig, "size_usd": 25.0})
    pos = conn.execute("SELECT size_usd FROM positions LIMIT 1").fetchone()
    assert pos is not None
    assert round(float(pos["size_usd"]), 4) == 10.0
