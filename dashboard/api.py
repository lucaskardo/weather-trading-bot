"""
Dashboard API server — Flask micro-server reading from bot.db.

Start with:
    cd weather_trading_bot
    python dashboard/api.py

Then open dashboard/index.html in a browser.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Allow running from project root or dashboard/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, jsonify, request
    from flask.wrappers import Response
except ImportError:
    print("Flask not installed. Run: pip install flask", file=sys.stderr)
    sys.exit(1)

# Last-cycle summary written by main.py via _write_last_cycle()
_LAST_CYCLE: dict = {}

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

from risk.guards import estimate_portfolio_var95
from research.autoresearch import recent_autoresearch_summary
from shared.params import PARAMS

DB_PATH = os.environ.get(
    "BOT_DB_PATH",
    str(Path(__file__).parent.parent / "data" / "bot.db"),
)

# CORS for local development
@app.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(query: str, params: tuple = ()) -> list[dict]:
    try:
        with _conn() as c:
            return [dict(r) for r in c.execute(query, params).fetchall()]
    except Exception as exc:
        return [{"error": str(exc)}]


def _one(query: str, params: tuple = ()) -> dict:
    try:
        with _conn() as c:
            row = c.execute(query, params).fetchone()
            return dict(row) if row else {}
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# Tab 1: Live / Paper Trading
# --------------------------------------------------------------------------- #


@app.route("/api/decisions")
def decisions():
    """Recent non-shadow decision candidates for a cockpit-style table.

    Uses persisted predictions as the source of truth so the dashboard only shows
    data the bot actually computed and stored.
    """
    limit = int(request.args.get("limit", 50))
    bankroll_row = _one("SELECT bankroll FROM portfolio WHERE id=1")
    bankroll = float(bankroll_row.get("bankroll") or PARAMS.base_bankroll)
    rows = _rows(
        """SELECT p.ticker as contract,
                  p.ticker,
                  p.city,
                  p.target_date,
                  COALESCE(m.exchange, 'kalshi') as exchange,
                  p.strategy_name,
                  p.fair_value as fair_prob,
                  p.market_price,
                  p.executable_price,
                  p.edge,
                  p.executable_edge,
                  p.confidence,
                  p.consensus_f,
                  p.agreement,
                  p.n_models,
                  p.created_at
           FROM predictions p
           LEFT JOIN markets m ON m.id = p.market_id
           WHERE p.is_shadow = 0
           ORDER BY p.created_at DESC
           LIMIT ?""",
        (limit,),
    )
    for row in rows:
        exe_edge = row.get("executable_edge")
        edge = exe_edge if exe_edge is not None else row.get("edge")
        edge = float(edge or 0.0)
        conf = float(row.get("confidence") or 0.0)
        row["recommended_action"] = "BUY" if edge > 0 else "HOLD"
        row["suggested_usd"] = round(max(0.0, bankroll * min(edge, PARAMS.max_kelly_fraction) * max(conf, 0.25)), 2)
        row["freshness_minutes"] = _one(
            "SELECT ROUND((julianday('now') - julianday(?)) * 24 * 60, 1) as mins",
            (row.get("created_at"),),
        ).get("mins")
        mins = float(row.get("freshness_minutes") or 0.0)
        row["stale"] = mins > 30.0
        row["edge_bucket"] = "high" if edge >= 0.10 else ("medium" if edge >= 0.05 else "low")
    return jsonify(rows)


@app.route("/api/truth_metrics")
def truth_metrics():
    """Summary truthfulness metrics for dashboard trust panels."""
    overall = _one(
        """SELECT
               COUNT(*) as resolved_predictions,
               AVG(brier_score) as avg_brier,
               AVG(CASE WHEN brier_score IS NOT NULL THEN fair_value END) as avg_forecast_prob,
               AVG(CASE WHEN brier_score IS NOT NULL THEN outcome END) as avg_outcome
           FROM predictions
           WHERE brier_score IS NOT NULL"""
    )
    recent = _one(
        """SELECT
               COUNT(*) as resolved_predictions_30d,
               AVG(brier_score) as avg_brier_30d
           FROM predictions
           WHERE brier_score IS NOT NULL
             AND created_at > datetime('now', '-30 days')"""
    )
    by_strategy = _rows(
        """SELECT strategy_name,
                  COUNT(*) as n,
                  AVG(brier_score) as avg_brier,
                  AVG(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl END) as avg_realized_pnl
           FROM predictions
           WHERE brier_score IS NOT NULL
           GROUP BY strategy_name
           ORDER BY avg_brier ASC, n DESC"""
    )
    by_model = _rows(
        """SELECT f.model_name,
                  COUNT(*) as n_obs,
                  AVG(f.predicted_high_f - s.actual_high_f) as bias_f,
                  AVG(ABS(f.predicted_high_f - s.actual_high_f)) as mae_f,
                  MAX(f.fetched_at) as last_fetched_at
           FROM forecasts f
           JOIN settlement_cache s
             ON s.city = f.city AND s.target_date = f.target_date
           GROUP BY f.model_name
           HAVING COUNT(*) >= 1
           ORDER BY mae_f ASC, n_obs DESC"""
    )
    payload = {**overall, **recent, "by_strategy": by_strategy, "by_model": by_model}
    return jsonify(payload)




@app.route("/api/truth")
def truth():
    """Compact truth center payload for dashboard badges and summaries."""
    payload = truth_metrics().get_json()  # type: ignore[attr-defined]
    return jsonify({
        "overall_brier": payload.get("avg_brier"),
        "overall_brier_30d": payload.get("avg_brier_30d"),
        "resolved_predictions": payload.get("resolved_predictions", 0),
        "models": payload.get("by_model", []),
        "strategies": payload.get("by_strategy", []),
        "last_settlement_badge": "NWS confirmed ✓" if payload.get("resolved_predictions", 0) else "No settlements yet",
    })




@app.route("/api/operator_brief")
def operator_brief():
    decisions_payload = decisions().get_json()  # type: ignore[attr-defined]
    truth_payload = truth().get_json()  # type: ignore[attr-defined]
    risk_payload = portfolio_risk().get_json()  # type: ignore[attr-defined]
    source_payload = source_health().get_json()  # type: ignore[attr-defined]
    open_positions = _one("SELECT COUNT(*) as n, COALESCE(SUM(size_usd),0) as exposure FROM positions WHERE status NOT IN ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')")
    top_decision = decisions_payload[0] if decisions_payload else None
    stale_sources = [s for s in source_payload if (s.get('hours_since_fetch') or 0) > PARAMS.stale_forecast_hours]
    return jsonify({
        "top_decision": top_decision,
        "open_positions": open_positions.get("n", 0),
        "open_exposure_usd": open_positions.get("exposure", 0.0),
        "overall_brier": truth_payload.get("overall_brier"),
        "overall_brier_30d": truth_payload.get("overall_brier_30d"),
        "var95_pct": risk_payload.get("var95_pct"),
        "stale_sources": len(stale_sources),
        "stale_source_models": [s.get("model_name") for s in stale_sources],
    })


@app.route("/api/refresh", methods=["POST", "GET"])
def refresh():
    """Simple polling-friendly endpoint for dashboards.

    Current UI is static/poll based, so this acts as a cheap heartbeat/refresh hook
    without introducing websockets yet.
    """
    return jsonify({"ok": True, "last_cycle": _LAST_CYCLE})


@app.route("/api/autoresearch_status")
def autoresearch_status():
    with _conn() as c:
        return jsonify(recent_autoresearch_summary(c))
@app.route("/api/source_health")
def source_health():
    """Status of stored forecast sources and freshness."""
    return jsonify(_rows(
        """SELECT model_name,
                  COUNT(*) as forecasts,
                  MAX(fetched_at) as last_fetched_at,
                  MAX(publish_time) as last_publish_time,
                  ROUND((julianday('now') - julianday(MAX(fetched_at))) * 24, 2) as hours_since_fetch,
                  MAX(source_url) as source_url
           FROM forecasts
           GROUP BY model_name
           ORDER BY last_fetched_at DESC"""
    ))


@app.route("/api/market_explorer")
def market_explorer():
    """Browse stored open weather markets across exchanges."""
    limit = int(request.args.get("limit", 100))
    exchange = request.args.get("exchange", "")
    city = request.args.get("city", "")
    q = request.args.get("q", "")
    where = ["1=1"]
    params: list = []
    if exchange:
        where.append("exchange = ?")
        params.append(exchange)
    if city:
        where.append("city = ?")
        params.append(city)
    if q:
        where.append("(ticker LIKE ? OR city LIKE ? OR market_type LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    params.append(limit)
    return jsonify(_rows(
        f"""SELECT id, ticker, city, target_date, market_type, low_f, high_f, nws_station, exchange, status, updated_at
            FROM markets
            WHERE {' AND '.join(where)}
            ORDER BY CASE WHEN status='open' THEN 0 ELSE 1 END, target_date DESC, ticker
            LIMIT ?""",
        tuple(params),
    ))



@app.route("/api/revision_audit")
def revision_audit():
    """Timestamp-doctrine audit trail for revision-sensitive decisions."""
    limit = int(request.args.get("limit", 50))
    return jsonify(_rows(
        """SELECT strategy_name, ticker, city, target_date,
                  provider_publish_time, model_run_time, bot_fetch_time,
                  parse_to_signal_time, market_snapshot_time, order_sent_time,
                  fill_received_time, revision_confirmed, revision_delta_f, created_at
           FROM decision_audit
           WHERE revision_delta_f IS NOT NULL OR revision_confirmed = 1
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,)
    ))



@app.route("/api/calibration_segments")
def calibration_segments():
    return jsonify(_rows(
        """SELECT segment_kind, segment_value, trade_count, avg_brier, avg_outcome, avg_prediction, computed_at
           FROM calibration_segments
           ORDER BY segment_kind, trade_count DESC, segment_value"""
    ))



@app.route("/api/portfolio_risk")
def portfolio_risk():
    positions = _rows(
        """SELECT city, size_usd, side, market_type FROM positions
           WHERE status NOT IN ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')"""
    )
    portfolio = _one("SELECT bankroll FROM portfolio WHERE id=1")
    bankroll = float(portfolio.get("bankroll") or 0.0)
    var95 = estimate_portfolio_var95(positions, bankroll, None, PARAMS)
    return jsonify({
        "bankroll": bankroll,
        "var95_usd": var95,
        "var95_pct": (var95 / bankroll) if bankroll > 0 else 0.0,
        "limit_pct": PARAMS.max_portfolio_var95_pct,
        "open_positions": len(positions),
    })

@app.route("/api/portfolio")
def portfolio():
    return jsonify(_one("SELECT * FROM portfolio WHERE id=1"))


@app.route("/api/positions")
def positions():
    status = request.args.get("status", "open")
    limit = int(request.args.get("limit", 50))
    if status == "open":
        where = """WHERE status NOT IN
            ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')"""
    elif status == "closed":
        where = """WHERE status IN
            ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')"""
    else:
        where = ""
    return jsonify(_rows(
        f"SELECT * FROM positions {where} ORDER BY opened_at DESC LIMIT ?",
        (limit,)
    ))


@app.route("/api/daily_pnl")
def daily_pnl():
    """Daily PnL history with running cumulative total."""
    return jsonify(_rows(
        """SELECT
               date,
               realized_pnl,
               num_trades,
               num_wins,
               CASE WHEN num_trades > 0
                    THEN ROUND(num_wins * 1.0 / num_trades, 3)
                    ELSE NULL END as win_rate,
               SUM(realized_pnl) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING)
                    as cumulative_pnl
           FROM daily_pnl
           ORDER BY date DESC
           LIMIT 90"""
    ))


@app.route("/api/trades/recent")
def recent_trades():
    limit = int(request.args.get("limit", 50))
    return jsonify(_rows(
        """SELECT p.*, pos.status, pos.realized_pnl as pos_pnl
           FROM predictions p
           LEFT JOIN positions pos ON pos.ticker = p.ticker
             AND pos.strategy_name = p.strategy_name
           WHERE p.is_shadow = 0
           ORDER BY p.created_at DESC
           LIMIT ?""",
        (limit,)
    ))


@app.route("/api/allocations")
def allocations():
    return jsonify(_rows(
        """SELECT strategy_name,
                  COUNT(*) as open_positions,
                  SUM(size_usd) as total_exposure
           FROM positions
           WHERE status NOT IN
               ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')
           GROUP BY strategy_name"""
    ))


@app.route("/api/cluster_exposure")
def cluster_exposure():
    return jsonify(_rows(
        """SELECT city,
                  COUNT(*) as open_positions,
                  SUM(size_usd) as exposure_usd
           FROM positions
           WHERE status NOT IN
               ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')
           GROUP BY city
           ORDER BY exposure_usd DESC"""
    ))


# --------------------------------------------------------------------------- #
# Tab 2: Strategy Performance
# --------------------------------------------------------------------------- #

@app.route("/api/strategy_metrics")
def strategy_metrics():
    return jsonify(_rows(
        """SELECT * FROM strategy_metrics
           ORDER BY computed_at DESC"""
    ))


@app.route("/api/strategy_performance")
def strategy_performance():
    return jsonify(_rows(
        """SELECT
               strategy_name,
               COUNT(*) as trade_count,
               SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate,
               SUM(realized_pnl) as total_pnl,
               AVG(realized_pnl) as avg_pnl,
               AVG(hold_time_hours) as avg_hold_hours,
               AVG(max_favorable_excursion) as avg_mfe,
               AVG(max_adverse_excursion) as avg_mae,
               is_shadow
           FROM positions
           WHERE realized_pnl IS NOT NULL
           GROUP BY strategy_name, is_shadow
           ORDER BY total_pnl DESC"""
    ))


@app.route("/api/brier_trend")
def brier_trend():
    strategy = request.args.get("strategy", "")
    where = "WHERE brier_score IS NOT NULL"
    params: tuple = ()
    if strategy:
        where += " AND strategy_name=?"
        params = (strategy,)
    return jsonify(_rows(
        f"""SELECT strategy_name,
                   DATE(created_at) as date,
                   AVG(brier_score) as avg_brier,
                   COUNT(*) as n
            FROM predictions
            {where}
            GROUP BY strategy_name, DATE(created_at)
            ORDER BY date""",
        params
    ))


@app.route("/api/mfe_mae")
def mfe_mae():
    strategy = request.args.get("strategy", "")
    where = "WHERE max_favorable_excursion IS NOT NULL AND max_adverse_excursion IS NOT NULL"
    params: tuple = ()
    if strategy:
        where += " AND strategy_name=?"
        params = (strategy,)
    return jsonify(_rows(
        f"""SELECT strategy_name, max_favorable_excursion as mfe,
                   max_adverse_excursion as mae, realized_pnl, city
            FROM positions {where}
            ORDER BY opened_at DESC LIMIT 500""",
        params
    ))


# --------------------------------------------------------------------------- #
# Tab 3: Autoresearch
# --------------------------------------------------------------------------- #

@app.route("/api/experiments")
def experiments():
    limit = int(request.args.get("limit", 50))
    return jsonify(_rows(
        """SELECT id, description, baseline_brier, candidate_brier,
                  improvement_pct, trade_count, status, created_at, completed_at
           FROM experiments
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,)
    ))


@app.route("/api/experiments/<exp_id>")
def experiment_detail(exp_id: str):
    return jsonify(_one(
        "SELECT * FROM experiments WHERE id=?", (exp_id,)
    ))


@app.route("/api/promotion_candidates")
def promotion_candidates():
    return jsonify(_rows(
        """SELECT * FROM strategy_metrics
           WHERE score > 75 AND is_live = 0
           ORDER BY score DESC"""
    ))


# --------------------------------------------------------------------------- #
# Tab 4: Forecast Quality
# --------------------------------------------------------------------------- #

@app.route("/api/forecast_bias")
def forecast_bias():
    return jsonify(_rows(
        """SELECT f.city, f.model_name,
                  COUNT(*) as n_obs,
                  AVG(f.predicted_high_f - s.actual_high_f) as bias_f,
                  AVG(ABS(f.predicted_high_f - s.actual_high_f)) as mae_f
           FROM forecasts f
           JOIN settlement_cache s
             ON s.city = f.city AND s.target_date = f.target_date
           WHERE f.predicted_high_f IS NOT NULL
             AND s.actual_high_f IS NOT NULL
           GROUP BY f.city, f.model_name
           HAVING COUNT(*) >= 5
           ORDER BY f.city, mae_f DESC"""
    ))


@app.route("/api/settlement_vs_forecast")
def settlement_vs_forecast():
    city = request.args.get("city", "")
    where = ""
    params: tuple = ()
    if city:
        where = "WHERE f.city=?"
        params = (city,)
    return jsonify(_rows(
        f"""SELECT f.city, f.model_name, f.target_date,
                   f.predicted_high_f, s.actual_high_f,
                   f.predicted_high_f - s.actual_high_f as residual
            FROM forecasts f
            JOIN settlement_cache s
              ON s.city = f.city AND s.target_date = f.target_date
            {where}
            ORDER BY f.target_date DESC
            LIMIT 200""",
        params
    ))


@app.route("/api/model_agreement")
def model_agreement():
    return jsonify(_rows(
        """SELECT city, target_date,
                  COUNT(DISTINCT model_name) as n_models,
                  AVG(predicted_high_f) as mean_high_f,
                  MAX(predicted_high_f) - MIN(predicted_high_f) as spread_f
           FROM forecasts
           GROUP BY city, target_date
           ORDER BY target_date DESC, city
           LIMIT 100"""
    ))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Last-cycle panel + quick-stats
# --------------------------------------------------------------------------- #

@app.route("/api/last_cycle")
def last_cycle():
    return jsonify(_LAST_CYCLE)


@app.route("/api/last_cycle", methods=["POST"])
def update_last_cycle():
    global _LAST_CYCLE
    _LAST_CYCLE = request.get_json(force=True) or {}
    return jsonify({"status": "ok"})


@app.route("/api/quick_stats")
def quick_stats():
    row = _one("""
        SELECT
            p.bankroll,
            p.total_pnl,
            (SELECT COUNT(*) FROM positions WHERE status NOT IN
                ('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')
            ) as open_positions,
            (SELECT COUNT(*) FROM positions WHERE DATE(opened_at) = DATE('now')
            ) as trades_today,
            (SELECT COALESCE(SUM(realized_pnl), 0) FROM positions
             WHERE DATE(closed_at) = DATE('now') AND realized_pnl IS NOT NULL
            ) as daily_pnl,
            (SELECT ROUND(AVG(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 1)
             FROM positions WHERE realized_pnl IS NOT NULL
               AND closed_at > datetime('now', '-30 days')
            ) as win_rate_30d,
            (SELECT COUNT(*) FROM positions WHERE realized_pnl IS NOT NULL
            ) as closed_positions,
            (SELECT COUNT(*) FROM positions WHERE realized_pnl > 0
            ) as winning_positions,
            (SELECT COALESCE(SUM(realized_pnl), 0) FROM positions
             WHERE realized_pnl IS NOT NULL
            ) as realized_pnl,
            (SELECT COUNT(*) FROM experiments
            ) as experiments_run
        FROM portfolio p WHERE p.id = 1
    """)
    row["last_cycle"] = _LAST_CYCLE
    # Derived win_rate for backward compat
    closed = row.get("closed_positions") or 0
    winning = row.get("winning_positions") or 0
    row["win_rate"] = round(winning / closed, 3) if closed else None
    return jsonify(row)


# --------------------------------------------------------------------------- #
# Autoresearch trigger
# --------------------------------------------------------------------------- #

@app.route("/api/autoresearch/run", methods=["POST"])
def run_autoresearch():
    """Trigger one autoresearch experiment cycle from the dashboard."""
    try:
        from research.autoresearch import ExperimentRegistry
        from research.calibrator import _build_trade_log
        conn = _conn()
        trade_log = _build_trade_log(conn)
        if len(trade_log) < 6:
            return jsonify({
                "error": f"Not enough resolved trades ({len(trade_log)}) to run experiments"
            }), 400
        registry = ExperimentRegistry(conn, n_windows=5)
        result = registry.run_cycle(trade_log)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #

@app.route("/api/health")
def health():
    try:
        with _conn() as c:
            c.execute("SELECT 1")
        return jsonify({"status": "ok", "db": DB_PATH})
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500


def run(port: int = 5001, debug: bool = False) -> None:
    print(f"[dashboard] API server running on http://localhost:{port}")
    print(f"[dashboard] Reading from: {DB_PATH}")
    print(f"[dashboard] Open dashboard/index.html in your browser")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(port=args.port, debug=args.debug)
