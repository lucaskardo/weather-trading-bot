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
    portfolio = _one("SELECT * FROM portfolio WHERE id=1")
    open_count = _one("SELECT COUNT(*) as n FROM positions WHERE status NOT IN "
                      "('WON','LOST','EXITED_CONVERGENCE','EXITED_STOP','EXITED_PRE_SETTLEMENT')")
    closed_count = _one("SELECT COUNT(*) as n FROM positions WHERE realized_pnl IS NOT NULL")
    win_count = _one("SELECT COUNT(*) as n FROM positions WHERE realized_pnl > 0")
    total_pnl = _one("SELECT SUM(realized_pnl) as total FROM positions WHERE realized_pnl IS NOT NULL")
    exp_count = _one("SELECT COUNT(*) as n FROM experiments")
    return jsonify({
        "bankroll": portfolio.get("bankroll"),
        "total_pnl": portfolio.get("total_pnl"),
        "open_positions": open_count.get("n", 0),
        "closed_positions": closed_count.get("n", 0),
        "winning_positions": win_count.get("n", 0),
        "win_rate": (
            round(win_count.get("n", 0) / closed_count["n"], 3)
            if closed_count.get("n") else None
        ),
        "realized_pnl": total_pnl.get("total"),
        "experiments_run": exp_count.get("n", 0),
        "last_cycle": _LAST_CYCLE,
    })


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
