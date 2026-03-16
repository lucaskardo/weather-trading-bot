"""
SQLite state management for the weather trading bot.
Replaces JSON file state with crash-safe WAL-mode SQLite.
"""

import sqlite3
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DB_PATH = Path(__file__).parent.parent / "data" / "bot.db"

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Return a thread-local connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = _open_connection(DB_PATH)
    return _local.conn


def _open_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def transaction() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on success, rolls back on exception."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Create all tables if they don't exist. Returns the connection."""
    target = path or DB_PATH
    conn = _open_connection(target)
    # Store as thread-local if using the default path
    if path is None:
        _local.conn = conn

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            id                TEXT PRIMARY KEY,
            ticker            TEXT NOT NULL,
            city              TEXT NOT NULL,
            target_date       TEXT NOT NULL,
            market_type       TEXT NOT NULL DEFAULT 'high_temp',
            low_f             REAL,
            high_f            REAL,
            nws_station       TEXT,
            exchange          TEXT NOT NULL DEFAULT 'kalshi',
            status            TEXT NOT NULL DEFAULT 'open',
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id         TEXT REFERENCES markets(id),
            city              TEXT NOT NULL,
            target_date       TEXT NOT NULL,
            model_name        TEXT NOT NULL,
            predicted_high_f  REAL NOT NULL,
            predicted_low_f   REAL,
            confidence        REAL,
            ensemble_members_json TEXT,
            run_id            TEXT,
            publish_time      TEXT,
            source_url        TEXT,
            fetched_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name     TEXT NOT NULL,
            market_id         TEXT REFERENCES markets(id),
            ticker            TEXT NOT NULL,
            city              TEXT NOT NULL,
            target_date       TEXT NOT NULL,
            fair_value        REAL NOT NULL,
            market_price      REAL NOT NULL,
            executable_price  REAL,
            edge              REAL NOT NULL,
            executable_edge   REAL,
            confidence        REAL,
            consensus_f       REAL,
            agreement         REAL,
            n_models          INTEGER,
            is_shadow         INTEGER NOT NULL DEFAULT 0,
            outcome           REAL,
            actual_high_f     REAL,
            brier_score       REAL,
            realized_pnl      REAL,
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS positions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id     INTEGER REFERENCES predictions(id),
            strategy_name     TEXT NOT NULL,
            market_id         TEXT REFERENCES markets(id),
            ticker            TEXT NOT NULL,
            city              TEXT NOT NULL,
            target_date       TEXT NOT NULL,
            high_f            REAL,
            low_f             REAL,
            market_type       TEXT,
            exchange          TEXT,
            side              TEXT NOT NULL,
            size_usd          REAL NOT NULL,
            entry_price       REAL NOT NULL,
            current_price     REAL,
            exit_price        REAL,
            status            TEXT NOT NULL DEFAULT 'OPENED',
            entry_reason      TEXT,
            exit_reason       TEXT,
            hold_time_hours   REAL,
            max_favorable_excursion REAL,
            max_adverse_excursion   REAL,
            realized_pnl      REAL,
            opened_at         TEXT NOT NULL DEFAULT (datetime('now')),
            closed_at         TEXT,
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS orders (
            id                TEXT PRIMARY KEY,
            position_id       INTEGER REFERENCES positions(id),
            ticker            TEXT NOT NULL,
            side              TEXT NOT NULL,
            order_type        TEXT NOT NULL DEFAULT 'limit',
            requested_price   REAL NOT NULL,
            requested_count   INTEGER NOT NULL,
            status            TEXT NOT NULL DEFAULT 'pending',
            exchange_order_id TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS fills (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id          TEXT REFERENCES orders(id),
            position_id       INTEGER REFERENCES positions(id),
            fill_price        REAL NOT NULL,
            fill_count        INTEGER NOT NULL,
            fees_paid         REAL NOT NULL DEFAULT 0.0,
            execution_type    TEXT DEFAULT 'taker',
            filled_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id                INTEGER PRIMARY KEY CHECK (id = 1),
            bankroll          REAL NOT NULL DEFAULT 1000.0,
            cash_available    REAL NOT NULL DEFAULT 1000.0,
            total_pnl         REAL NOT NULL DEFAULT 0.0,
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS daily_pnl (
            date              TEXT PRIMARY KEY,
            realized_pnl      REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl    REAL NOT NULL DEFAULT 0.0,
            num_trades        INTEGER NOT NULL DEFAULT 0,
            num_wins          INTEGER NOT NULL DEFAULT 0,
            bankroll_eod      REAL,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS strategy_metrics (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name     TEXT NOT NULL,
            computed_at       TEXT NOT NULL DEFAULT (datetime('now')),
            trade_count       INTEGER NOT NULL DEFAULT 0,
            win_rate          REAL,
            sharpe            REAL,
            avg_brier         REAL,
            avg_edge          REAL,
            max_drawdown      REAL,
            score             REAL,
            is_live           INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS settlement_cache (
            city              TEXT NOT NULL,
            target_date       TEXT NOT NULL,
            actual_high_f     REAL NOT NULL,
            station           TEXT NOT NULL,
            source_url        TEXT,
            fetched_at        TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (city, target_date)
        );



        CREATE TABLE IF NOT EXISTS decision_audit (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id     INTEGER REFERENCES predictions(id),
            position_id       INTEGER REFERENCES positions(id),
            strategy_name     TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            city              TEXT,
            target_date       TEXT,
            provider_publish_time TEXT,
            model_run_time    TEXT,
            bot_fetch_time    TEXT,
            parse_to_signal_time TEXT,
            market_snapshot_time TEXT,
            order_sent_time   TEXT,
            fill_received_time TEXT,
            revision_confirmed INTEGER NOT NULL DEFAULT 0,
            revision_delta_f  REAL,
            source_models     TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS calibration_segments (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at       TEXT NOT NULL DEFAULT (datetime('now')),
            segment_kind      TEXT NOT NULL,
            segment_value     TEXT NOT NULL,
            trade_count       INTEGER NOT NULL DEFAULT 0,
            avg_brier         REAL,
            avg_outcome       REAL,
            avg_prediction    REAL
        );

        CREATE TABLE IF NOT EXISTS calibration_profiles (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at       TEXT NOT NULL DEFAULT (datetime('now')),
            segment_kind      TEXT NOT NULL,
            segment_value     TEXT NOT NULL,
            trade_count       INTEGER NOT NULL DEFAULT 0,
            avg_brier         REAL,
            avg_outcome       REAL,
            avg_prediction    REAL,
            prob_adjustment   REAL NOT NULL DEFAULT 0.0
        );


        CREATE TABLE IF NOT EXISTS experiments (
            id            TEXT PRIMARY KEY,
            description   TEXT,
            params_json   TEXT NOT NULL,
            baseline_brier  REAL,
            candidate_brier REAL,
            improvement_pct REAL,
            trade_count   INTEGER,
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS promoted_params (
            key        TEXT PRIMARY KEY,
            value      REAL NOT NULL,
            source_exp TEXT,
            promoted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        INSERT OR IGNORE INTO portfolio (id, bankroll, cash_available, total_pnl)
        VALUES (1, 1000.0, 1000.0, 0.0);
    """)
    conn.commit()
    _migrate_schema(conn)
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """
    Apply incremental schema migrations for existing databases.
    Uses ALTER TABLE ADD COLUMN which is a no-op if column already exists
    (wrapped in try/except for SQLite compatibility).
    """
    _add_column_if_missing(conn, "positions", "high_f", "REAL")
    _add_column_if_missing(conn, "positions", "low_f", "REAL")
    _add_column_if_missing(conn, "positions", "market_type", "TEXT")
    _add_column_if_missing(conn, "positions", "exchange", "TEXT")
    # decision_audit currently created as a full table above; no incremental columns needed yet
    conn.commit()


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add a column to a table if it doesn't already exist."""
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def assert_db_integrity(conn: sqlite3.Connection | None = None) -> None:
    """Run PRAGMA integrity_check and raise if not OK."""
    c = conn or get_connection()
    result = c.execute("PRAGMA integrity_check").fetchone()
    if result[0] != "ok":
        raise RuntimeError(f"DB integrity check failed: {result[0]}")


def migrate_from_json(
    json_dir: str | Path | None = None,
    db_path: Path | None = None,
) -> dict[str, int]:
    """
    Import existing JSON state files into SQLite on first run.
    Returns counts of records migrated per entity type.
    """
    counts: dict[str, int] = {"markets": 0, "predictions": 0, "positions": 0}

    if json_dir is None:
        json_dir = Path(__file__).parent.parent / "state_legacy"
    json_dir = Path(json_dir)

    conn = get_connection() if db_path is None else _open_connection(db_path)

    # Migrate markets
    markets_file = json_dir / "markets.json"
    if markets_file.exists():
        with open(markets_file) as f:
            markets: list[dict[str, Any]] = json.load(f)
        for m in markets:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO markets
                       (id, ticker, city, target_date, market_type, low_f, high_f,
                        nws_station, exchange, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        m.get("id", m.get("ticker", "")),
                        m.get("ticker", ""),
                        m.get("city", ""),
                        m.get("target_date", ""),
                        m.get("market_type", "high_temp"),
                        m.get("low_f"),
                        m.get("high_f"),
                        m.get("nws_station"),
                        m.get("exchange", "kalshi"),
                        m.get("status", "open"),
                    ),
                )
                counts["markets"] += 1
            except sqlite3.IntegrityError:
                pass

    # Migrate predictions (from legacy trade log JSONL or JSON list)
    for fname in ["predictions.json", "trade_log.jsonl", "trade_log.json"]:
        pred_file = json_dir / fname
        if not pred_file.exists():
            continue
        try:
            if fname.endswith(".jsonl"):
                with open(pred_file) as f:
                    preds = [json.loads(line) for line in f if line.strip()]
            else:
                with open(pred_file) as f:
                    preds = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for p in preds:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO predictions
                       (strategy_name, ticker, city, target_date, fair_value,
                        market_price, edge, is_shadow, outcome, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p.get("strategy_name", "legacy"),
                        p.get("ticker", ""),
                        p.get("city", ""),
                        p.get("target_date", ""),
                        p.get("fair_value", p.get("model_prob", 0.5)),
                        p.get("market_price", 0.5),
                        p.get("edge", 0.0),
                        p.get("is_shadow", 0),
                        p.get("outcome"),
                        p.get("created_at", ""),
                    ),
                )
                counts["predictions"] += 1
            except sqlite3.IntegrityError:
                pass

    conn.commit()
    return counts
