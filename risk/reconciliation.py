"""
Phase 4.2 — Reconciliation.

Daily sync of local DB positions against exchange APIs.

Workflow:
  1. Load all non-terminal positions from local DB
  2. Query exchange API for current position/order state
  3. Detect discrepancies (missing locally, missing on exchange, price mismatch)
  4. Log CRITICAL for any discrepancy
  5. Force-update local DB to match exchange truth
  6. Detect orphaned orders (on exchange but not in DB)

Runs at 00:00 UTC daily and on bot startup.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class DiscrepancyType(str, Enum):
    MISSING_LOCALLY = "missing_locally"       # on exchange, not in local DB
    MISSING_ON_EXCHANGE = "missing_on_exchange"  # in local DB, not on exchange
    PRICE_MISMATCH = "price_mismatch"         # entry price differs > threshold
    STATUS_MISMATCH = "status_mismatch"       # local says HOLDING, exchange says closed
    ORPHANED_ORDER = "orphaned_order"         # open order on exchange without DB entry


class Discrepancy:
    """A single reconciliation discrepancy."""

    def __init__(
        self,
        discrepancy_type: DiscrepancyType,
        ticker: str,
        local_data: Optional[dict[str, Any]],
        exchange_data: Optional[dict[str, Any]],
        detail: str = "",
    ):
        self.discrepancy_type = discrepancy_type
        self.ticker = ticker
        self.local_data = local_data
        self.exchange_data = exchange_data
        self.detail = detail
        self.detected_at = datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        return (
            f"Discrepancy({self.discrepancy_type}, ticker={self.ticker}, "
            f"detail={self.detail!r})"
        )

    def is_critical(self) -> bool:
        """Returns True for discrepancies that require immediate intervention."""
        return self.discrepancy_type in {
            DiscrepancyType.MISSING_LOCALLY,
            DiscrepancyType.STATUS_MISMATCH,
        }


class ReconciliationResult:
    """Summary of one reconciliation run."""

    def __init__(
        self,
        local_positions: int,
        exchange_positions: int,
        discrepancies: list[Discrepancy],
        corrections: int,
        reconciled_at: str,
    ):
        self.local_positions = local_positions
        self.exchange_positions = exchange_positions
        self.discrepancies = discrepancies
        self.corrections = corrections
        self.reconciled_at = reconciled_at

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.is_critical())

    def __repr__(self) -> str:
        return (
            f"ReconciliationResult(local={self.local_positions}, "
            f"exchange={self.exchange_positions}, "
            f"discrepancies={len(self.discrepancies)}, "
            f"critical={self.critical_count})"
        )


_PRICE_MISMATCH_THRESHOLD = 0.02   # 2 cents
_NON_TERMINAL_STATUSES = {
    "OPENED", "HOLDING", "TAKE_PROFIT_PARTIAL"
}


def reconcile_positions(
    conn: sqlite3.Connection,
    exchange_positions: list[dict[str, Any]],
    auto_correct: bool = True,
) -> ReconciliationResult:
    """
    Reconcile local DB positions against exchange-reported positions.

    Args:
        conn:               SQLite connection.
        exchange_positions: List of position dicts from exchange API.
                            Each must have: ticker, side, entry_price, status.
                            Optional: current_price, size_usd.
        auto_correct:       If True, update local DB to match exchange truth.

    Returns:
        ReconciliationResult with discrepancy list and correction count.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Load local open positions
    rows = conn.execute(
        """SELECT id, ticker, side, entry_price, current_price, size_usd, status, city
           FROM positions
           WHERE status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')"""
    ).fetchall()
    local = {r["ticker"]: dict(r) for r in rows}

    # Index exchange positions by ticker
    exchange = {p["ticker"]: p for p in exchange_positions}

    discrepancies: list[Discrepancy] = []
    corrections = 0

    # --- Check local against exchange ---
    for ticker, local_pos in local.items():
        if ticker not in exchange:
            d = Discrepancy(
                discrepancy_type=DiscrepancyType.MISSING_ON_EXCHANGE,
                ticker=ticker,
                local_data=local_pos,
                exchange_data=None,
                detail=f"Position {ticker} is in local DB but not on exchange",
            )
            discrepancies.append(d)
            _log_discrepancy(d)

            if auto_correct:
                # Mark as EXITED_STOP if exchange has no record (likely already settled/closed)
                conn.execute(
                    "UPDATE positions SET status='EXITED_STOP', updated_at=? WHERE ticker=? AND status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')",
                    (now, ticker),
                )
                corrections += 1
        else:
            ex_pos = exchange[ticker]

            # Status mismatch
            ex_status = ex_pos.get("status", "").upper()
            local_status = local_pos["status"]
            if ex_status in ("CLOSED", "SETTLED", "EXPIRED") and local_status in _NON_TERMINAL_STATUSES:
                d = Discrepancy(
                    discrepancy_type=DiscrepancyType.STATUS_MISMATCH,
                    ticker=ticker,
                    local_data=local_pos,
                    exchange_data=ex_pos,
                    detail=f"Local={local_status}, Exchange={ex_status}",
                )
                discrepancies.append(d)
                _log_discrepancy(d)

                if auto_correct:
                    new_status = "WON" if ex_pos.get("won") else "LOST"
                    conn.execute(
                        "UPDATE positions SET status=?, updated_at=? WHERE ticker=? AND status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')",
                        (new_status, now, ticker),
                    )
                    corrections += 1

            # Price mismatch
            ex_price = ex_pos.get("entry_price")
            local_price = local_pos.get("entry_price")
            if ex_price is not None and local_price is not None:
                if abs(float(ex_price) - float(local_price)) > _PRICE_MISMATCH_THRESHOLD:
                    d = Discrepancy(
                        discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                        ticker=ticker,
                        local_data=local_pos,
                        exchange_data=ex_pos,
                        detail=f"local_price={local_price:.4f}, exchange_price={ex_price:.4f}",
                    )
                    discrepancies.append(d)
                    _log_discrepancy(d)

                    if auto_correct:
                        conn.execute(
                            "UPDATE positions SET entry_price=?, updated_at=? WHERE ticker=? AND status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')",
                            (float(ex_price), now, ticker),
                        )
                        corrections += 1

            # Update current price if exchange provides it
            if auto_correct and ex_pos.get("current_price") is not None:
                conn.execute(
                    "UPDATE positions SET current_price=?, updated_at=? WHERE ticker=? AND status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')",
                    (float(ex_pos["current_price"]), now, ticker),
                )

    # --- Check exchange against local (missing locally / orphaned) ---
    for ticker, ex_pos in exchange.items():
        if ticker not in local:
            ex_status = ex_pos.get("status", "").upper()
            if ex_status not in ("CLOSED", "SETTLED", "EXPIRED", "CANCELLED"):
                d = Discrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_LOCALLY,
                    ticker=ticker,
                    local_data=None,
                    exchange_data=ex_pos,
                    detail=f"Open position on exchange not found in local DB",
                )
                discrepancies.append(d)
                _log_discrepancy(d)

                if auto_correct:
                    _insert_missing_position(conn, ex_pos, now)
                    corrections += 1

    if corrections > 0:
        conn.commit()

    return ReconciliationResult(
        local_positions=len(local),
        exchange_positions=len(exchange_positions),
        discrepancies=discrepancies,
        corrections=corrections,
        reconciled_at=now,
    )


def check_orphaned_orders(
    conn: sqlite3.Connection,
    exchange_orders: list[dict[str, Any]],
) -> list[Discrepancy]:
    """
    Find open orders on the exchange that have no corresponding DB entry.
    These are orphaned and should be cancelled manually.
    """
    local_tickers = {
        r["ticker"] for r in conn.execute(
            "SELECT ticker FROM positions WHERE status IN ('OPENED','HOLDING','TAKE_PROFIT_PARTIAL')"
        ).fetchall()
    }

    orphans: list[Discrepancy] = []
    for order in exchange_orders:
        ticker = order.get("ticker", "")
        if ticker not in local_tickers:
            d = Discrepancy(
                discrepancy_type=DiscrepancyType.ORPHANED_ORDER,
                ticker=ticker,
                local_data=None,
                exchange_data=order,
                detail=f"Open order on exchange with no local DB record",
            )
            orphans.append(d)
            _log_discrepancy(d)

    return orphans


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _log_discrepancy(d: Discrepancy) -> None:
    import sys
    level = "CRITICAL" if d.is_critical() else "WARNING"
    print(
        f"[reconciliation] {level}: {d.discrepancy_type} | {d.ticker} | {d.detail}",
        file=sys.stderr,
    )


def _insert_missing_position(
    conn: sqlite3.Connection,
    ex_pos: dict[str, Any],
    now: str,
) -> None:
    """Insert an exchange position that is missing from local DB."""
    conn.execute(
        """INSERT OR IGNORE INTO positions
           (ticker, city, target_date, side, size_usd, entry_price,
            current_price, status, strategy_name, opened_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ex_pos.get("ticker", ""),
            ex_pos.get("city", ""),
            ex_pos.get("target_date", ""),
            ex_pos.get("side", "YES"),
            float(ex_pos.get("size_usd", 0.0)),
            float(ex_pos.get("entry_price", 0.5)),
            float(ex_pos.get("current_price", ex_pos.get("entry_price", 0.5))),
            "HOLDING",
            ex_pos.get("strategy_name", "unknown"),
            now,
            now,
        ),
    )
