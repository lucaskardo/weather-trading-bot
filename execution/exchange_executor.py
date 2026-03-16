"""
Exchange executor abstraction.

Provides a clean ABC so Brain never calls exchange APIs directly.
Swap PaperExecutor for KalshiExecutor by passing the right instance
to Brain.__init__().

Usage:
    # Paper mode (default)
    executor = PaperExecutor()

    # Live Kalshi mode
    executor = KalshiExecutor(api_key=..., private_key=...)

    brain = Brain(conn, params, executor=executor)
"""

from __future__ import annotations

import abc
import json
from datetime import datetime, timezone
from typing import Any


class ExchangeExecutor(abc.ABC):
    """Abstract executor — placed an order, return fill details."""

    @abc.abstractmethod
    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = False,
        depth_usd: float | None = None,
    ) -> dict[str, Any]:
        """
        Place an order on the exchange.

        Args:
            ticker:   Contract ticker, e.g. "KXHIGH-25JAN15-NYC75".
            side:     "YES" or "NO".
            size_usd: Dollar amount to risk.
            price:    Limit price (0–1 as a probability).
            dry_run:  If True, simulate without sending to exchange.

        Returns:
            dict with at minimum:
              {
                "status": "filled" | "partial" | "rejected" | "simulated",
                "fill_price": float,
                "fill_size_usd": float,
                "order_id": str | None,
              }
        """
        ...


class PaperExecutor(ExchangeExecutor):
    """
    Paper executor — simulates fills locally, never calls an exchange.

    Used for backtesting and paper trading. Uses a deterministic depth-aware
    partial-fill model when estimated depth is available so paper results are
    a bit less optimistic in thin books.
    """

    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = False,
        depth_usd: float | None = None,
    ) -> dict[str, Any]:
        fill_ratio = 1.0
        if depth_usd is not None and size_usd > 0:
            fill_ratio = max(0.0, min(1.0, float(depth_usd) / float(size_usd)))

        if fill_ratio <= 0.0:
            return {
                "status": "rejected",
                "fill_price": price,
                "fill_size_usd": 0.0,
                "fill_ratio": 0.0,
                "requested_size_usd": size_usd,
                "order_id": None,
                "simulated_at": datetime.now(timezone.utc).isoformat(),
            }

        status = "simulated" if fill_ratio >= 0.999 else "partial"
        return {
            "status": status,
            "fill_price": price,
            "fill_size_usd": size_usd * fill_ratio,
            "fill_ratio": fill_ratio,
            "requested_size_usd": size_usd,
            "order_id": None,
            "simulated_at": datetime.now(timezone.utc).isoformat(),
        }


class MakerFirstExecutor(ExchangeExecutor):
    """
    Maker-first execution wrapper.

    Wraps any ExchangeExecutor and implements passive-first order flow:
    1. Try to place a post_only (maker) order at a slightly better price
    2. In paper mode: fill immediately at the maker price (no wait)
    3. In live mode: the caller is responsible for polling next cycle
       and calling taker_fallback() if unfilled

    The fills table records execution_type='maker' or 'taker'.
    Scorecard gives bonus to strategies with high maker fill rate.
    """

    def __init__(
        self,
        inner: ExchangeExecutor,
        maker_improvement: float = 0.005,   # improve price by 0.5¢
    ):
        self.inner = inner
        self.maker_improvement = maker_improvement

    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = False,
        depth_usd: float | None = None,
    ) -> dict[str, Any]:
        """
        Attempt maker order first. In paper/dry_run, fills at maker price.
        In live, places post_only and returns 'pending_maker' status for
        the caller to track next cycle.
        """
        # Maker price: bid slightly below for YES, above for NO
        if side.upper() == "YES":
            maker_price = max(0.01, price - self.maker_improvement)
        else:
            maker_price = min(0.99, price + self.maker_improvement)

        if dry_run:
            # Paper maker simulation: optimistic on price, conservative on fill ratio.
            maker_depth = None if depth_usd is None else max(0.0, depth_usd * 0.6)
            result = self.inner.place_order(
                ticker=ticker,
                side=side,
                size_usd=size_usd,
                price=maker_price,
                dry_run=True,
                depth_usd=maker_depth,
            )
            result["execution_type"] = "maker"
            if result.get("status") == "simulated":
                result["status"] = "simulated_maker"
            return result

        # Live: delegate to inner executor with maker price
        result = self.inner.place_order(
            ticker=ticker,
            side=side,
            size_usd=size_usd,
            price=maker_price,
            dry_run=False,
            depth_usd=depth_usd,
        )
        result["execution_type"] = "maker"
        result["maker_price"] = maker_price
        result["taker_fallback_price"] = price
        return result


class KalshiExecutor(ExchangeExecutor):
    """
    Live Kalshi executor.

    Requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY environment variables
    (or pass them explicitly). Calls the Kalshi REST API to place limit
    orders and returns fill details from the response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        private_key: str | None = None,
        base_url: str = "https://trading-api.kalshi.com/trade-api/v2",
    ) -> None:
        import os
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY", "")
        self.private_key = private_key or os.environ.get("KALSHI_PRIVATE_KEY", "")
        self.base_url = base_url

        if not self.api_key or not self.private_key:
            raise ValueError(
                "KalshiExecutor requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY"
            )

    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = False,
        depth_usd: float | None = None,
    ) -> dict[str, Any]:
        """Place a limit order on Kalshi. Returns fill details."""
        if dry_run:
            return {
                "status": "simulated",
                "fill_price": price,
                "fill_size_usd": size_usd,
                "fill_ratio": 1.0,
                "requested_size_usd": size_usd,
                "order_id": None,
            }

        try:
            import requests
        except ImportError:
            raise RuntimeError("requests library required: pip install requests")

        # Kalshi uses cents (0–100) not probability (0–1)
        price_cents = round(price * 100)

        # Side-aware contract count: risk per YES contract = price, NO = (1 - price)
        from execution.orderbook import kalshi_taker_fee
        risk_per_contract = price if side.upper() == "YES" else (1.0 - price)
        fee_per_contract = kalshi_taker_fee(price)
        cost_per_contract = risk_per_contract + fee_per_contract
        count = max(1, int(size_usd / max(cost_per_contract, 0.01)))

        payload = {
            "ticker": ticker,
            "side": side.lower(),   # "yes" | "no"
            "type": "limit",
            "count": count,
            "yes_price": price_cents if side.upper() == "YES" else (100 - price_cents),
            "action": "buy",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(
            f"{self.base_url}/portfolio/orders",
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        order = data.get("order", {})
        fill_price_cents = order.get("yes_price", price_cents)
        fill_price = fill_price_cents / 100.0
        fill_count = order.get("count", count)

        return {
            "status": order.get("status", "unknown"),
            "fill_price": fill_price,
            "fill_size_usd": float(fill_count),
            "fill_ratio": 1.0,
            "requested_size_usd": size_usd,
            "order_id": order.get("order_id"),
        }
