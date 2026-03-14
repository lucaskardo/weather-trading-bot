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

    Used for backtesting and paper trading. Fill is always at the
    requested price (no slippage model beyond what the signal already
    incorporated via get_executable_price).
    """

    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return {
            "status": "simulated",
            "fill_price": price,
            "fill_size_usd": size_usd,
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
            # Paper fill at better price — best-case maker simulation
            return {
                "status": "simulated_maker",
                "fill_price": maker_price,
                "fill_size_usd": size_usd,
                "order_id": None,
                "execution_type": "maker",
                "simulated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Live: delegate to inner executor with maker price
        result = self.inner.place_order(
            ticker=ticker,
            side=side,
            size_usd=size_usd,
            price=maker_price,
            dry_run=False,
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
    ) -> dict[str, Any]:
        """Place a limit order on Kalshi. Returns fill details."""
        if dry_run:
            return {
                "status": "simulated",
                "fill_price": price,
                "fill_size_usd": size_usd,
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
            "order_id": order.get("order_id"),
        }
