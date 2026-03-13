"""
Signal generation: converts forecasts + market prices into tradeable edges.

OLD: edge = model_prob - market_price
NEW: executable_edge = model_prob - exec_info.executable_price

The executable edge accounts for fees, slippage, and orderbook depth,
giving the *real* edge we capture after all transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execution.orderbook import ExecutionInfo, OrderbookLevel, get_executable_price
from shared.params import PARAMS, Params


@dataclass
class EdgeResult:
    """Full edge breakdown for a single market opportunity."""
    ticker: str
    side: str
    model_prob: float           # our probability estimate
    market_price: float         # current mid-market price
    raw_edge: float             # model_prob - market_price (old metric, kept for reference)
    executable_edge: float      # model_prob - executable_price (real edge after costs)
    exec_info: ExecutionInfo    # full cost breakdown
    passes_filter: bool         # True if executable_edge >= min_executable_edge


def calculate_edge(
    ticker: str,
    side: str,
    model_prob: float,
    market_price: float,
    target_size_usd: float = 100.0,
    orderbook: Optional[list[OrderbookLevel]] = None,
    params: Optional[Params] = None,
) -> EdgeResult:
    """
    Calculate the executable edge for a potential trade.

    Args:
        ticker:           Market ticker.
        side:             "YES" or "NO".
        model_prob:       Our model's probability (0.0–1.0).
        market_price:     Current market mid-price (0.0–1.0).
        target_size_usd:  Intended trade size for VWAP calculation.
        orderbook:        Optional live orderbook levels.
        params:           Params instance (defaults to module singleton).

    Returns:
        EdgeResult with raw_edge, executable_edge, and cost breakdown.
    """
    p = params or PARAMS

    exec_info = get_executable_price(
        ticker=ticker,
        side=side,
        target_size_usd=target_size_usd,
        fallback_price=market_price,
        orderbook=orderbook,
        params=p,
    )

    raw_edge = model_prob - market_price
    executable_edge = model_prob - exec_info.executable_price

    passes = (
        executable_edge >= p.min_executable_edge
        and exec_info.is_liquid
    )

    return EdgeResult(
        ticker=ticker,
        side=side,
        model_prob=model_prob,
        market_price=market_price,
        raw_edge=raw_edge,
        executable_edge=executable_edge,
        exec_info=exec_info,
        passes_filter=passes,
    )


def select_side(model_prob: float, market_price: float) -> str:
    """Return 'YES' if model is above market, 'NO' if below."""
    return "YES" if model_prob > market_price else "NO"


def find_opportunities(
    markets: list[dict],
    model_probs: dict[str, float],
    params: Optional[Params] = None,
) -> list[EdgeResult]:
    """
    Scan all markets and return those with positive executable edge.

    Args:
        markets:     List of market dicts with keys: ticker, market_price,
                     and optionally orderbook (list of {price, size_usd}).
        model_probs: Map of ticker → model probability.
        params:      Params instance.

    Returns:
        List of EdgeResult sorted by executable_edge descending.
    """
    p = params or PARAMS
    results: list[EdgeResult] = []

    for market in markets:
        ticker = market["ticker"]
        market_price = market["market_price"]
        model_prob = model_probs.get(ticker)
        if model_prob is None:
            continue

        side = select_side(model_prob, market_price)

        raw_ob = market.get("orderbook")
        orderbook = None
        if raw_ob:
            orderbook = [
                OrderbookLevel(price=lv["price"], size_usd=lv["size_usd"])
                for lv in raw_ob
            ]

        edge = calculate_edge(
            ticker=ticker,
            side=side,
            model_prob=model_prob,
            market_price=market_price,
            target_size_usd=market.get("target_size_usd", 100.0),
            orderbook=orderbook,
            params=p,
        )
        results.append(edge)

    results.sort(key=lambda e: e.executable_edge, reverse=True)
    return results
