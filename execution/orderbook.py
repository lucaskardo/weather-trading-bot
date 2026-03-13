"""
Orderbook-aware execution pricing.

Computes the true cost to execute a trade by walking the orderbook
(if available) or falling back to mid-price + slippage + fees.

All prices are in cents-per-dollar (0.0–1.0 probability space).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.params import PARAMS, Params


@dataclass
class OrderbookLevel:
    """A single price level in an orderbook."""
    price: float   # 0.0–1.0
    size_usd: float


@dataclass
class ExecutionInfo:
    """Result of get_executable_price."""
    executable_price: float   # price we expect to pay/receive after fees+slippage
    vwap_price: float         # volume-weighted average price across depth consumed
    depth_usd: float          # total liquidity available at/better than target price
    fees_est: float           # estimated fee cost (fraction of notional)
    slippage_est: float       # estimated slippage vs mid (fraction)
    is_liquid: bool           # True if depth_usd >= min_depth_usd


def get_executable_price(
    ticker: str,
    side: str,
    target_size_usd: float,
    fallback_price: float,
    orderbook: Optional[list[OrderbookLevel]] = None,
    params: Optional[Params] = None,
) -> ExecutionInfo:
    """
    Compute the true executable price for a trade.

    Args:
        ticker:           Market ticker (informational; used for logging).
        side:             "YES" or "NO".
        target_size_usd:  Notional size we want to trade in USD.
        fallback_price:   Mid-market price (0.0–1.0) to use when no orderbook.
        orderbook:        Optional list of OrderbookLevel sorted by price.
                          For a YES buy: ascending price (cheapest first).
                          For a NO buy:  ascending price.
                          If None, uses fallback_price + slippage.
        params:           Params instance. Defaults to module singleton.

    Returns:
        ExecutionInfo with all cost components populated.
    """
    p = params or PARAMS
    fee_rate = p.taker_fee_pct
    slippage_cents = p.slippage_buffer_cents / 100.0  # convert cents → fraction
    min_depth = p.min_depth_usd

    if orderbook:
        vwap, depth = _compute_vwap(orderbook, target_size_usd)
    else:
        vwap = fallback_price
        depth = 0.0  # unknown depth when no orderbook provided

    # Slippage: add buffer on top of VWAP (buying costs more, selling costs less)
    if side.upper() == "YES":
        raw_exec = vwap + slippage_cents
    else:
        raw_exec = vwap + slippage_cents  # same for NO (buying NO contracts)

    raw_exec = max(0.01, min(0.99, raw_exec))  # clamp to valid probability range

    # Fees: taker fee applied to notional
    fees_est = fee_rate  # expressed as fraction of notional per trade

    # Executable price = VWAP + slippage + fees
    executable_price = raw_exec + fees_est
    executable_price = max(0.01, min(0.99, executable_price))

    slippage_est = abs(vwap - fallback_price) + slippage_cents

    # Liquidity check
    effective_depth = depth if orderbook else _estimate_depth(fallback_price, target_size_usd)
    is_liquid = effective_depth >= min_depth

    return ExecutionInfo(
        executable_price=executable_price,
        vwap_price=vwap,
        depth_usd=effective_depth,
        fees_est=fees_est,
        slippage_est=slippage_est,
        is_liquid=is_liquid,
    )


def _compute_vwap(
    levels: list[OrderbookLevel],
    target_size_usd: float,
) -> tuple[float, float]:
    """
    Walk the orderbook levels and compute VWAP for *target_size_usd*.

    Returns (vwap_price, total_depth_usd_available).
    """
    total_cost = 0.0
    total_filled = 0.0
    remaining = target_size_usd

    for level in levels:
        if remaining <= 0:
            break
        fill = min(remaining, level.size_usd)
        total_cost += fill * level.price
        total_filled += fill
        remaining -= fill

    total_depth = sum(lv.size_usd for lv in levels)

    if total_filled == 0:
        # Orderbook is empty — use first level price or 0.5 fallback
        vwap = levels[0].price if levels else 0.5
    else:
        vwap = total_cost / total_filled

    return vwap, total_depth


def _estimate_depth(mid_price: float, target_size_usd: float) -> float:
    """
    When no real orderbook is available, estimate depth heuristically.
    Uses a conservative multiplier so illiquid markets fail the depth check.
    """
    # Assume 2× the target size is available around mid (very rough)
    return target_size_usd * 2.0
