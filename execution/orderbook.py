"""
Orderbook-aware execution pricing.

Computes the true cost to execute a trade by walking the orderbook
(if available) or falling back to mid-price + slippage + fees.

All prices are in cents-per-dollar (0.0–1.0 probability space).

Includes Kalshi's Feb 2026 price-dependent fee schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.params import PARAMS, Params


# ---------------------------------------------------------------------------
# Kalshi fee schedule (Feb 2026 official formula)
# ---------------------------------------------------------------------------

def kalshi_taker_fee(price: float) -> float:
    """
    Kalshi Feb 2026 taker fee per contract.
    Official formula: fee = 0.07 * P * (1 - P)
    where P = contract price as probability (0–1).
    Returns fee as a fraction of $1 contract value.

    At 50¢: 0.07 * 0.5 * 0.5 = 0.0175 (1.75¢)
    At 10¢: 0.07 * 0.1 * 0.9 = 0.0063 (0.63¢)
    At  5¢: 0.07 * 0.05 * 0.95 = 0.0033 (0.33¢)
    """
    p = max(0.01, min(0.99, price))
    return 0.07 * p * (1.0 - p)


def kalshi_maker_fee(price: float) -> float:
    """Maker fee is 1/4 of taker fee."""
    return kalshi_taker_fee(price) * 0.25


# Pipeline alias — backwards compatible
def kalshi_fee_rate(price: float) -> float:
    """Alias for kalshi_taker_fee (used in executable price calculation)."""
    return kalshi_taker_fee(price)


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
    slippage_cents = p.slippage_buffer_cents / 100.0  # convert cents → fraction
    min_depth = p.min_depth_usd

    if orderbook:
        vwap, depth = _compute_vwap(orderbook, target_size_usd)
    else:
        vwap = fallback_price
        depth = 0.0  # unknown depth when no orderbook provided

    # Slippage: add buffer on top of VWAP (buying costs more, selling costs less)
    raw_exec = vwap + slippage_cents
    raw_exec = max(0.01, min(0.99, raw_exec))  # clamp to valid probability range

    # Fees: use Kalshi's price-dependent schedule (peaks near 50¢ contracts)
    fees_est = kalshi_fee_rate(raw_exec)

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


def get_exit_price(current_price: float, side: str, params: Optional[Params] = None) -> float:
    """
    Net proceeds per dollar invested when exiting a position at current_price.

    Deducts slippage and taker fee from the gross exit value so that
    _process_exit PnL reflects real round-trip costs.

    For YES: we sell at (current_price - slippage - fee).
    For NO:  our contract is worth (1 - current_price); we deduct costs from that.

    Returns a value in (0, 1).
    """
    p = params or PARAMS
    slippage = p.slippage_buffer_cents / 100.0
    fee = kalshi_taker_fee(current_price)
    if side == "YES":
        return max(0.01, current_price - slippage - fee)
    else:
        return max(0.01, (1.0 - current_price) - slippage - fee)


def _estimate_depth(mid_price: float, target_size_usd: float) -> float:
    """
    When no real orderbook is available, estimate depth heuristically.
    Uses a conservative multiplier so illiquid markets fail the depth check.
    """
    # Assume 2× the target size is available around mid (very rough)
    return target_size_usd * 2.0
