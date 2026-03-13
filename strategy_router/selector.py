"""
Phase 2.3 — Signal Selector.

Filters and sizes signals before execution:
  - Shadow signals → logged with is_shadow=True, never executed
  - Cluster exposure cap enforcement
  - Per-city position limit enforcement
  - Sort by executable_edge × confidence
  - Kelly-based position sizing within allocated capital
"""

from __future__ import annotations

import math
from typing import Any

from shared.params import Params, PARAMS, get_cluster
from strategies.base import Signal


def select_signals(
    signals: list[Signal],
    allocations: dict[str, float],      # strategy_name → USD available
    open_positions: list[dict[str, Any]],
    bankroll: float,
    params: Params = PARAMS,
) -> list[dict[str, Any]]:
    """
    Filter and size signals for execution.

    Returns a list of execution orders:
      {signal, size_usd, reason_skipped (if filtered)}
    Only orders without reason_skipped should be sent to exchange.
    """
    # Pre-compute cluster exposure from open positions
    cluster_exposure = _cluster_exposure(open_positions, bankroll)

    # Pre-compute per-city open position counts
    city_counts: dict[str, int] = {}
    for pos in open_positions:
        city = pos.get("city", "")
        city_counts[city] = city_counts.get(city, 0) + 1

    # Sort all signals by executable_edge × confidence (best first)
    ranked = sorted(signals, key=lambda s: s.executable_edge * s.confidence, reverse=True)

    orders: list[dict[str, Any]] = []
    # Track cluster exposure added this cycle (to prevent intra-cycle breaches)
    cycle_cluster_add: dict[str, float] = {}

    for sig in ranked:
        order: dict[str, Any] = {"signal": sig}

        # --- Gate 1: Shadow — log only ---
        if sig.is_shadow or not _strategy_is_live(sig.strategy_name):
            order["size_usd"] = 0.0
            order["reason_skipped"] = "shadow"
            orders.append(order)
            continue

        # --- Gate 2: Per-city limit ---
        if city_counts.get(sig.city, 0) >= params.max_positions_per_city:
            order["size_usd"] = 0.0
            order["reason_skipped"] = "city_limit"
            orders.append(order)
            continue

        # --- Gate 3: Cluster exposure cap ---
        cluster = get_cluster(sig.city, params)
        current_exposure = cluster_exposure.get(cluster, 0.0)
        cycle_add = cycle_cluster_add.get(cluster, 0.0)
        if cluster and (current_exposure + cycle_add) / max(bankroll, 1.0) >= params.max_cluster_exposure_pct:
            order["size_usd"] = 0.0
            order["reason_skipped"] = "cluster_cap"
            orders.append(order)
            continue

        # --- Gate 4: Allocated capital available ---
        strategy_budget = allocations.get(sig.strategy_name, 0.0)
        if strategy_budget <= 0:
            order["size_usd"] = 0.0
            order["reason_skipped"] = "no_budget"
            orders.append(order)
            continue

        # --- Size: Kelly within budget ---
        size_usd = _kelly_size(sig, strategy_budget, params)
        if size_usd <= 0:
            order["size_usd"] = 0.0
            order["reason_skipped"] = "kelly_too_small"
            orders.append(order)
            continue

        order["size_usd"] = size_usd
        order["reason_skipped"] = None

        # Update tracking
        city_counts[sig.city] = city_counts.get(sig.city, 0) + 1
        if cluster:
            cycle_cluster_add[cluster] = cycle_cluster_add.get(cluster, 0.0) + size_usd

        orders.append(order)

    return orders


def _kelly_size(sig: Signal, budget: float, params: Params) -> float:
    """
    Fractional Kelly position size.

    For a binary bet with edge e and win probability p:
      kelly_f = (p * (1/cost - 1) - (1-p)) / (1/cost - 1)
      simplified for small positions: kelly_f ≈ edge / (1 - price)

    Clamp to [min_kelly_fraction, max_kelly_fraction] of budget.
    """
    edge = sig.executable_edge
    price = sig.executable_price
    if price <= 0 or price >= 1 or edge <= 0:
        return 0.0

    odds = (1.0 - price) / price  # net odds per dollar risked
    p = sig.fair_value
    q = 1.0 - p
    kelly_f = (p * odds - q) / odds if odds > 0 else 0.0

    # Fractional Kelly (half Kelly is common)
    kelly_f = kelly_f * 0.5

    kelly_f = max(params.min_kelly_fraction, min(params.max_kelly_fraction, kelly_f))
    return kelly_f * budget


def _cluster_exposure(
    open_positions: list[dict[str, Any]],
    bankroll: float,
) -> dict[str, float]:
    """Compute current USD exposure per cluster from open positions."""
    exposure: dict[str, float] = {}
    for pos in open_positions:
        cluster = get_cluster(pos.get("city", ""))
        if cluster:
            exposure[cluster] = exposure.get(cluster, 0.0) + pos.get("size_usd", 0.0)
    return exposure


def _strategy_is_live(strategy_name: str) -> bool:
    """Look up whether a strategy is live by importing its class."""
    _LIVE_STRATEGIES = {"value_entry"}   # maintained manually until auto-promotion
    return strategy_name in _LIVE_STRATEGIES
