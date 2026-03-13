"""Tests for Phase 0.4: Executable Edge (fees + slippage)."""

from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.params import PARAMS, Params, get_cluster
from execution.orderbook import (
    ExecutionInfo,
    OrderbookLevel,
    _compute_vwap,
    get_executable_price,
)
from core.signals import EdgeResult, calculate_edge, find_opportunities, select_side


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def default_params(**overrides) -> Params:
    p = Params()
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #

class TestParams:
    def test_default_fee(self):
        assert PARAMS.taker_fee_pct == pytest.approx(0.01)

    def test_default_slippage(self):
        assert PARAMS.slippage_buffer_cents == pytest.approx(1.0)

    def test_default_min_depth(self):
        assert PARAMS.min_depth_usd == pytest.approx(50.0)

    def test_router_weights_sum_to_one(self):
        total = (
            PARAMS.router_w_sharpe
            + PARAMS.router_w_calibration
            + PARAMS.router_w_exec
            + PARAMS.router_w_dd
            + PARAMS.router_w_instability
        )
        assert total == pytest.approx(1.0)

    def test_cluster_definitions_present(self):
        required = {"northeast", "midwest", "south", "west"}
        assert required.issubset(set(PARAMS.clusters.keys()))

    def test_nyc_in_northeast(self):
        assert "NYC" in PARAMS.clusters["northeast"]

    def test_get_cluster_known_city(self):
        assert get_cluster("NYC") == "northeast"
        assert get_cluster("CHI") == "midwest"
        assert get_cluster("MIA") == "south"
        assert get_cluster("LA") == "west"

    def test_get_cluster_unknown_city(self):
        assert get_cluster("ZZZ") is None


# --------------------------------------------------------------------------- #
# _compute_vwap
# --------------------------------------------------------------------------- #

class TestComputeVwap:
    def test_single_level_full_fill(self):
        levels = [OrderbookLevel(price=0.55, size_usd=200.0)]
        vwap, depth = _compute_vwap(levels, 100.0)
        assert vwap == pytest.approx(0.55)
        assert depth == pytest.approx(200.0)

    def test_two_levels_partial_fill(self):
        levels = [
            OrderbookLevel(price=0.54, size_usd=60.0),
            OrderbookLevel(price=0.56, size_usd=100.0),
        ]
        # Buy 100: 60 @ 0.54 + 40 @ 0.56
        vwap, depth = _compute_vwap(levels, 100.0)
        expected = (60 * 0.54 + 40 * 0.56) / 100.0
        assert vwap == pytest.approx(expected)
        assert depth == pytest.approx(160.0)

    def test_depth_exceeds_target(self):
        levels = [OrderbookLevel(price=0.60, size_usd=500.0)]
        vwap, depth = _compute_vwap(levels, 100.0)
        assert vwap == pytest.approx(0.60)
        assert depth == pytest.approx(500.0)

    def test_empty_orderbook_returns_fallback(self):
        vwap, depth = _compute_vwap([], 100.0)
        assert vwap == pytest.approx(0.5)
        assert depth == pytest.approx(0.0)

    def test_three_levels_vwap(self):
        levels = [
            OrderbookLevel(price=0.50, size_usd=50.0),
            OrderbookLevel(price=0.52, size_usd=50.0),
            OrderbookLevel(price=0.54, size_usd=50.0),
        ]
        vwap, depth = _compute_vwap(levels, 150.0)
        expected = (50 * 0.50 + 50 * 0.52 + 50 * 0.54) / 150.0
        assert vwap == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# get_executable_price
# --------------------------------------------------------------------------- #

class TestGetExecutablePrice:
    def test_executable_price_higher_than_market(self):
        """Buying always costs more than mid due to fees+slippage."""
        p = default_params(taker_fee_pct=0.01, slippage_buffer_cents=1.0)
        info = get_executable_price("TKR", "YES", 100.0, 0.55, params=p)
        assert info.executable_price > 0.55

    def test_fees_included(self):
        # Fee schedule is now price-dependent (Kalshi Feb 2026).
        # At 0.55 price: effective_cents = min(55,45) = 45, bracket <51 → rate=7%
        # fee = 45/100 * 0.07 = 0.0315
        from execution.orderbook import kalshi_fee_rate
        p = default_params(taker_fee_pct=0.01, slippage_buffer_cents=0.0)
        info = get_executable_price("TKR", "YES", 100.0, 0.55, params=p)
        expected_fee = kalshi_fee_rate(0.55)
        assert info.fees_est == pytest.approx(expected_fee)

    def test_slippage_included(self):
        p = default_params(taker_fee_pct=0.0, slippage_buffer_cents=2.0)
        info = get_executable_price("TKR", "YES", 100.0, 0.55, params=p)
        assert info.slippage_est >= 0.02

    def test_executable_price_clamped_to_valid_range(self):
        # Even at near-1.0 price, stays < 1.0
        p = default_params(taker_fee_pct=0.01, slippage_buffer_cents=1.0)
        info = get_executable_price("TKR", "YES", 100.0, 0.98, params=p)
        assert info.executable_price <= 0.99
        assert info.executable_price >= 0.01

    def test_is_liquid_true_when_depth_sufficient(self):
        p = default_params(min_depth_usd=50.0)
        levels = [OrderbookLevel(price=0.55, size_usd=200.0)]
        info = get_executable_price("TKR", "YES", 100.0, 0.55, orderbook=levels, params=p)
        assert info.is_liquid is True

    def test_is_liquid_false_when_depth_insufficient(self):
        p = default_params(min_depth_usd=50.0)
        levels = [OrderbookLevel(price=0.55, size_usd=10.0)]  # only $10
        info = get_executable_price("TKR", "YES", 100.0, 0.55, orderbook=levels, params=p)
        assert info.is_liquid is False

    def test_no_orderbook_uses_fallback_price(self):
        p = default_params(taker_fee_pct=0.01, slippage_buffer_cents=1.0)
        info = get_executable_price("TKR", "YES", 100.0, 0.60, orderbook=None, params=p)
        assert info.vwap_price == pytest.approx(0.60)

    def test_with_orderbook_uses_vwap(self):
        p = default_params(taker_fee_pct=0.0, slippage_buffer_cents=0.0)
        levels = [
            OrderbookLevel(price=0.54, size_usd=60.0),
            OrderbookLevel(price=0.58, size_usd=100.0),
        ]
        info = get_executable_price("TKR", "YES", 100.0, 0.56, orderbook=levels, params=p)
        expected_vwap = (60 * 0.54 + 40 * 0.58) / 100.0
        assert info.vwap_price == pytest.approx(expected_vwap)

    def test_returns_execution_info_type(self):
        info = get_executable_price("TKR", "YES", 100.0, 0.55)
        assert isinstance(info, ExecutionInfo)

    def test_all_required_fields_present(self):
        info = get_executable_price("TKR", "YES", 100.0, 0.55)
        assert hasattr(info, "executable_price")
        assert hasattr(info, "vwap_price")
        assert hasattr(info, "depth_usd")
        assert hasattr(info, "fees_est")
        assert hasattr(info, "slippage_est")
        assert hasattr(info, "is_liquid")


# --------------------------------------------------------------------------- #
# calculate_edge (core/signals.py)
# --------------------------------------------------------------------------- #

class TestCalculateEdge:
    def test_executable_edge_less_than_raw_edge(self):
        """Executable edge must be less than raw edge due to costs."""
        result = calculate_edge("TKR", "YES", model_prob=0.70, market_price=0.55)
        assert result.raw_edge > result.executable_edge

    def test_raw_edge_formula(self):
        result = calculate_edge("TKR", "YES", model_prob=0.70, market_price=0.55)
        assert result.raw_edge == pytest.approx(0.70 - 0.55)

    def test_executable_edge_accounts_for_costs(self):
        p = default_params(taker_fee_pct=0.01, slippage_buffer_cents=1.0)
        result = calculate_edge("TKR", "YES", model_prob=0.70, market_price=0.55, params=p)
        # executable_price = 0.55 + 0.01 (slippage) + 0.01 (fee) = 0.57
        assert result.executable_edge < result.raw_edge
        assert result.executable_edge == pytest.approx(0.70 - result.exec_info.executable_price)

    def test_passes_filter_when_edge_sufficient(self):
        p = default_params(min_executable_edge=0.05, min_depth_usd=0.0)
        result = calculate_edge("TKR", "YES", model_prob=0.80, market_price=0.55, params=p)
        assert result.passes_filter is True

    def test_fails_filter_when_edge_too_small(self):
        p = default_params(
            min_executable_edge=0.10,
            taker_fee_pct=0.01,
            slippage_buffer_cents=1.0,
            min_depth_usd=0.0,
        )
        # raw edge = 0.06, after costs < 0.10 threshold
        result = calculate_edge("TKR", "YES", model_prob=0.61, market_price=0.55, params=p)
        assert result.passes_filter is False

    def test_fails_filter_when_illiquid(self):
        p = default_params(min_executable_edge=0.0, min_depth_usd=500.0)
        # No orderbook → estimated depth = 2 * target_size = 200 < 500
        result = calculate_edge(
            "TKR", "YES", model_prob=0.80, market_price=0.55,
            target_size_usd=100.0, params=p
        )
        assert result.passes_filter is False

    def test_returns_edge_result_type(self):
        result = calculate_edge("TKR", "YES", 0.65, 0.50)
        assert isinstance(result, EdgeResult)

    def test_exec_info_attached(self):
        result = calculate_edge("TKR", "YES", 0.65, 0.50)
        assert isinstance(result.exec_info, ExecutionInfo)

    def test_with_orderbook(self):
        levels = [OrderbookLevel(price=0.55, size_usd=200.0)]
        result = calculate_edge(
            "TKR", "YES", model_prob=0.72, market_price=0.55,
            orderbook=levels
        )
        assert result.exec_info.vwap_price == pytest.approx(0.55)
        assert result.exec_info.depth_usd == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# select_side
# --------------------------------------------------------------------------- #

class TestSelectSide:
    def test_yes_when_model_above_market(self):
        assert select_side(0.70, 0.55) == "YES"

    def test_no_when_model_below_market(self):
        assert select_side(0.40, 0.55) == "NO"

    def test_no_at_exact_boundary(self):
        # model == market → no edge; falls through to NO (no reason to prefer YES)
        assert select_side(0.55, 0.55) == "NO"


# --------------------------------------------------------------------------- #
# find_opportunities
# --------------------------------------------------------------------------- #

class TestFindOpportunities:
    def _markets(self):
        return [
            {"ticker": "TKR_A", "market_price": 0.50},  # big edge
            {"ticker": "TKR_B", "market_price": 0.60},  # small edge
            {"ticker": "TKR_C", "market_price": 0.45},  # no model prob
        ]

    def _probs(self):
        return {
            "TKR_A": 0.75,
            "TKR_B": 0.63,
        }

    def test_returns_only_markets_with_model_prob(self):
        results = find_opportunities(self._markets(), self._probs())
        tickers = [r.ticker for r in results]
        assert "TKR_C" not in tickers

    def test_sorted_by_executable_edge_descending(self):
        results = find_opportunities(self._markets(), self._probs())
        edges = [r.executable_edge for r in results]
        assert edges == sorted(edges, reverse=True)

    def test_returns_edge_result_objects(self):
        results = find_opportunities(self._markets(), self._probs())
        assert all(isinstance(r, EdgeResult) for r in results)

    def test_empty_markets(self):
        assert find_opportunities([], {"TKR_A": 0.7}) == []

    def test_empty_model_probs(self):
        assert find_opportunities(self._markets(), {}) == []

    def test_respects_custom_params(self):
        # Set very high min_executable_edge → nothing passes filter
        p = default_params(min_executable_edge=0.99, min_depth_usd=0.0)
        results = find_opportunities(self._markets(), self._probs(), params=p)
        assert all(not r.passes_filter for r in results)

    def test_with_orderbook_data(self):
        markets = [
            {
                "ticker": "TKR_A",
                "market_price": 0.50,
                "orderbook": [{"price": 0.51, "size_usd": 300.0}],
            }
        ]
        results = find_opportunities(markets, {"TKR_A": 0.75})
        assert len(results) == 1
        assert results[0].exec_info.vwap_price == pytest.approx(0.51)
