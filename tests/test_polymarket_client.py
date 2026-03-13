"""
Tests for clients/polymarket_client.py

All HTTP calls are mocked — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clients.polymarket_client import (
    _extract_city,
    _extract_date,
    _extract_threshold,
    fetch_all_weather_markets,
)


# ---------------------------------------------------------------------------
# City extraction
# ---------------------------------------------------------------------------

class TestExtractCity:
    def test_new_york(self):
        assert _extract_city("Will New York exceed 70°F?") == "NYC"

    def test_chicago(self):
        assert _extract_city("Chicago high above 85 degrees") == "CHI"

    def test_los_angeles(self):
        assert _extract_city("Will Los Angeles reach 90°F?") == "LA"

    def test_washington(self):
        assert _extract_city("Washington DC high temperature") == "DC"

    def test_san_francisco(self):
        assert _extract_city("San Francisco fog and 65°F") == "SF"

    def test_houston(self):
        assert _extract_city("Houston temperature above 95") == "HOU"

    def test_miami(self):
        assert _extract_city("Miami beach high above 88°F") == "MIA"

    def test_boston(self):
        assert _extract_city("Boston daily high below 40°F") == "BOS"

    def test_case_insensitive(self):
        assert _extract_city("CHICAGO HIGH TEMP") == "CHI"

    def test_unknown_city(self):
        assert _extract_city("Denver high temperature above 60") is None


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

class TestExtractDate:
    def test_iso_format(self):
        assert _extract_date("High on 2026-03-13") == "2026-03-13"

    def test_natural_language(self):
        assert _extract_date("Will exceed 70°F on March 13, 2026?") == "2026-03-13"

    def test_natural_no_comma(self):
        assert _extract_date("March 13 2026 high temperature") == "2026-03-13"

    def test_no_date(self):
        assert _extract_date("No date here") is None

    def test_january(self):
        assert _extract_date("January 1, 2027") == "2027-01-01"


# ---------------------------------------------------------------------------
# Threshold extraction
# ---------------------------------------------------------------------------

class TestExtractThreshold:
    def test_above(self):
        mtype, high_f, low_f = _extract_threshold("Will high exceed 72°F?")
        assert mtype == "above"
        assert high_f == 72.0
        assert low_f is None

    def test_above_keyword_variants(self):
        for text in ["above 68", "over 68 degrees", "higher than 68"]:
            mtype, high_f, _ = _extract_threshold(text)
            assert mtype == "above", f"failed for: {text}"
            assert high_f == 68.0

    def test_below(self):
        mtype, high_f, low_f = _extract_threshold("Will high be below 40°F?")
        assert mtype == "below"
        assert high_f == 40.0

    def test_band(self):
        mtype, high_f, low_f = _extract_threshold("between 65 and 70 degrees")
        assert mtype == "band"
        assert low_f == 65.0
        assert high_f == 70.0

    def test_no_threshold(self):
        mtype, high_f, low_f = _extract_threshold("no temperature here")
        assert high_f is None
        assert low_f is None


# ---------------------------------------------------------------------------
# fetch_all_weather_markets
# ---------------------------------------------------------------------------

def _make_gamma_response(city="New York", threshold=70, date="2026-03-13"):
    return [
        {
            "title": f"Will {city} high exceed {threshold}°F on {date}?",
            "markets": [
                {
                    "question": f"Will {city} high exceed {threshold}°F on {date}?",
                    "conditionId": "0xabc123",
                    "outcomePrices": ["0.62", "0.38"],
                    "volume": "5000.00",
                    "liquidity": "1000.00",
                }
            ]
        }
    ]


class TestFetchAllWeatherMarkets:
    def _mock_session(self, response_data):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = response_data
        session.get.return_value = resp
        return session

    def test_returns_list(self):
        session = self._mock_session([])
        result = fetch_all_weather_markets(session=session)
        assert isinstance(result, list)

    def test_parses_above_market(self):
        session = self._mock_session(_make_gamma_response("New York", 70, "2026-03-13"))
        result = fetch_all_weather_markets(session=session)
        assert len(result) == 1
        m = result[0]
        assert m["city"] == "NYC"
        assert m["target_date"] == "2026-03-13"
        assert m["market_type"] == "above"
        assert m["high_f"] == 70.0
        assert m["exchange"] == "polymarket"

    def test_market_price_from_outcome_prices(self):
        session = self._mock_session(_make_gamma_response())
        result = fetch_all_weather_markets(session=session)
        assert abs(result[0]["market_price"] - 0.62) < 1e-9

    def test_skips_unknown_city(self):
        data = [
            {
                "title": "Denver high above 80°F on 2026-03-13?",
                "markets": [{
                    "question": "Denver high above 80°F on 2026-03-13?",
                    "conditionId": "0xddd",
                    "outcomePrices": ["0.5", "0.5"],
                    "volume": "0",
                    "liquidity": "0",
                }]
            }
        ]
        session = self._mock_session(data)
        result = fetch_all_weather_markets(session=session)
        assert len(result) == 0

    def test_skips_market_with_no_threshold(self):
        data = [
            {
                "title": "Will New York have nice weather on 2026-03-13?",
                "markets": [{
                    "question": "Will New York have nice weather on 2026-03-13?",
                    "conditionId": "0xeee",
                    "outcomePrices": ["0.5", "0.5"],
                    "volume": "0",
                    "liquidity": "0",
                }]
            }
        ]
        session = self._mock_session(data)
        result = fetch_all_weather_markets(session=session)
        assert len(result) == 0

    def test_api_error_returns_empty(self):
        session = MagicMock()
        session.get.side_effect = Exception("timeout")
        result = fetch_all_weather_markets(session=session)
        assert result == []

    def test_required_fields_present(self):
        session = self._mock_session(_make_gamma_response())
        result = fetch_all_weather_markets(session=session)
        assert len(result) == 1
        required = {"id", "ticker", "city", "target_date", "market_type",
                    "high_f", "market_price", "exchange", "volume", "open_interest"}
        assert required.issubset(result[0].keys())

    def test_market_price_clamped(self):
        data = _make_gamma_response()
        data[0]["markets"][0]["outcomePrices"] = ["1.5", "-0.5"]  # invalid
        session = self._mock_session(data)
        result = fetch_all_weather_markets(session=session)
        assert 0.01 <= result[0]["market_price"] <= 0.99

    def test_multiple_markets_in_event(self):
        data = [
            {
                "title": "New York temperature markets March 13 2026",
                "markets": [
                    {
                        "question": "Will New York high exceed 68°F on March 13 2026?",
                        "conditionId": "0x001",
                        "outcomePrices": ["0.70", "0.30"],
                        "volume": "1000",
                        "liquidity": "500",
                    },
                    {
                        "question": "Will New York high exceed 72°F on March 13 2026?",
                        "conditionId": "0x002",
                        "outcomePrices": ["0.40", "0.60"],
                        "volume": "800",
                        "liquidity": "300",
                    },
                ]
            }
        ]
        session = self._mock_session(data)
        result = fetch_all_weather_markets(session=session)
        assert len(result) == 2
