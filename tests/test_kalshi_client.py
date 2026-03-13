"""
Tests for clients/kalshi_client.py

All HTTP calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clients.kalshi_client import (
    SERIES_CITY,
    WEATHER_SERIES,
    _parse_kalshi_date,
    _parse_ticker,
    fetch_all_weather_markets,
    fetch_orderbook,
)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestParseKalshiDate:
    def test_standard(self):
        assert _parse_kalshi_date("26MAR13") == "2026-03-13"

    def test_jan(self):
        assert _parse_kalshi_date("26JAN01") == "2026-01-01"

    def test_dec(self):
        assert _parse_kalshi_date("25DEC31") == "2025-12-31"

    def test_invalid_month(self):
        assert _parse_kalshi_date("26XYZ13") is None

    def test_bad_format(self):
        assert _parse_kalshi_date("bad") is None

    def test_zero_pads_day(self):
        result = _parse_kalshi_date("26MAR05")
        assert result == "2026-03-05"


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

class TestParseTicker:
    def test_above_market(self):
        date, mtype, high_f, low_f = _parse_ticker("KXHIGHNY-26MAR13-T68", "")
        assert date == "2026-03-13"
        assert mtype == "above"
        assert high_f == 68.0
        assert low_f is None

    def test_band_market(self):
        date, mtype, high_f, low_f = _parse_ticker("KXHIGHNY-26MAR13-B65T66", "")
        assert date == "2026-03-13"
        assert mtype == "band"
        assert high_f == 66.0
        assert low_f == 65.0

    def test_below_from_subtitle(self):
        date, mtype, high_f, low_f = _parse_ticker(
            "KXHIGHNY-26MAR13-T68", "below 68 degrees"
        )
        assert mtype == "below"

    def test_decimal_threshold(self):
        date, mtype, high_f, low_f = _parse_ticker("KXHIGHNY-26MAR13-T68.5", "")
        assert high_f == 68.5

    def test_invalid_ticker_returns_none_date(self):
        date, mtype, high_f, low_f = _parse_ticker("INVALID", "")
        assert date is None

    def test_above_default_when_no_subtitle_hint(self):
        date, mtype, high_f, low_f = _parse_ticker("KXHIGHCHI-26JUL04-T90", "")
        assert mtype == "above"


# ---------------------------------------------------------------------------
# fetch_all_weather_markets
# ---------------------------------------------------------------------------

def _make_event_response(event_ticker="KXHIGHNY-26MAR13"):
    return {"events": [{"event_ticker": event_ticker}]}


def _make_markets_response(ticker="KXHIGHNY-26MAR13-T68", status="open"):
    return {
        "markets": [{
            "ticker": ticker,
            "status": status,
            "subtitle": "above 68 degrees",
            "yes_bid": 55,
            "yes_ask": 57,
            "volume": 1000,
            "open_interest": 500,
        }]
    }


class TestFetchAllWeatherMarkets:
    def _mock_session(self, event_resp, markets_resp):
        session = MagicMock()
        events_r = MagicMock()
        events_r.json.return_value = event_resp
        markets_r = MagicMock()
        markets_r.json.return_value = markets_resp

        def get_side_effect(url, **kwargs):
            if "/events" in url and "/markets" not in url:
                return events_r
            return markets_r

        session.get.side_effect = get_side_effect
        return session

    def test_returns_list(self):
        session = self._mock_session(
            {"events": []}, {"markets": []}
        )
        result = fetch_all_weather_markets(session=session)
        assert isinstance(result, list)

    def test_parses_above_market(self):
        session = self._mock_session(
            _make_event_response("KXHIGHNY-26MAR13"),
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        # Only mock for NYC series
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)

        assert len(result) == 1
        m = result[0]
        assert m["city"] == "NYC"
        assert m["target_date"] == "2026-03-13"
        assert m["market_type"] == "above"
        assert m["high_f"] == 68.0
        assert m["exchange"] == "kalshi"

    def test_market_price_from_yes_ask(self):
        session = self._mock_session(
            _make_event_response("KXHIGHNY-26MAR13"),
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert abs(result[0]["market_price"] - 0.57) < 1e-9

    def test_skips_non_open_markets(self):
        session = self._mock_session(
            _make_event_response("KXHIGHNY-26MAR13"),
            _make_markets_response("KXHIGHNY-26MAR13-T68", status="settled"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert len(result) == 0

    def test_skips_unparseable_ticker(self):
        session = self._mock_session(
            _make_event_response("KXHIGHNY-26MAR13"),
            {"markets": [{"ticker": "INVALID", "status": "open",
                           "yes_bid": 50, "yes_ask": 52,
                           "volume": 0, "open_interest": 0}]},
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert len(result) == 0

    def test_api_error_returns_empty(self):
        session = MagicMock()
        session.get.side_effect = Exception("network error")
        result = fetch_all_weather_markets(session=session)
        assert result == []

    def test_required_fields_present(self):
        session = self._mock_session(
            _make_event_response("KXHIGHNY-26MAR13"),
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)

        required = {"id", "ticker", "city", "target_date", "market_type",
                    "high_f", "market_price", "exchange", "volume", "open_interest"}
        assert required.issubset(result[0].keys())


# ---------------------------------------------------------------------------
# fetch_orderbook
# ---------------------------------------------------------------------------

class TestFetchOrderbook:
    def test_returns_list_of_levels(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "orderbook": {"yes": [[55, 100], [57, 200]]}
        }
        session.get.return_value = resp

        levels = fetch_orderbook("KXHIGHNY-26MAR13-T68", session=session)
        assert len(levels) == 2
        assert levels[0]["price"] == pytest.approx(0.55)
        assert levels[0]["size_usd"] == pytest.approx(0.55 * 100)

    def test_sorted_by_price(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "orderbook": {"yes": [[70, 50], [55, 100], [60, 75]]}
        }
        session.get.return_value = resp

        levels = fetch_orderbook("TICKER", session=session)
        prices = [l["price"] for l in levels]
        assert prices == sorted(prices)

    def test_api_error_returns_empty(self):
        session = MagicMock()
        session.get.side_effect = Exception("timeout")
        levels = fetch_orderbook("TICKER", session=session)
        assert levels == []

    def test_empty_orderbook(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"orderbook": {"yes": []}}
        session.get.return_value = resp
        levels = fetch_orderbook("TICKER", session=session)
        assert levels == []
