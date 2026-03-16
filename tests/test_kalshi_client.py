"""
Tests for clients/kalshi_client.py

All HTTP calls are mocked — no real API calls.

The Kalshi client now uses a single endpoint per series:
  GET /markets?series_ticker=KXHIGHNY&status=open
No events, no nested-markets flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import clients.kalshi_client as kc
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

def _make_markets_response(ticker="KXHIGHNY-26MAR13-T68", status="open"):
    return {
        "markets": [{
            "ticker": ticker,
            "status": status,
            "subtitle": "above 68 degrees",
            # Legacy integer-cent fields (still supported as fallback)
            "yes_bid": 55,
            "yes_ask": 57,
            # New fixed-point dollar fields (preferred as of March 2026)
            "yes_bid_dollars": "0.55",
            "yes_ask_dollars": "0.57",
            "volume": 1000,
            "open_interest": 500,
        }],
        "cursor": None,
    }


class TestFetchAllWeatherMarkets:
    def _mock_session(self, markets_resp):
        """Mock session that returns markets_resp for any GET /markets call."""
        session = MagicMock()
        resp_mock = MagicMock()
        resp_mock.json.return_value = markets_resp
        session.get.return_value = resp_mock
        return session

    def test_returns_list(self):
        session = self._mock_session({"markets": [], "cursor": None})
        result = fetch_all_weather_markets(session=session)
        assert isinstance(result, list)

    def test_parses_above_market(self):
        session = self._mock_session(
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)

        assert len(result) == 1
        m = result[0]
        assert m["city"] == "NYC"
        assert m["target_date"] == "2026-03-13"
        assert m["market_type"] == "above"
        assert m["high_f"] == 68.0
        assert m["exchange"] == "kalshi"

    def test_market_price_from_yes_ask_dollars(self):
        """yes_ask_dollars field used as market price."""
        session = self._mock_session(
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert abs(result[0]["market_price"] - 0.57) < 1e-9

    def test_dollars_fields_take_precedence_over_cents(self):
        """yes_ask_dollars wins over yes_ask (cents) when both present."""
        session = self._mock_session({
            "markets": [{
                "ticker": "KXHIGHNY-26MAR13-T68",
                "status": "open",
                "subtitle": "above 68 degrees",
                "yes_bid": 40,            # legacy cents (0.40) — ignored
                "yes_ask": 42,            # legacy cents (0.42) — ignored
                "yes_bid_dollars": "0.62",
                "yes_ask_dollars": "0.64",
                "volume": 100,
                "open_interest": 50,
            }],
            "cursor": None,
        })
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert abs(result[0]["market_price"] - 0.64) < 1e-9

    def test_legacy_cents_used_when_no_dollars_fields(self):
        """Falls back to integer cents when _dollars fields are absent."""
        session = self._mock_session({
            "markets": [{
                "ticker": "KXHIGHNY-26MAR13-T68",
                "status": "open",
                "subtitle": "above 68 degrees",
                "yes_bid": 58,
                "yes_ask": 60,
                "volume": 100,
                "open_interest": 50,
            }],
            "cursor": None,
        })
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert abs(result[0]["market_price"] - 0.60) < 1e-9

    def test_skips_non_open_markets(self):
        session = self._mock_session(
            _make_markets_response("KXHIGHNY-26MAR13-T68", status="settled"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)
        assert len(result) == 0

    def test_skips_unparseable_ticker(self):
        session = self._mock_session({
            "markets": [{"ticker": "INVALID", "status": "open",
                         "yes_bid": 50, "yes_ask": 52,
                         "volume": 0, "open_interest": 0}],
            "cursor": None,
        })
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
            _make_markets_response("KXHIGHNY-26MAR13-T68"),
        )
        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)

        required = {"id", "ticker", "city", "target_date", "market_type",
                    "high_f", "market_price", "exchange", "volume", "open_interest"}
        assert required.issubset(result[0].keys())

    def test_pagination_follows_cursor(self):
        """Fetches next page when cursor is returned."""
        session = MagicMock()
        page1 = MagicMock()
        page1.json.return_value = {
            "markets": [{"ticker": "KXHIGHNY-26MAR13-T68", "status": "open",
                         "subtitle": "above 68 degrees", "yes_ask_dollars": "0.55",
                         "yes_bid_dollars": "0.53", "volume": 10, "open_interest": 5}],
            "cursor": "next_page_token",
        }
        page2 = MagicMock()
        page2.json.return_value = {
            "markets": [{"ticker": "KXHIGHNY-26MAR14-T70", "status": "open",
                         "subtitle": "above 70 degrees", "yes_ask_dollars": "0.45",
                         "yes_bid_dollars": "0.43", "volume": 8, "open_interest": 4}],
            "cursor": None,
        }
        session.get.side_effect = [page1, page2]

        with patch("clients.kalshi_client.WEATHER_SERIES", ["KXHIGHNY"]):
            result = fetch_all_weather_markets(session=session)

        assert len(result) == 2
        assert session.get.call_count == 2


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

    def test_orderbook_fp_preferred_over_legacy(self):
        """orderbook_fp.yes_dollars takes precedence over orderbook.yes (cents)."""
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "orderbook": {"yes": [[40, 100], [42, 200]]},   # legacy — ignored
            "orderbook_fp": {"yes_dollars": [["0.62", "100"], ["0.64", "200"]]},
        }
        session.get.return_value = resp
        levels = fetch_orderbook("TICKER", session=session)
        assert len(levels) == 2
        assert levels[0]["price"] == pytest.approx(0.62)
        assert levels[1]["price"] == pytest.approx(0.64)

    def test_orderbook_legacy_fallback_when_no_fp(self):
        """Falls back to legacy integer-cent orderbook when orderbook_fp absent."""
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "orderbook": {"yes": [[55, 100], [57, 200]]},
        }
        session.get.return_value = resp
        levels = fetch_orderbook("TICKER", session=session)
        assert levels[0]["price"] == pytest.approx(0.55)
        assert levels[1]["price"] == pytest.approx(0.57)


def test_scan_weather_series_filters_weather_titles(monkeypatch):
    class DummyResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {
                "markets": [
                    {"ticker": "KXHIGHNY-26MAR13-T68", "status": "open", "title": "NY temp above 68", "yes_ask_dollars": "0.44"},
                    {"ticker": "OTHER-1", "status": "open", "title": "stocks market", "yes_ask_dollars": "0.55"},
                ],
                "cursor": None,
            }

    class DummySession:
        def get(self, url, params=None, timeout=None):
            return DummyResp()

    rows = kc.scan_weather_series(session=DummySession(), extra_series=["KXRAINNYC"])
    assert rows
    assert rows[0]["ticker"] == "KXHIGHNY-26MAR13-T68"
    assert rows[0]["discovery_series"] in {"KXHIGHNY", "KXRAINNYC"}
