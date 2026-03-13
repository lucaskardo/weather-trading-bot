"""Tests for Phase 0.2: NWS Settlement Truth."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import clients.nws_settlement as nws
from clients.nws_settlement import (
    STATION_MAP,
    SettlementError,
    _parse_iem_response,
    fetch_settlement,
)
from state.db import init_db


# --------------------------------------------------------------------------- #
# Station map
# --------------------------------------------------------------------------- #

class TestStationMap:
    def test_all_required_cities_present(self):
        required = {"NYC", "CHI", "MIA", "LA", "DC", "SF", "HOU", "BOS", "DAL", "ATL", "SEA"}
        missing = required - set(STATION_MAP.keys())
        assert not missing, f"Missing cities: {missing}"

    def test_station_codes_are_icao(self):
        for city, station in STATION_MAP.items():
            assert len(station) == 4, f"{city}: station '{station}' not 4 chars"
            assert station.isupper(), f"{city}: station '{station}' not uppercase"

    def test_nyc_maps_to_klga(self):
        assert STATION_MAP["NYC"] == "KLGA"

    def test_chi_maps_to_kord(self):
        assert STATION_MAP["CHI"] == "KORD"


# --------------------------------------------------------------------------- #
# fetch_settlement — unknown city
# --------------------------------------------------------------------------- #

class TestFetchSettlementValidation:
    def test_unknown_city_raises_key_error(self):
        with pytest.raises(KeyError, match="UNKNOWN"):
            fetch_settlement("UNKNOWN", "2026-06-01")


# --------------------------------------------------------------------------- #
# IEM ASOS parser
# --------------------------------------------------------------------------- #

class TestParseIemResponse:
    def test_returns_max_temp(self):
        csv = (
            "station,valid,tmpf\n"
            "KLGA,2026-06-01 00:00,68.0\n"
            "KLGA,2026-06-01 12:00,85.0\n"
            "KLGA,2026-06-01 18:00,82.3\n"
        )
        result = _parse_iem_response(csv, "KLGA", "2026-06-01")
        assert result == pytest.approx(85.0)

    def test_skips_missing_values(self):
        csv = (
            "station,valid,tmpf\n"
            "KLGA,2026-06-01 00:00,M\n"
            "KLGA,2026-06-01 12:00,78.5\n"
            "KLGA,2026-06-01 18:00,T\n"
        )
        result = _parse_iem_response(csv, "KLGA", "2026-06-01")
        assert result == pytest.approx(78.5)

    def test_raises_on_all_missing(self):
        csv = (
            "station,valid,tmpf\n"
            "KLGA,2026-06-01 00:00,M\n"
        )
        with pytest.raises(SettlementError):
            _parse_iem_response(csv, "KLGA", "2026-06-01")

    def test_handles_empty_response(self):
        with pytest.raises(SettlementError):
            _parse_iem_response("", "KLGA", "2026-06-01")

    def test_single_observation(self):
        csv = "station,valid,tmpf\nKLGA,2026-06-01 14:00,91.2\n"
        result = _parse_iem_response(csv, "KLGA", "2026-06-01")
        assert result == pytest.approx(91.2)


# --------------------------------------------------------------------------- #
# fetch_settlement — mocked HTTP
# --------------------------------------------------------------------------- #

class TestFetchSettlementIemFallback:
    """Test fetch_settlement when NOAA token is absent → falls back to IEM."""

    IEM_CSV = (
        "station,valid,tmpf\n"
        "KLGA,2026-06-01 00:00,65.0\n"
        "KLGA,2026-06-01 12:00,88.0\n"
        "KLGA,2026-06-01 18:00,84.0\n"
    )

    def _mock_response(self, text: str, url: str = "http://iem.fake/asos") -> MagicMock:
        mock = MagicMock()
        mock.text = text
        mock.url = url
        mock.raise_for_status.return_value = None
        return mock

    def test_uses_iem_when_no_token(self, monkeypatch):
        monkeypatch.delenv("NOAA_CDO_TOKEN", raising=False)
        # Disable cache
        monkeypatch.setattr(nws, "_load_cache", lambda *a: None)
        monkeypatch.setattr(nws, "_save_cache", lambda *a: None)

        with patch("requests.get", return_value=self._mock_response(self.IEM_CSV)):
            result = fetch_settlement("NYC", "2026-06-01")

        assert result["actual_high_f"] == pytest.approx(88.0)
        assert result["station"] == "KLGA"
        assert result["source"] == "iem_asos"

    def test_result_contains_required_keys(self, monkeypatch):
        monkeypatch.delenv("NOAA_CDO_TOKEN", raising=False)
        monkeypatch.setattr(nws, "_load_cache", lambda *a: None)
        monkeypatch.setattr(nws, "_save_cache", lambda *a: None)

        with patch("requests.get", return_value=self._mock_response(self.IEM_CSV)):
            result = fetch_settlement("CHI", "2026-06-01")

        assert "actual_high_f" in result
        assert "station" in result
        assert "source_url" in result
        assert "source" in result

    def test_raises_settlement_error_when_both_fail(self, monkeypatch):
        monkeypatch.delenv("NOAA_CDO_TOKEN", raising=False)
        monkeypatch.setattr(nws, "_load_cache", lambda *a: None)
        monkeypatch.setattr(nws, "_save_cache", lambda *a: None)

        with patch("requests.get", side_effect=Exception("network error")):
            with pytest.raises(SettlementError):
                fetch_settlement("NYC", "2026-06-01")


class TestFetchSettlementNOAA:
    """Test fetch_settlement when NOAA token IS present."""

    NOAA_JSON = {
        "results": [{"value": 91.4, "datatype": "TMAX", "date": "2026-06-01T00:00:00"}]
    }

    def _mock_noaa_response(self) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = self.NOAA_JSON
        mock.url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?..."
        mock.raise_for_status.return_value = None
        return mock

    def test_uses_noaa_when_token_present(self, monkeypatch):
        monkeypatch.setenv("NOAA_CDO_TOKEN", "test-token-123")
        monkeypatch.setattr(nws, "_load_cache", lambda *a: None)
        monkeypatch.setattr(nws, "_save_cache", lambda *a: None)

        with patch("requests.get", return_value=self._mock_noaa_response()):
            result = fetch_settlement("NYC", "2026-06-01")

        assert result["actual_high_f"] == pytest.approx(91.4)
        assert result["source"] == "noaa_cdo"

    def test_falls_back_to_iem_when_noaa_returns_empty(self, monkeypatch):
        monkeypatch.setenv("NOAA_CDO_TOKEN", "test-token-123")
        monkeypatch.setattr(nws, "_load_cache", lambda *a: None)
        monkeypatch.setattr(nws, "_save_cache", lambda *a: None)

        iem_csv = "station,valid,tmpf\nKLGA,2026-06-01 14:00,77.5\n"

        noaa_mock = MagicMock()
        noaa_mock.json.return_value = {"results": []}  # empty
        noaa_mock.raise_for_status.return_value = None
        noaa_mock.url = "https://noaa.fake/cdo"

        iem_mock = MagicMock()
        iem_mock.text = iem_csv
        iem_mock.url = "https://iem.fake/asos"
        iem_mock.raise_for_status.return_value = None

        with patch("requests.get", side_effect=[noaa_mock, iem_mock]):
            result = fetch_settlement("NYC", "2026-06-01")

        assert result["actual_high_f"] == pytest.approx(77.5)
        assert result["source"] == "iem_asos"


# --------------------------------------------------------------------------- #
# SQLite cache
# --------------------------------------------------------------------------- #

class TestSettlementCache:
    def test_returns_cached_result(self, tmp_path, monkeypatch):
        db_path = tmp_path / "cache_test.db"
        init_db(db_path)

        # Pre-populate cache
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """INSERT INTO settlement_cache
               (city, target_date, actual_high_f, station, source_url)
               VALUES (?, ?, ?, ?, ?)""",
            ("NYC", "2026-06-01", 87.3, "KLGA", "http://cached"),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(nws, "_DB_PATH_OVERRIDE", db_path)

        # Should NOT call requests.get
        with patch("requests.get") as mock_get:
            result = fetch_settlement("NYC", "2026-06-01")
            mock_get.assert_not_called()

        assert result["actual_high_f"] == pytest.approx(87.3)
        assert result["source"] == "cache"

    def test_saves_to_cache_after_fetch(self, tmp_path, monkeypatch):
        db_path = tmp_path / "save_cache_test.db"
        init_db(db_path)
        monkeypatch.setattr(nws, "_DB_PATH_OVERRIDE", db_path)
        monkeypatch.delenv("NOAA_CDO_TOKEN", raising=False)

        iem_csv = "station,valid,tmpf\nKLGA,2026-06-01 14:00,83.0\n"
        mock_resp = MagicMock()
        mock_resp.text = iem_csv
        mock_resp.url = "http://iem.fake"
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            fetch_settlement("NYC", "2026-06-01")

        # Verify it's in the cache now
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT * FROM settlement_cache WHERE city='NYC' AND target_date='2026-06-01'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[2] == pytest.approx(83.0)  # actual_high_f column

    def test_cache_miss_hits_network(self, tmp_path, monkeypatch):
        db_path = tmp_path / "miss_test.db"
        init_db(db_path)
        monkeypatch.setattr(nws, "_DB_PATH_OVERRIDE", db_path)
        monkeypatch.delenv("NOAA_CDO_TOKEN", raising=False)

        iem_csv = "station,valid,tmpf\nKORD,2026-07-04 14:00,92.0\n"
        mock_resp = MagicMock()
        mock_resp.text = iem_csv
        mock_resp.url = "http://iem.fake"
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = fetch_settlement("CHI", "2026-07-04")
            mock_get.assert_called_once()

        assert result["actual_high_f"] == pytest.approx(92.0)
