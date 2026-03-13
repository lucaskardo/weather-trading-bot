"""Tests for Phase 0.3: Forecast Lineage."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.types import ModelForecast, ConsensusForecast
from clients.weather import (
    CITY_COORDS,
    WeatherFetchError,
    _parse_open_meteo,
    fetch_model,
    fetch_all_models,
    fetch_and_store,
)
import clients.weather as weather_module
from state.db import init_db


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _om_response(target_date: str, high: float = 85.0, low: float = 68.0) -> dict:
    """Minimal Open-Meteo daily response."""
    return {
        "daily": {
            "time": [target_date],
            "temperature_2m_max": [high],
            "temperature_2m_min": [low],
        }
    }


def _mock_get(target_date: str, high: float = 85.0, low: float = 68.0) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = _om_response(target_date, high, low)
    mock.url = "https://api.open-meteo.com/v1/forecast?..."
    mock.raise_for_status.return_value = None
    return mock


# --------------------------------------------------------------------------- #
# ModelForecast dataclass
# --------------------------------------------------------------------------- #

class TestModelForecastDataclass:
    def test_has_lineage_fields(self):
        f = ModelForecast(
            model_name="GFS",
            city="NYC",
            target_date="2026-06-01",
            predicted_high_f=82.0,
            run_id="2026060100",
            publish_time="2026-06-01T00:00:00+00:00",
            source_url="https://api.open-meteo.com/v1/forecast?models=gfs_seamless",
        )
        assert f.run_id == "2026060100"
        assert f.publish_time == "2026-06-01T00:00:00+00:00"
        assert f.source_url.startswith("https://")

    def test_lineage_fields_default_to_empty_string(self):
        f = ModelForecast(
            model_name="ECMWF",
            city="CHI",
            target_date="2026-06-01",
            predicted_high_f=75.0,
        )
        # Defaults exist — callers must fill them before storing
        assert f.run_id == ""
        assert f.publish_time == ""
        assert f.source_url == ""

    def test_optional_fields(self):
        f = ModelForecast(
            model_name="ICON",
            city="NYC",
            target_date="2026-06-01",
            predicted_high_f=80.0,
        )
        assert f.predicted_low_f is None
        assert f.confidence is None
        assert f.market_id is None

    def test_all_fields_assignable(self):
        f = ModelForecast(
            model_name="GFS",
            city="NYC",
            target_date="2026-06-01",
            predicted_high_f=82.0,
            predicted_low_f=65.0,
            confidence=0.85,
            run_id="2026060106",
            publish_time="2026-06-01T06:00:00+00:00",
            source_url="https://example.com",
            fetched_at="2026-06-01T07:00:00+00:00",
            market_id="mkt-nyc-001",
        )
        assert f.confidence == pytest.approx(0.85)
        assert f.market_id == "mkt-nyc-001"


class TestConsensusForecastDataclass:
    def test_instantiation(self):
        c = ConsensusForecast(
            city="NYC",
            target_date="2026-06-01",
            consensus_high_f=83.5,
            consensus_low_f=66.0,
            agreement=2.1,
            n_models=4,
            model_names=["GFS", "ECMWF", "ICON", "NOAA_NWS"],
            model_highs_f=[82.0, 84.0, 83.5, 84.5],
        )
        assert c.n_models == 4
        assert c.consensus_high_f == pytest.approx(83.5)


# --------------------------------------------------------------------------- #
# City coords
# --------------------------------------------------------------------------- #

class TestCityCoords:
    def test_all_station_map_cities_have_coords(self):
        from clients.nws_settlement import STATION_MAP
        missing = set(STATION_MAP.keys()) - set(CITY_COORDS.keys())
        assert not missing, f"Cities in STATION_MAP but missing coords: {missing}"

    def test_coords_are_valid_floats(self):
        for city, (lat, lon) in CITY_COORDS.items():
            assert -90 <= lat <= 90, f"{city} lat out of range"
            assert -180 <= lon <= 180, f"{city} lon out of range"


# --------------------------------------------------------------------------- #
# _parse_open_meteo
# --------------------------------------------------------------------------- #

class TestParseOpenMeteo:
    def test_parses_high_low(self):
        data = _om_response("2026-06-01", high=88.0, low=71.0)
        f = _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake")
        assert f.predicted_high_f == pytest.approx(88.0)
        assert f.predicted_low_f == pytest.approx(71.0)

    def test_lineage_fields_populated(self):
        data = _om_response("2026-06-01")
        f = _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake-url")
        assert f.run_id != "", "run_id must be non-empty"
        assert f.publish_time != "", "publish_time must be non-empty"
        assert f.source_url == "http://fake-url"

    def test_run_id_format(self):
        """run_id should be YYYYMMDDCC (10 chars, numeric)."""
        data = _om_response("2026-06-01")
        f = _parse_open_meteo(data, "NYC", "2026-06-01", "ECMWF", "http://fake")
        assert len(f.run_id) == 10
        assert f.run_id.isdigit()

    def test_run_id_cycle_hour_is_multiple_of_6(self):
        data = _om_response("2026-06-01")
        f = _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake")
        cycle_hour = int(f.run_id[8:10])
        assert cycle_hour in (0, 6, 12, 18)

    def test_raises_when_date_not_in_response(self):
        data = _om_response("2026-06-02")  # different date
        with pytest.raises(WeatherFetchError, match="not in Open-Meteo"):
            _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake")

    def test_raises_on_null_high(self):
        data = {
            "daily": {
                "time": ["2026-06-01"],
                "temperature_2m_max": [None],
                "temperature_2m_min": [65.0],
            }
        }
        with pytest.raises(WeatherFetchError, match="null high temp"):
            _parse_open_meteo(data, "NYC", "2026-06-01", "GFS", "http://fake")

    def test_raises_on_missing_daily_key(self):
        with pytest.raises(WeatherFetchError):
            _parse_open_meteo({}, "NYC", "2026-06-01", "GFS", "http://fake")

    def test_model_name_preserved(self):
        data = _om_response("2026-06-01")
        for model in ["GFS", "ECMWF", "ICON"]:
            f = _parse_open_meteo(data, "NYC", "2026-06-01", model, "http://x")
            assert f.model_name == model

    def test_city_preserved(self):
        data = _om_response("2026-06-01")
        f = _parse_open_meteo(data, "CHI", "2026-06-01", "GFS", "http://x")
        assert f.city == "CHI"


# --------------------------------------------------------------------------- #
# fetch_model (mocked HTTP)
# --------------------------------------------------------------------------- #

class TestFetchModel:
    def test_returns_model_forecast(self):
        with patch("requests.get", return_value=_mock_get("2026-06-01", 90.0, 72.0)):
            f = fetch_model("NYC", "2026-06-01", "GFS")
        assert isinstance(f, ModelForecast)
        assert f.predicted_high_f == pytest.approx(90.0)

    def test_lineage_complete(self):
        with patch("requests.get", return_value=_mock_get("2026-06-01")):
            f = fetch_model("NYC", "2026-06-01", "ECMWF")
        assert f.run_id != ""
        assert f.publish_time != ""
        assert f.source_url != ""
        assert f.fetched_at != ""

    def test_raises_on_unknown_city(self):
        with pytest.raises(KeyError):
            fetch_model("ZZZ", "2026-06-01", "GFS")

    def test_raises_on_unknown_model(self):
        with pytest.raises(KeyError):
            fetch_model("NYC", "2026-06-01", "DEEPMIND")

    def test_raises_weather_fetch_error_on_http_failure(self):
        with patch("requests.get", side_effect=Exception("timeout")):
            with pytest.raises(WeatherFetchError):
                fetch_model("NYC", "2026-06-01", "GFS")


# --------------------------------------------------------------------------- #
# fetch_all_models (mocked HTTP)
# --------------------------------------------------------------------------- #

class TestFetchAllModels:
    def test_returns_multiple_forecasts(self):
        with patch("requests.get", return_value=_mock_get("2026-06-01")):
            forecasts = fetch_all_models("NYC", "2026-06-01")
        assert len(forecasts) >= 1
        assert all(isinstance(f, ModelForecast) for f in forecasts)

    def test_all_forecasts_have_lineage(self):
        with patch("requests.get", return_value=_mock_get("2026-06-01")):
            forecasts = fetch_all_models("NYC", "2026-06-01")
        for f in forecasts:
            assert f.run_id != "", f"{f.model_name} missing run_id"
            assert f.source_url != "", f"{f.model_name} missing source_url"

    def test_individual_model_failure_does_not_abort(self):
        """One model erroring should not prevent others from returning."""
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("first model failed")
            return _mock_get("2026-06-01")

        with patch("requests.get", side_effect=side_effect):
            forecasts = fetch_all_models("NYC", "2026-06-01")
        assert len(forecasts) >= 1  # remaining models still returned

    def test_raises_on_unknown_city(self):
        with pytest.raises(KeyError):
            fetch_all_models("UNKNOWN", "2026-06-01")


# --------------------------------------------------------------------------- #
# fetch_and_store (SQLite persistence)
# --------------------------------------------------------------------------- #

class TestFetchAndStore:
    def test_forecasts_stored_in_db(self, tmp_path):
        conn = init_db(tmp_path / "lineage_test.db")

        with patch("requests.get", return_value=_mock_get("2026-06-01", 85.0, 67.0)):
            forecasts = fetch_and_store("NYC", "2026-06-01", conn=conn)

        rows = conn.execute("SELECT * FROM forecasts").fetchall()
        assert len(rows) == len(forecasts)

    def test_stored_lineage_fields_non_empty(self, tmp_path):
        conn = init_db(tmp_path / "lineage_fields.db")

        with patch("requests.get", return_value=_mock_get("2026-06-01")):
            fetch_and_store("CHI", "2026-06-01", conn=conn)

        rows = conn.execute("SELECT * FROM forecasts").fetchall()
        assert len(rows) > 0
        for row in rows:
            assert row["run_id"] != "", "run_id must be stored"
            assert row["publish_time"] != "", "publish_time must be stored"
            assert row["source_url"] != "", "source_url must be stored"

    def test_stored_temperatures_correct(self, tmp_path):
        conn = init_db(tmp_path / "temp_correct.db")

        with patch("requests.get", return_value=_mock_get("2026-06-01", 91.5, 74.2)):
            fetch_and_store("ATL", "2026-06-01", conn=conn)

        row = conn.execute("SELECT * FROM forecasts LIMIT 1").fetchone()
        assert row["predicted_high_f"] == pytest.approx(91.5)
        assert row["predicted_low_f"] == pytest.approx(74.2)

    def test_multiple_models_stored(self, tmp_path):
        conn = init_db(tmp_path / "multi_model.db")

        with patch("requests.get", return_value=_mock_get("2026-06-01")):
            forecasts = fetch_and_store("NYC", "2026-06-01", conn=conn)

        count = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        assert count == len(forecasts)
        assert count >= 2  # at least GFS + one other
