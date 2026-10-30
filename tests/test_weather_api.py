import pandas as pd
import pytest
import requests

from src.api.weather_api import (
    WeatherAPI
)


class FakeResponse:

    def __init__(
        self,
        response_data
    ):

        self.response_data = (
            response_data
        )

    def raise_for_status(
        self
    ):

        return None

    def json(
        self
    ):

        return (
            self.response_data
        )


def create_weather_response(
    times
):

    length = len(
        times
    )

    return {
        "hourly": {
            "time":
                times,

            "temperature_2m":
                [
                    10.0
                ] * length,

            "wind_speed_10m":
                [
                    5.0
                ] * length,

            "cloud_cover":
                [
                    50.0
                ] * length,

            "precipitation":
                [
                    0.0
                ] * length
        }
    }


def test_locations_are_mapped_to_expected_price_areas():

    api = WeatherAPI()

    assert api.locations[
        "DK1"
    ][
        "name"
    ] == "Aarhus"

    assert api.locations[
        "DK2"
    ][
        "name"
    ] == "Copenhagen"


def test_unknown_price_area_is_rejected():

    api = WeatherAPI()

    with pytest.raises(
        ValueError
    ):

        api._validate_price_area(
            "DK3"
        )


def test_relative_dates_use_copenhagen_clock(
    monkeypatch
):

    api = WeatherAPI()

    fixed_now = pd.Timestamp(
        "2026-10-30 19:00:00"
    )

    monkeypatch.setattr(
        api,
        "_current_copenhagen_time",
        lambda: fixed_now
    )

    start = (
        api._resolve_start_date(
            "now-P1D"
        )
    )

    end = (
        api._resolve_end_date(
            "now"
        )
    )

    assert start == pd.Timestamp(
        "2026-10-29 19:00:00"
    )

    assert end == fixed_now


def test_timezone_aware_input_is_converted_to_copenhagen():

    api = WeatherAPI()

    result = (
        api._resolve_start_date(
            "2026-10-30 10:00:00-07:00"
        )
    )

    # On October 30 Copenhagen is UTC+1.
    # 10:00 PDT = 18:00 Copenhagen.

    assert result == pd.Timestamp(
        "2026-10-30 18:00:00"
    )

    assert result.tzinfo is None


def test_convert_response_returns_chronological_data():

    api = WeatherAPI()

    response = (
        create_weather_response(
            [
                "2026-10-30T12:00",
                "2026-10-30T10:00",
                "2026-10-30T11:00"
            ]
        )
    )

    result = (
        api._convert_response(
            response,
            "DK1"
        )
    )

    assert result[
        "TimeDK"
    ].is_monotonic_increasing


def test_convert_response_removes_duplicate_times():

    api = WeatherAPI()

    response = (
        create_weather_response(
            [
                "2026-10-30T10:00",
                "2026-10-30T10:00",
                "2026-10-30T11:00"
            ]
        )
    )

    result = (
        api._convert_response(
            response,
            "DK1"
        )
    )

    assert not result[
        "TimeDK"
    ].duplicated().any()


def test_forecast_only_returns_future_copenhagen_hours(
    monkeypatch
):

    api = WeatherAPI()

    fixed_now = pd.Timestamp(
        "2026-10-30 19:30:00"
    )

    monkeypatch.setattr(
        api,
        "_current_copenhagen_time",
        lambda: fixed_now
    )

    response_data = (
        create_weather_response(
            [
                "2026-10-30T19:00",
                "2026-10-30T20:00",
                "2026-10-30T21:00",
                "2026-10-30T22:00"
            ]
        )
    )

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs:
            FakeResponse(
                response_data
            )
    )

    result = (
        api.get_forecast_weather(
            price_area="DK1",
            forecast_hours=2
        )
    )

    assert len(
        result
    ) == 2

    assert result[
        "TimeDK"
    ].iloc[
        0
    ] == pd.Timestamp(
        "2026-10-30 20:00:00"
    )

    assert (
        result[
            "TimeDK"
        ]
        >
        fixed_now.floor(
            "h"
        )
    ).all()


def test_invalid_forecast_hours_returns_empty():

    api = WeatherAPI()

    result = (
        api.get_forecast_weather(
            price_area="DK1",
            forecast_hours="invalid"
        )
    )

    assert result.empty


def test_forecast_request_failure_returns_empty(
    monkeypatch
):

    api = WeatherAPI()

    def raise_error(
        *args,
        **kwargs
    ):

        raise (
            requests.exceptions
            .RequestException(
                "API unavailable"
            )
        )

    monkeypatch.setattr(
        requests,
        "get",
        raise_error
    )

    result = (
        api.get_forecast_weather(
            price_area="DK1",
            forecast_hours=24
        )
    )

    assert result.empty