import numpy as np
import pandas as pd
import pytest

from src.forecasting.feature_engineering import (
    FeatureEngineer
)


def create_input_data(
    observations=60
):

    times = pd.date_range(
        "2026-01-01",
        periods=observations,
        freq="h"
    )

    prices = pd.DataFrame(
        {
            "TimeDK":
                times,

            "DayAheadPriceEUR":
                np.arange(
                    observations,
                    dtype=float
                )
                + 100
        }
    )

    factors = pd.DataFrame(
        {
            "TimeDK":
                times,

            "WindGenerationMW":
                np.arange(
                    observations,
                    dtype=float
                )
                + 1000,

            "ElectricityDemandMW":
                np.arange(
                    observations,
                    dtype=float
                )
                + 2000,

            "TemperatureC":
                np.arange(
                    observations,
                    dtype=float
                ),

            "WeatherWindSpeedMS":
                np.arange(
                    observations,
                    dtype=float
                )
                / 10,

            "CloudCoverPct":
                np.arange(
                    observations,
                    dtype=float
                ),

            "PrecipitationMM":
                np.arange(
                    observations,
                    dtype=float
                )
                / 100
        }
    )

    return prices, factors


def test_feature_data_is_chronological():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    prices = (
        prices.sample(
            frac=1,
            random_state=42
        )
    )

    factors = (
        factors.sample(
            frac=1,
            random_state=42
        )
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    assert not result.empty

    assert result[
        "TimeDK"
    ].is_monotonic_increasing


def test_price_lags_use_previous_prices():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    row = result.iloc[
        30
    ]

    assert row[
        "PriceLag1"
    ] == pytest.approx(
        result.iloc[
            29
        ][
            "DayAheadPriceEUR"
        ]
    )

    assert row[
        "PriceLag2"
    ] == pytest.approx(
        result.iloc[
            28
        ][
            "DayAheadPriceEUR"
        ]
    )

    assert row[
        "PriceLag24"
    ] == pytest.approx(
        result.iloc[
            6
        ][
            "DayAheadPriceEUR"
        ]
    )


def test_target_calendar_describes_next_hour():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    row = result.iloc[
        10
    ]

    expected_time = (
        row[
            "TimeDK"
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    assert row[
        "TargetTime"
    ] == expected_time

    assert row[
        "TargetHourOfDay"
    ] == expected_time.hour

    assert row[
        "TargetDayOfWeek"
    ] == expected_time.dayofweek


def test_target_weather_uses_next_realized_hour():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    row = result.iloc[
        20
    ]

    next_row = result.iloc[
        21
    ]

    assert row[
        "TargetTemperatureC"
    ] == pytest.approx(
        next_row[
            "TemperatureC"
        ]
    )

    assert row[
        "TargetWeatherWindSpeedMS"
    ] == pytest.approx(
        next_row[
            "WeatherWindSpeedMS"
        ]
    )


def test_future_price_change_does_not_change_earlier_price_features():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data(
            observations=80
        )
    )

    original = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    modified_prices = (
        prices.copy()
    )

    modified_prices.loc[
        60:,
        "DayAheadPriceEUR"
    ] = 100000

    modified = (
        engineer
        .create_market_features(
            modified_prices,
            factors
        )
    )

    comparison_index = 50

    columns = [
        "PriceLag1",
        "PriceLag2",
        "PriceLag24",
        "PriceMovingAverage6",
        "PriceMovingAverage24",
        "PriceVolatility24"
    ]

    for column in columns:

        assert original.iloc[
            comparison_index
        ][
            column
        ] == pytest.approx(
            modified.iloc[
                comparison_index
            ][
                column
            ]
        )


def test_future_weather_only_affects_target_weather_when_relevant():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data(
            observations=80
        )
    )

    original = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    modified_factors = (
        factors.copy()
    )

    modified_factors.loc[
        60:,
        "TemperatureC"
    ] = 1000

    modified = (
        engineer
        .create_market_features(
            prices,
            modified_factors
        )
    )

    comparison_index = 50

    assert original.iloc[
        comparison_index
    ][
        "TemperatureLag1"
    ] == pytest.approx(
        modified.iloc[
            comparison_index
        ][
            "TemperatureLag1"
        ]
    )

    assert original.iloc[
        comparison_index
    ][
        "TemperatureAverage6"
    ] == pytest.approx(
        modified.iloc[
            comparison_index
        ][
            "TemperatureAverage6"
        ]
    )


def test_zero_demand_does_not_create_infinite_ratio():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    factors.loc[
        20,
        "ElectricityDemandMW"
    ] = 0

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    assert not np.isinf(
        result[
            "WindDemandRatio"
        ]
    ).any()


def test_missing_required_price_column_returns_empty():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    prices = prices.drop(
        columns=[
            "DayAheadPriceEUR"
        ]
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    assert result.empty


def test_realized_weather_proxy_is_explicitly_flagged():

    engineer = (
        FeatureEngineer()
    )

    prices, factors = (
        create_input_data()
    )

    result = (
        engineer
        .create_market_features(
            prices,
            factors
        )
    )

    assert (
        result[
            "UsesRealizedTargetWeatherProxy"
        ]
        .all()
    )