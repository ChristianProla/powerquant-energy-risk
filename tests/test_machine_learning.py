import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.forecasting.machine_learning import (
    MachineLearningForecaster
)


def create_feature_data(
    observations=120
):

    times = pd.date_range(
        "2026-01-01",
        periods=observations,
        freq="h"
    )

    price = (
        100
        + np.arange(
            observations,
            dtype=float
        )
        * 0.25
    )

    dataframe = pd.DataFrame(
        {
            "TimeDK":
                times,

            "DayAheadPriceEUR":
                price,

            "PriceLag1":
                pd.Series(
                    price
                ).shift(
                    1
                ),

            "PriceLag2":
                pd.Series(
                    price
                ).shift(
                    2
                ),

            "PriceLag24":
                pd.Series(
                    price
                ).shift(
                    24
                ),

            "PriceMovingAverage6":
                pd.Series(
                    price
                ).rolling(
                    6
                ).mean(),

            "PriceMovingAverage24":
                pd.Series(
                    price
                ).rolling(
                    24
                ).mean(),

            "PriceVolatility24":
                pd.Series(
                    price
                ).rolling(
                    24
                ).std(),

            "HourOfDay":
                times.hour,

            "DayOfWeek":
                times.dayofweek,

            "Month":
                times.month,

            "IsWeekend":
                (
                    times.dayofweek
                    >= 5
                ).astype(
                    int
                ),

            "WindGenerationMW":
                1000
                + np.arange(
                    observations
                ),

            "ElectricityDemandMW":
                2000
                + np.arange(
                    observations
                ),

            "WeatherWindSpeedMS":
                5.0,

            "TemperatureC":
                20.0,

            "CloudCoverPct":
                50.0,

            "PrecipitationMM":
                0.0
        }
    )

    dataframe[
        "WindDemandRatio"
    ] = (
        dataframe[
            "WindGenerationMW"
        ]
        /
        dataframe[
            "ElectricityDemandMW"
        ]
    )

    dataframe[
        "WindLag1"
    ] = (
        dataframe[
            "WindGenerationMW"
        ].shift(
            1
        )
    )

    dataframe[
        "DemandLag1"
    ] = (
        dataframe[
            "ElectricityDemandMW"
        ].shift(
            1
        )
    )

    dataframe[
        "WeatherWindLag1"
    ] = (
        dataframe[
            "WeatherWindSpeedMS"
        ].shift(
            1
        )
    )

    dataframe[
        "TemperatureLag1"
    ] = (
        dataframe[
            "TemperatureC"
        ].shift(
            1
        )
    )

    dataframe[
        "CloudCoverLag1"
    ] = (
        dataframe[
            "CloudCoverPct"
        ].shift(
            1
        )
    )

    dataframe[
        "WeatherWindAverage6"
    ] = (
        dataframe[
            "WeatherWindSpeedMS"
        ].rolling(
            6
        ).mean()
    )

    dataframe[
        "TemperatureAverage6"
    ] = (
        dataframe[
            "TemperatureC"
        ].rolling(
            6
        ).mean()
    )

    target_time = (
        dataframe[
            "TimeDK"
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    dataframe[
        "TargetTemperatureC"
    ] = 20.0

    dataframe[
        "TargetWeatherWindSpeedMS"
    ] = 5.0

    dataframe[
        "TargetCloudCoverPct"
    ] = 50.0

    dataframe[
        "TargetPrecipitationMM"
    ] = 0.0

    dataframe[
        "TargetHourOfDay"
    ] = (
        target_time.dt.hour
    )

    dataframe[
        "TargetDayOfWeek"
    ] = (
        target_time.dt.dayofweek
    )

    dataframe[
        "TargetMonth"
    ] = (
        target_time.dt.month
    )

    dataframe[
        "TargetIsWeekend"
    ] = (
        target_time.dt.dayofweek
        >= 5
    ).astype(
        int
    )

    return dataframe


def test_model_target_is_next_price():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data()

    model_data, error = (
        forecaster
        ._prepare_model_data(
            data,
            forecaster.feature_columns
        )
    )

    assert error is None

    row = model_data.iloc[
        10
    ]

    source_time = row[
        "TimeDK"
    ]

    source_index = (
        data.index[
            data[
                "TimeDK"
            ] == source_time
        ][
            0
        ]
    )

    assert row[
        "NextPrice"
    ] == pytest.approx(
        data.iloc[
            source_index + 1
        ][
            "DayAheadPriceEUR"
        ]
    )


def test_linear_regression_uses_standardization_pipeline():

    forecaster = (
        MachineLearningForecaster()
    )

    model = (
        forecaster
        ._create_model(
            "Linear Regression"
        )
    )

    assert isinstance(
        model,
        Pipeline
    )

    assert "scaler" in (
        model.named_steps
    )

    assert "regressor" in (
        model.named_steps
    )


def test_time_series_split_has_gap():

    forecaster = (
        MachineLearningForecaster()
    )

    splitter = (
        forecaster
        ._create_splitter(
            3
        )
    )

    data = np.arange(
        100
    )

    for (
        train_index,
        test_index
    ) in splitter.split(
        data
    ):

        assert (
            test_index[
                0
            ]
            -
            train_index[
                -1
            ]
            >= 2
        )


def test_latest_feature_row_uses_latest_known_price():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data()

    latest = (
        forecaster
        ._get_latest_feature_row(
            data,
            forecaster.feature_columns
        )
    )

    assert latest is not None

    assert latest[
        "DayAheadPriceEUR"
    ].iloc[
        0
    ] == pytest.approx(
        data[
            "DayAheadPriceEUR"
        ].iloc[
            -1
        ]
    )


def test_compare_models_current_price_is_latest_price():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data(
        observations=140
    )

    result = (
        forecaster
        .compare_models(
            data,
            n_splits=3
        )
    )

    assert result[
        "success"
    ]

    assert result[
        "current_price"
    ] == pytest.approx(
        data[
            "DayAheadPriceEUR"
        ].iloc[
            -1
        ]
    )


def test_feature_ablation_current_price_is_latest_price():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data(
        observations=140
    )

    result = (
        forecaster
        .compare_feature_sets(
            data,
            model_name="Linear Regression",
            n_splits=3
        )
    )

    assert result[
        "success"
    ]

    assert result[
        "current_price"
    ] == pytest.approx(
        data[
            "DayAheadPriceEUR"
        ].iloc[
            -1
        ]
    )


def test_unknown_ablation_model_is_rejected():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data()

    result = (
        forecaster
        .compare_feature_sets(
            data,
            model_name="Fake Model"
        )
    )

    assert not result[
        "success"
    ]


def test_linear_feature_importance_is_normalized():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data(
        observations=140
    )

    model_data, error = (
        forecaster
        ._prepare_model_data(
            data,
            forecaster.feature_columns
        )
    )

    assert error is None

    model = (
        forecaster
        ._create_model(
            "Linear Regression"
        )
    )

    model.fit(
        model_data[
            forecaster.feature_columns
        ],
        model_data[
            "NextPrice"
        ]
    )

    importance = (
        forecaster
        ._feature_importance_from_model(
            model,
            forecaster.feature_columns
        )
    )

    assert not importance.empty

    assert importance[
        "Importance"
    ].sum() == pytest.approx(
        1.0
    )


def test_forward_forecast_requires_25_prices_for_true_lag24():

    forecaster = (
        MachineLearningForecaster()
    )

    data = create_feature_data(
        observations=24
    )

    result = (
        forecaster
        .create_forward_forecast(
            data,
            price_area="DK1",
            horizon=1,
            n_splits=2
        )
    )

    assert not result[
        "success"
    ]