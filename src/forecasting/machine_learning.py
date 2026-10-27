import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    TimeSeriesSplit
)

from src.api.weather_api import (
    WeatherAPI
)


class MachineLearningForecaster:

    def __init__(self):

        self.weather_api = (
            WeatherAPI()
        )

        # ==================================================
        # Historical feature groups
        # ==================================================

        self.price_features = [
            "DayAheadPriceEUR",
            "PriceLag1",
            "PriceLag2",
            "PriceLag24",
            "PriceMovingAverage6",
            "PriceMovingAverage24",
            "PriceVolatility24",
            "HourOfDay",
            "DayOfWeek",
            "Month",
            "IsWeekend"
        ]

        self.market_features = [
            "WindGenerationMW",
            "ElectricityDemandMW",
            "WindDemandRatio",
            "WindLag1",
            "DemandLag1"
        ]

        self.weather_features = [
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM",
            "WeatherWindLag1",
            "TemperatureLag1",
            "CloudCoverLag1",
            "WeatherWindAverage6",
            "TemperatureAverage6"
        ]

        self.feature_columns = (
            self.price_features
            + self.market_features
            + self.weather_features
        )

        # ==================================================
        # Ablation feature sets
        # ==================================================

        self.feature_sets = {
            "Price History":
                self.price_features,

            "Price + Market Factors":
                (
                    self.price_features
                    + self.market_features
                ),

            "Price + Market + Weather":
                (
                    self.price_features
                    + self.market_features
                    + self.weather_features
                )
        }

        # ==================================================
        # Forward forecasting features
        #
        # Important:
        # We do NOT use future electricity demand
        # or future wind-generation observations here,
        # because they are not known at forecast time.
        #
        # Instead we use:
        # - historical prices
        # - target-hour calendar
        # - future weather forecast
        # ==================================================

        self.forward_feature_columns = [
            "DayAheadPriceEUR",
            "PriceLag1",
            "PriceLag2",
            "PriceLag24",
            "PriceMovingAverage6",
            "PriceMovingAverage24",
            "PriceVolatility24",
            "TargetTemperatureC",
            "TargetWeatherWindSpeedMS",
            "TargetCloudCoverPct",
            "TargetPrecipitationMM",
            "TargetHourOfDay",
            "TargetDayOfWeek",
            "TargetMonth",
            "TargetIsWeekend"
        ]

    # ==================================================
    # Create model
    # ==================================================

    def _create_model(
        self,
        model_name,
        random_state=42
    ):

        if (
            model_name
            == "Linear Regression"
        ):

            return (
                LinearRegression()
            )

        if (
            model_name
            == "Gradient Boosting"
        ):

            return (
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state
                )
            )

        return (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        )

    def _create_models(
        self,
        random_state=42
    ):

        return {
            "Linear Regression":
                self._create_model(
                    "Linear Regression",
                    random_state
                ),

            "Random Forest":
                self._create_model(
                    "Random Forest",
                    random_state
                ),

            "Gradient Boosting":
                self._create_model(
                    "Gradient Boosting",
                    random_state
                )
        }

    # ==================================================
    # Prepare historical model data
    # ==================================================

    def _prepare_model_data(
        self,
        feature_dataframe,
        feature_columns
    ):

        if feature_dataframe.empty:

            return None, (
                "No feature data available."
            )

        data = (
            feature_dataframe.copy()
        )

        missing_columns = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if missing_columns:

            return None, (
                "Missing required features: "
                + ", ".join(
                    missing_columns
                )
            )

        data[
            "NextPrice"
        ] = (
            data[
                "DayAheadPriceEUR"
            ].shift(
                -1
            )
        )

        data = data.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        model_data = (
            data.dropna(
                subset=(
                    feature_columns
                    + [
                        "NextPrice"
                    ]
                )
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        if (
            len(
                model_data
            )
            < 40
        ):

            return None, (
                "Not enough complete observations "
                "for machine-learning validation. "
                "Try using 1 Month or 1 Year."
            )

        return (
            model_data,
            None
        )

    # ==================================================
    # Standard model comparison
    # ==================================================

    def compare_models(
        self,
        feature_dataframe,
        n_splits=5,
        random_state=42
    ):

        (
            model_data,
            error_message
        ) = self._prepare_model_data(
            feature_dataframe,
            self.feature_columns
        )

        if (
            error_message
            is not None
        ):

            return (
                self._empty_comparison_result(
                    error_message
                )
            )

        X = model_data[
            self.feature_columns
        ]

        y = model_data[
            "NextPrice"
        ]

        times = model_data[
            "TimeDK"
        ]

        n_splits = self._safe_splits(
            len(
                model_data
            ),
            n_splits
        )

        if (
            n_splits
            < 2
        ):

            return (
                self._empty_comparison_result(
                    "Not enough data for time-series validation."
                )
            )

        splitter = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        predictions_store = {
            "Naive Baseline":
                [],

            "Linear Regression":
                [],

            "Random Forest":
                [],

            "Gradient Boosting":
                []
        }

        actual_values = []

        prediction_times = []

        fold_results = []

        # ==================================================
        # Walk-forward validation
        # ==================================================

        for (
            fold_number,
            (
                train_index,
                test_index
            )
        ) in enumerate(
            splitter.split(
                X
            ),
            start=1
        ):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]

            y_test = y.iloc[
                test_index
            ]

            test_times = times.iloc[
                test_index
            ]

            baseline_predictions = (
                X_test[
                    "DayAheadPriceEUR"
                ].to_numpy()
            )

            predictions_store[
                "Naive Baseline"
            ].extend(
                baseline_predictions
            )

            models = (
                self._create_models(
                    random_state
                )
            )

            fold_scores = {}

            for (
                model_name,
                model
            ) in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                predictions = (
                    model.predict(
                        X_test
                    )
                )

                predictions_store[
                    model_name
                ].extend(
                    predictions
                )

                fold_scores[
                    model_name
                ] = (
                    mean_absolute_error(
                        y_test,
                        predictions
                    )
                )

            fold_results.append(
                {
                    "fold":
                        fold_number,

                    "train_size":
                        len(
                            train_index
                        ),

                    "test_size":
                        len(
                            test_index
                        ),

                    "baseline_mae":
                        mean_absolute_error(
                            y_test,
                            baseline_predictions
                        ),

                    "linear_regression_mae":
                        fold_scores[
                            "Linear Regression"
                        ],

                    "random_forest_mae":
                        fold_scores[
                            "Random Forest"
                        ],

                    "gradient_boosting_mae":
                        fold_scores[
                            "Gradient Boosting"
                        ]
                }
            )

            actual_values.extend(
                y_test.to_numpy()
            )

            prediction_times.extend(
                test_times.to_numpy()
            )

        actual_values = np.array(
            actual_values
        )

        prediction_times = np.array(
            prediction_times
        )

        comparison_rows = []

        baseline_mae = 0

        for (
            model_name,
            predictions
        ) in predictions_store.items():

            predictions = np.array(
                predictions
            )

            mae = (
                mean_absolute_error(
                    actual_values,
                    predictions
                )
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual_values,
                    predictions
                )
            )

            if (
                len(
                    actual_values
                )
                >= 2
            ):

                r2 = (
                    r2_score(
                        actual_values,
                        predictions
                    )
                )

            else:

                r2 = 0

            if (
                model_name
                == "Naive Baseline"
            ):

                baseline_mae = mae

            comparison_rows.append(
                {
                    "Model":
                        model_name,

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2
                }
            )

        comparison_table = (
            pd.DataFrame(
                comparison_rows
            )
        )

        improvements = []

        for _, row in (
            comparison_table.iterrows()
        ):

            improvements.append(
                self._calculate_improvement(
                    baseline_mae,
                    row[
                        "MAE"
                    ]
                )
            )

        comparison_table[
            "ImprovementVsBaseline"
        ] = improvements

        comparison_table = (
            comparison_table.sort_values(
                by="MAE"
            )
            .reset_index(
                drop=True
            )
        )

        best_model_name = (
            comparison_table.iloc[
                0
            ][
                "Model"
            ]
        )

        # ==================================================
        # Final models
        # ==================================================

        fitted_models = {}

        models = (
            self._create_models(
                random_state
            )
        )

        for (
            model_name,
            model
        ) in models.items():

            model.fit(
                X,
                y
            )

            fitted_models[
                model_name
            ] = model

        current_price = (
            model_data[
                "DayAheadPriceEUR"
            ].iloc[
                -1
            ]
        )

        if (
            best_model_name
            == "Naive Baseline"
        ):

            next_price_prediction = (
                current_price
            )

            feature_importance = (
                pd.DataFrame()
            )

        else:

            best_model = (
                fitted_models[
                    best_model_name
                ]
            )

            latest_features = (
                model_data[
                    self.feature_columns
                ].iloc[
                    [-1]
                ]
            )

            next_price_prediction = (
                best_model.predict(
                    latest_features
                )[0]
            )

            feature_importance = (
                self._feature_importance_from_model(
                    best_model,
                    self.feature_columns
                )
            )

        predicted_change = (
            next_price_prediction
            - current_price
        )

        if (
            current_price
            != 0
        ):

            predicted_change_percent = (
                predicted_change
                / abs(
                    current_price
                )
            ) * 100

        else:

            predicted_change_percent = 0

        return {
            "success":
                True,

            "message":
                "Model comparison completed successfully.",

            "comparison_table":
                comparison_table,

            "best_model":
                best_model_name,

            "current_price":
                current_price,

            "next_price_prediction":
                next_price_prediction,

            "predicted_change":
                predicted_change,

            "predicted_change_percent":
                predicted_change_percent,

            "actual_prices":
                actual_values,

            "best_predictions":
                np.array(
                    predictions_store[
                        best_model_name
                    ]
                ),

            "prediction_times":
                prediction_times,

            "feature_importance":
                feature_importance,

            "fold_results":
                pd.DataFrame(
                    fold_results
                ),

            "total_observations":
                len(
                    model_data
                ),

            "validation_observations":
                len(
                    actual_values
                ),

            "n_splits":
                n_splits
        }

    # ==================================================
    # Feature ablation
    # ==================================================

    def compare_feature_sets(
        self,
        feature_dataframe,
        model_name="Random Forest",
        n_splits=5,
        random_state=42
    ):

        (
            full_model_data,
            error_message
        ) = self._prepare_model_data(
            feature_dataframe,
            self.feature_columns
        )

        if (
            error_message
            is not None
        ):

            return (
                self._empty_ablation_result(
                    error_message
                )
            )

        n_splits = self._safe_splits(
            len(
                full_model_data
            ),
            n_splits
        )

        if (
            n_splits
            < 2
        ):

            return (
                self._empty_ablation_result(
                    "Not enough data for feature ablation testing."
                )
            )

        y = full_model_data[
            "NextPrice"
        ]

        times = full_model_data[
            "TimeDK"
        ]

        splitter = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        folds = list(
            splitter.split(
                full_model_data
            )
        )

        results = []

        prediction_store = {}

        actual_values = None

        prediction_times = None

        for (
            feature_set_name,
            feature_columns
        ) in self.feature_sets.items():

            X = full_model_data[
                feature_columns
            ]

            all_predictions = []

            all_actual = []

            all_times = []

            fold_mae_values = []

            for (
                train_index,
                test_index
            ) in folds:

                X_train = X.iloc[
                    train_index
                ]

                X_test = X.iloc[
                    test_index
                ]

                y_train = y.iloc[
                    train_index
                ]

                y_test = y.iloc[
                    test_index
                ]

                model = (
                    self._create_model(
                        model_name,
                        random_state
                    )
                )

                model.fit(
                    X_train,
                    y_train
                )

                predictions = (
                    model.predict(
                        X_test
                    )
                )

                all_predictions.extend(
                    predictions
                )

                all_actual.extend(
                    y_test.to_numpy()
                )

                all_times.extend(
                    times.iloc[
                        test_index
                    ].to_numpy()
                )

                fold_mae_values.append(
                    mean_absolute_error(
                        y_test,
                        predictions
                    )
                )

            all_predictions = np.array(
                all_predictions
            )

            all_actual = np.array(
                all_actual
            )

            all_times = np.array(
                all_times
            )

            mae = (
                mean_absolute_error(
                    all_actual,
                    all_predictions
                )
            )

            rmse = np.sqrt(
                mean_squared_error(
                    all_actual,
                    all_predictions
                )
            )

            if (
                len(
                    all_actual
                )
                >= 2
            ):

                r2 = (
                    r2_score(
                        all_actual,
                        all_predictions
                    )
                )

            else:

                r2 = 0

            results.append(
                {
                    "FeatureSet":
                        feature_set_name,

                    "FeatureCount":
                        len(
                            feature_columns
                        ),

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2,

                    "FoldMAEMean":
                        np.mean(
                            fold_mae_values
                        ),

                    "FoldMAEStd":
                        np.std(
                            fold_mae_values
                        )
                }
            )

            prediction_store[
                feature_set_name
            ] = all_predictions

            if (
                actual_values
                is None
            ):

                actual_values = (
                    all_actual
                )

                prediction_times = (
                    all_times
                )

        comparison_table = (
            pd.DataFrame(
                results
            )
        )

        price_mae = (
            self._get_feature_set_metric(
                comparison_table,
                "Price History",
                "MAE"
            )
        )

        market_mae = (
            self._get_feature_set_metric(
                comparison_table,
                "Price + Market Factors",
                "MAE"
            )
        )

        weather_mae = (
            self._get_feature_set_metric(
                comparison_table,
                "Price + Market + Weather",
                "MAE"
            )
        )

        comparison_table[
            "ImprovementVsPriceOnly"
        ] = (
            comparison_table[
                "MAE"
            ].apply(
                lambda value:
                self._calculate_improvement(
                    price_mae,
                    value
                )
            )
        )

        comparison_table = (
            comparison_table.sort_values(
                by="MAE"
            )
            .reset_index(
                drop=True
            )
        )

        best_feature_set = (
            comparison_table.iloc[
                0
            ][
                "FeatureSet"
            ]
        )

        best_features = (
            self.feature_sets[
                best_feature_set
            ]
        )

        final_model = (
            self._create_model(
                model_name,
                random_state
            )
        )

        final_model.fit(
            full_model_data[
                best_features
            ],
            y
        )

        feature_importance = (
            self._feature_importance_from_model(
                final_model,
                best_features
            )
        )

        current_price = (
            full_model_data[
                "DayAheadPriceEUR"
            ].iloc[
                -1
            ]
        )

        next_price_prediction = (
            final_model.predict(
                full_model_data[
                    best_features
                ].iloc[
                    [-1]
                ]
            )[0]
        )

        predicted_change = (
            next_price_prediction
            - current_price
        )

        if (
            current_price
            != 0
        ):

            predicted_change_percent = (
                predicted_change
                / abs(
                    current_price
                )
            ) * 100

        else:

            predicted_change_percent = 0

        return {
            "success":
                True,

            "message":
                "Feature ablation testing completed successfully.",

            "model_name":
                model_name,

            "comparison_table":
                comparison_table,

            "best_feature_set":
                best_feature_set,

            "market_improvement":
                self._calculate_improvement(
                    price_mae,
                    market_mae
                ),

            "weather_improvement_vs_price":
                self._calculate_improvement(
                    price_mae,
                    weather_mae
                ),

            "weather_incremental_improvement":
                self._calculate_improvement(
                    market_mae,
                    weather_mae
                ),

            "actual_prices":
                actual_values,

            "prediction_times":
                prediction_times,

            "best_predictions":
                prediction_store[
                    best_feature_set
                ],

            "prediction_store":
                prediction_store,

            "feature_importance":
                feature_importance,

            "current_price":
                current_price,

            "next_price_prediction":
                next_price_prediction,

            "predicted_change":
                predicted_change,

            "predicted_change_percent":
                predicted_change_percent,

            "total_observations":
                len(
                    full_model_data
                ),

            "validation_observations":
                len(
                    actual_values
                ),

            "n_splits":
                n_splits
        }

    # ==================================================
    # NEW:
    # Forward-looking forecast
    # ==================================================

    def create_forward_forecast(
        self,
        feature_dataframe,
        price_area,
        horizon=24,
        n_splits=5,
        random_state=42
    ):

        (
            model_data,
            error_message
        ) = self._prepare_model_data(
            feature_dataframe,
            self.forward_feature_columns
        )

        if (
            error_message
            is not None
        ):

            return (
                self._empty_forward_result(
                    error_message
                )
            )

        horizon = max(
            1,
            min(
                int(
                    horizon
                ),
                24
            )
        )

        # ==================================================
        # Validate models specifically for forward forecast
        # ==================================================

        X = model_data[
            self.forward_feature_columns
        ]

        y = model_data[
            "NextPrice"
        ]

        n_splits = self._safe_splits(
            len(
                model_data
            ),
            n_splits
        )

        if (
            n_splits
            < 2
        ):

            return (
                self._empty_forward_result(
                    "Not enough observations for forward-model validation."
                )
            )

        splitter = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        validation_predictions = {
            "Linear Regression":
                [],

            "Random Forest":
                [],

            "Gradient Boosting":
                []
        }

        validation_actual = []

        for (
            train_index,
            test_index
        ) in splitter.split(
            X
        ):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]

            y_test = y.iloc[
                test_index
            ]

            models = (
                self._create_models(
                    random_state
                )
            )

            for (
                model_name,
                model
            ) in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                predictions = (
                    model.predict(
                        X_test
                    )
                )

                validation_predictions[
                    model_name
                ].extend(
                    predictions
                )

            validation_actual.extend(
                y_test.to_numpy()
            )

        validation_actual = (
            np.array(
                validation_actual
            )
        )

        comparison_rows = []

        for (
            model_name,
            predictions
        ) in (
            validation_predictions.items()
        ):

            predictions = np.array(
                predictions
            )

            mae = (
                mean_absolute_error(
                    validation_actual,
                    predictions
                )
            )

            rmse = np.sqrt(
                mean_squared_error(
                    validation_actual,
                    predictions
                )
            )

            r2 = (
                r2_score(
                    validation_actual,
                    predictions
                )
                if (
                    len(
                        validation_actual
                    )
                    >= 2
                )
                else 0
            )

            comparison_rows.append(
                {
                    "Model":
                        model_name,

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2
                }
            )

        model_comparison = (
            pd.DataFrame(
                comparison_rows
            )
            .sort_values(
                by="MAE"
            )
            .reset_index(
                drop=True
            )
        )

        best_model_name = (
            model_comparison.iloc[
                0
            ][
                "Model"
            ]
        )

        validation_mae = (
            model_comparison.iloc[
                0
            ][
                "MAE"
            ]
        )

        validation_rmse = (
            model_comparison.iloc[
                0
            ][
                "RMSE"
            ]
        )

        validation_r2 = (
            model_comparison.iloc[
                0
            ][
                "R2"
            ]
        )

        # ==================================================
        # Train best model on all historical data
        # ==================================================

        model = (
            self._create_model(
                best_model_name,
                random_state
            )
        )

        model.fit(
            X,
            y
        )

        feature_importance = (
            self._feature_importance_from_model(
                model,
                self.forward_feature_columns
            )
        )

        # ==================================================
        # Fetch future weather forecast
        # ==================================================

        try:

            future_weather = (
                self.weather_api
                .get_forecast_weather(
                    price_area=price_area,
                    forecast_hours=horizon
                )
            )

        except Exception as error:

            return (
                self._empty_forward_result(
                    (
                        "Could not load future weather forecast: "
                        f"{error}"
                    )
                )
            )

        if future_weather.empty:

            return (
                self._empty_forward_result(
                    "No future weather forecast was returned."
                )
            )

        available_horizon = min(
            horizon,
            len(
                future_weather
            )
        )

        future_weather = (
            future_weather.head(
                available_horizon
            )
            .reset_index(
                drop=True
            )
        )

        # ==================================================
        # Historical prices used by recursive forecast
        # ==================================================

        historical_data = (
            feature_dataframe[
                [
                    "TimeDK",
                    "DayAheadPriceEUR"
                ]
            ]
            .dropna()
            .sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        price_history = (
            historical_data[
                "DayAheadPriceEUR"
            ]
            .astype(
                float
            )
            .tolist()
        )

        if (
            len(
                price_history
            )
            < 24
        ):

            return (
                self._empty_forward_result(
                    (
                        "At least 24 hourly price observations "
                        "are required for a forward forecast."
                    )
                )
            )

        current_price = (
            price_history[
                -1
            ]
        )

        forecast_times = []

        forecast_prices = []

        # ==================================================
        # Recursive multi-hour forecast
        # ==================================================

        for (
            weather_index,
            weather_row
        ) in (
            future_weather.iterrows()
        ):

            target_time = pd.Timestamp(
                weather_row[
                    "TimeDK"
                ]
            )

            # Current price available at this point
            model_current_price = (
                price_history[
                    -1
                ]
            )

            price_lag_1 = (
                price_history[
                    -2
                ]
            )

            price_lag_2 = (
                price_history[
                    -3
                ]
            )

            price_lag_24 = (
                price_history[
                    -24
                ]
            )

            recent_6 = np.array(
                price_history[
                    -6:
                ],
                dtype=float
            )

            recent_24 = np.array(
                price_history[
                    -24:
                ],
                dtype=float
            )

            moving_average_6 = (
                recent_6.mean()
            )

            moving_average_24 = (
                recent_24.mean()
            )

            if (
                len(
                    recent_24
                )
                >= 2
            ):

                volatility_24 = (
                    recent_24.std(
                        ddof=1
                    )
                )

            else:

                volatility_24 = 0

            target_day_of_week = (
                target_time.dayofweek
            )

            target_is_weekend = (
                1
                if (
                    target_day_of_week
                    >= 5
                )
                else 0
            )

            input_row = pd.DataFrame(
                [
                    {
                        "DayAheadPriceEUR":
                            model_current_price,

                        "PriceLag1":
                            price_lag_1,

                        "PriceLag2":
                            price_lag_2,

                        "PriceLag24":
                            price_lag_24,

                        "PriceMovingAverage6":
                            moving_average_6,

                        "PriceMovingAverage24":
                            moving_average_24,

                        "PriceVolatility24":
                            volatility_24,

                        "TargetTemperatureC":
                            weather_row[
                                "TemperatureC"
                            ],

                        "TargetWeatherWindSpeedMS":
                            weather_row[
                                "WeatherWindSpeedMS"
                            ],

                        "TargetCloudCoverPct":
                            weather_row[
                                "CloudCoverPct"
                            ],

                        "TargetPrecipitationMM":
                            weather_row[
                                "PrecipitationMM"
                            ],

                        "TargetHourOfDay":
                            target_time.hour,

                        "TargetDayOfWeek":
                            target_day_of_week,

                        "TargetMonth":
                            target_time.month,

                        "TargetIsWeekend":
                            target_is_weekend
                    }
                ]
            )

            if (
                input_row[
                    self.forward_feature_columns
                ]
                .isna()
                .any()
                .any()
            ):

                continue

            predicted_price = (
                model.predict(
                    input_row[
                        self.forward_feature_columns
                    ]
                )[0]
            )

            forecast_times.append(
                target_time
            )

            forecast_prices.append(
                predicted_price
            )

            # Add prediction back into history.
            # The next forecast step can now use
            # this predicted value.

            price_history.append(
                predicted_price
            )

        if (
            len(
                forecast_prices
            )
            == 0
        ):

            return (
                self._empty_forward_result(
                    (
                        "The forecast could not be generated "
                        "because future weather observations "
                        "contained missing values."
                    )
                )
            )

        forecast_prices = np.array(
            forecast_prices
        )

        forecast_times = np.array(
            forecast_times
        )

        ending_price = (
            forecast_prices[
                -1
            ]
        )

        price_change = (
            ending_price
            - current_price
        )

        if (
            current_price
            != 0
        ):

            price_change_percent = (
                price_change
                / abs(
                    current_price
                )
            ) * 100

        else:

            price_change_percent = 0

        average_forecast_price = (
            forecast_prices.mean()
        )

        maximum_forecast_price = (
            forecast_prices.max()
        )

        minimum_forecast_price = (
            forecast_prices.min()
        )

        # ==================================================
        # History for chart
        # ==================================================

        history_chart = (
            historical_data.tail(
                48
            )
        )

        return {
            "success":
                True,

            "message":
                "Forward-looking electricity price forecast completed.",

            "best_model":
                best_model_name,

            "model_comparison":
                model_comparison,

            "validation_mae":
                validation_mae,

            "validation_rmse":
                validation_rmse,

            "validation_r2":
                validation_r2,

            "current_price":
                current_price,

            "ending_price":
                ending_price,

            "price_change":
                price_change,

            "price_change_percent":
                price_change_percent,

            "average_forecast_price":
                average_forecast_price,

            "maximum_forecast_price":
                maximum_forecast_price,

            "minimum_forecast_price":
                minimum_forecast_price,

            "forecast_times":
                forecast_times,

            "forecast_prices":
                forecast_prices,

            "forecast_hours":
                len(
                    forecast_prices
                ),

            "history_times":
                history_chart[
                    "TimeDK"
                ].to_numpy(),

            "history_prices":
                history_chart[
                    "DayAheadPriceEUR"
                ].to_numpy(),

            "feature_importance":
                feature_importance,

            "weather_location":
                (
                    self.weather_api
                    .locations[
                        price_area
                    ][
                        "name"
                    ]
                ),

            "training_observations":
                len(
                    model_data
                ),

            "n_splits":
                n_splits
        }

    # ==================================================
    # Feature importance
    # ==================================================

    def _feature_importance_from_model(
        self,
        model,
        feature_columns
    ):

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            importance = np.abs(
                model.coef_
            )

            total = (
                importance.sum()
            )

            if (
                total
                > 0
            ):

                importance = (
                    importance
                    / total
                )

        else:

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance"
                ]
            )

        return (
            pd.DataFrame(
                {
                    "Feature":
                        feature_columns,

                    "Importance":
                        importance
                }
            )
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # Helpers
    # ==================================================

    def _safe_splits(
        self,
        observations,
        requested_splits
    ):

        maximum_splits = (
            observations
            - 1
        )

        return min(
            requested_splits,
            maximum_splits
        )

    def _calculate_improvement(
        self,
        original_mae,
        new_mae
    ):

        if (
            original_mae
            <= 0
        ):

            return 0

        return (
            (
                original_mae
                - new_mae
            )
            / original_mae
        ) * 100

    def _get_feature_set_metric(
        self,
        comparison_table,
        feature_set,
        metric
    ):

        row = (
            comparison_table[
                comparison_table[
                    "FeatureSet"
                ]
                == feature_set
            ]
        )

        if row.empty:

            return 0

        return (
            row.iloc[
                0
            ][
                metric
            ]
        )

    # ==================================================
    # Empty results
    # ==================================================

    def _empty_comparison_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "comparison_table":
                pd.DataFrame(),

            "best_model":
                "None",

            "current_price":
                0,

            "next_price_prediction":
                0,

            "predicted_change":
                0,

            "predicted_change_percent":
                0,

            "actual_prices":
                np.array([]),

            "best_predictions":
                np.array([]),

            "prediction_times":
                np.array([]),

            "feature_importance":
                pd.DataFrame(),

            "fold_results":
                pd.DataFrame(),

            "total_observations":
                0,

            "validation_observations":
                0,

            "n_splits":
                0
        }

    def _empty_ablation_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "model_name":
                "None",

            "comparison_table":
                pd.DataFrame(),

            "best_feature_set":
                "None",

            "market_improvement":
                0,

            "weather_improvement_vs_price":
                0,

            "weather_incremental_improvement":
                0,

            "actual_prices":
                np.array([]),

            "prediction_times":
                np.array([]),

            "best_predictions":
                np.array([]),

            "prediction_store":
                {},

            "feature_importance":
                pd.DataFrame(),

            "current_price":
                0,

            "next_price_prediction":
                0,

            "predicted_change":
                0,

            "predicted_change_percent":
                0,

            "total_observations":
                0,

            "validation_observations":
                0,

            "n_splits":
                0
        }

    def _empty_forward_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "best_model":
                "None",

            "model_comparison":
                pd.DataFrame(),

            "validation_mae":
                0,

            "validation_rmse":
                0,

            "validation_r2":
                0,

            "current_price":
                0,

            "ending_price":
                0,

            "price_change":
                0,

            "price_change_percent":
                0,

            "average_forecast_price":
                0,

            "maximum_forecast_price":
                0,

            "minimum_forecast_price":
                0,

            "forecast_times":
                np.array([]),

            "forecast_prices":
                np.array([]),

            "forecast_hours":
                0,

            "history_times":
                np.array([]),

            "history_prices":
                np.array([]),

            "feature_importance":
                pd.DataFrame(),

            "weather_location":
                "",

            "training_observations":
                0,

            "n_splits":
                0
        }