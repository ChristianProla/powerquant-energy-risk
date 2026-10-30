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

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    StandardScaler
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
        # Future realized electricity demand and generation
        # are deliberately excluded.
        #
        # Forward forecasting instead uses:
        #
        # - current / historical prices
        # - deterministic target calendar
        # - weather forecast for target hour
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

        self.valid_models = [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
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

            # --------------------------------------------------
            # Linear coefficients are only meaningfully
            # comparable when predictors are standardized.
            # --------------------------------------------------

            return Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "regressor",
                        LinearRegression()
                    )
                ]
            )

        if (
            model_name
            == "Random Forest"
        ):

            return RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )

        if (
            model_name
            == "Gradient Boosting"
        ):

            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            )

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    def _create_models(
        self,
        random_state=42
    ):

        return {
            model_name:
                self._create_model(
                    model_name,
                    random_state
                )
            for model_name in self.valid_models
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

        required_columns = (
            list(
                feature_columns
            )
            + [
                "DayAheadPriceEUR"
            ]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            return None, (
                "Missing required features: "
                + ", ".join(
                    sorted(
                        set(
                            missing_columns
                        )
                    )
                )
            )

        if (
            "TimeDK"
            in data.columns
        ):

            data[
                "TimeDK"
            ] = pd.to_datetime(
                data[
                    "TimeDK"
                ],
                errors="coerce"
            )

            data = (
                data.sort_values(
                    "TimeDK"
                )
                .reset_index(
                    drop=True
                )
            )

        # --------------------------------------------------
        # Feature row at time t predicts electricity price
        # at t + 1.
        # --------------------------------------------------

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
                    list(
                        feature_columns
                    )
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
    # Latest available prediction row
    #
    # Unlike model_data, this dataframe does NOT require
    # NextPrice to exist.
    #
    # It therefore allows us to predict the genuinely
    # unseen next price from the latest known feature row.
    # ==================================================

    def _get_latest_feature_row(
        self,
        feature_dataframe,
        feature_columns
    ):

        if feature_dataframe.empty:

            return None

        missing_columns = [
            column
            for column in feature_columns
            if column not in feature_dataframe.columns
        ]

        if missing_columns:

            return None

        data = (
            feature_dataframe.copy()
        )

        if (
            "TimeDK"
            in data.columns
        ):

            data[
                "TimeDK"
            ] = pd.to_datetime(
                data[
                    "TimeDK"
                ],
                errors="coerce"
            )

            data = (
                data.dropna(
                    subset=[
                        "TimeDK"
                    ]
                )
                .sort_values(
                    "TimeDK"
                )
            )

        data = data.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        complete = (
            data.dropna(
                subset=list(
                    feature_columns
                )
            )
        )

        if complete.empty:

            return None

        return (
            complete.iloc[
                [-1]
            ].copy()
        )

    # ==================================================
    # Time-series splitter
    #
    # gap=1 creates a stricter separation between train
    # and test samples for one-step-ahead forecasting.
    # ==================================================

    def _create_splitter(
        self,
        n_splits
    ):

        return TimeSeriesSplit(
            n_splits=n_splits,
            gap=1
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
            self._create_splitter(
                n_splits
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
        # Time-series cross-validation
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

            # --------------------------------------------------
            # Naive one-hour forecast:
            #
            # price(t + 1) = price(t)
            # --------------------------------------------------

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

                    "train_end_time":
                        times.iloc[
                            train_index[
                                -1
                            ]
                        ],

                    "test_start_time":
                        times.iloc[
                            test_index[
                                0
                            ]
                        ],

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

        comparison_table[
            "ImprovementVsBaseline"
        ] = (
            comparison_table[
                "MAE"
            ].apply(
                lambda value:
                    self._calculate_improvement(
                        baseline_mae,
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

        best_model_name = (
            comparison_table.iloc[
                0
            ][
                "Model"
            ]
        )

        # ==================================================
        # Fit final models on all historical training pairs
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

        # ==================================================
        # Genuine next unseen price prediction
        # ==================================================

        latest_row = (
            self._get_latest_feature_row(
                feature_dataframe,
                self.feature_columns
            )
        )

        if latest_row is None:

            return (
                self._empty_comparison_result(
                    (
                        "No complete latest feature row "
                        "was available for prediction."
                    )
                )
            )

        current_price = float(
            latest_row[
                "DayAheadPriceEUR"
            ].iloc[
                0
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

            next_price_prediction = float(
                best_model.predict(
                    latest_row[
                        self.feature_columns
                    ]
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
                n_splits,

            "methodology":
                {
                    "validation":
                        "TimeSeriesSplit with one-observation gap",

                    "forecast_horizon":
                        "One hour ahead",

                    "linear_regression":
                        "Standardized predictors",

                    "baseline":
                        "Next price equals current price"
                }
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

        if (
            model_name
            not in self.valid_models
        ):

            return (
                self._empty_ablation_result(
                    f"Unknown model: {model_name}"
                )
            )

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
            self._create_splitter(
                n_splits
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

        # ==================================================
        # Genuine next unseen price prediction
        # ==================================================

        latest_row = (
            self._get_latest_feature_row(
                feature_dataframe,
                best_features
            )
        )

        if latest_row is None:

            return (
                self._empty_ablation_result(
                    (
                        "No complete latest feature row "
                        "was available for prediction."
                    )
                )
            )

        current_price = float(
            latest_row[
                "DayAheadPriceEUR"
            ].iloc[
                0
            ]
        )

        next_price_prediction = float(
            final_model.predict(
                latest_row[
                    best_features
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
                n_splits,

            "methodology":
                {
                    "validation":
                        "Identical TimeSeriesSplit folds with one-observation gap",

                    "row_comparison":
                        (
                            "All feature sets use the same complete "
                            "historical observations"
                        ),

                    "linear_regression":
                        "Standardized predictors"
                }
        }

    # ==================================================
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

        try:

            horizon = int(
                horizon
            )

        except (
            TypeError,
            ValueError
        ):

            return (
                self._empty_forward_result(
                    "Forecast horizon must be numeric."
                )
            )

        horizon = max(
            1,
            min(
                horizon,
                24
            )
        )

        # ==================================================
        # Historical validation
        #
        # Target-hour historical weather uses realized
        # next-hour weather as a proxy for forecast weather.
        #
        # Therefore these validation metrics may be
        # optimistic relative to live deployment.
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
                    (
                        "Not enough observations for "
                        "forward-model validation."
                    )
                )
            )

        splitter = (
            self._create_splitter(
                n_splits
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

        validation_actual = np.array(
            validation_actual
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

        validation_mae = float(
            model_comparison.iloc[
                0
            ][
                "MAE"
            ]
        )

        validation_rmse = float(
            model_comparison.iloc[
                0
            ][
                "RMSE"
            ]
        )

        validation_r2 = float(
            model_comparison.iloc[
                0
            ][
                "R2"
            ]
        )

        # ==================================================
        # Train best model on all historical observations
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
        # Historical prices for recursive forecast
        # ==================================================

        if (
            "TimeDK"
            not in feature_dataframe.columns
            or
            "DayAheadPriceEUR"
            not in feature_dataframe.columns
        ):

            return (
                self._empty_forward_result(
                    "Historical price data is incomplete."
                )
            )

        historical_data = (
            feature_dataframe[
                [
                    "TimeDK",
                    "DayAheadPriceEUR"
                ]
            ]
            .copy()
        )

        historical_data[
            "TimeDK"
        ] = pd.to_datetime(
            historical_data[
                "TimeDK"
            ],
            errors="coerce"
        )

        historical_data[
            "DayAheadPriceEUR"
        ] = pd.to_numeric(
            historical_data[
                "DayAheadPriceEUR"
            ],
            errors="coerce"
        )

        historical_data = (
            historical_data
            .dropna()
            .sort_values(
                "TimeDK"
            )
            .drop_duplicates(
                subset=[
                    "TimeDK"
                ],
                keep="last"
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

        # --------------------------------------------------
        # We need t plus 24 earlier observations to obtain
        # a true PriceLag24 value.
        # --------------------------------------------------

        if (
            len(
                price_history
            )
            < 25
        ):

            return (
                self._empty_forward_result(
                    (
                        "At least 25 hourly price observations "
                        "are required for a forward forecast."
                    )
                )
            )

        current_price = float(
            price_history[
                -1
            ]
        )

        latest_historical_time = (
            historical_data[
                "TimeDK"
            ].iloc[
                -1
            ]
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

        required_weather_columns = [
            "TimeDK",
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM"
        ]

        missing_weather_columns = [
            column
            for column in required_weather_columns
            if column not in future_weather.columns
        ]

        if missing_weather_columns:

            return (
                self._empty_forward_result(
                    (
                        "Future weather forecast is missing: "
                        + ", ".join(
                            missing_weather_columns
                        )
                    )
                )
            )

        future_weather = (
            future_weather.copy()
        )

        future_weather[
            "TimeDK"
        ] = pd.to_datetime(
            future_weather[
                "TimeDK"
            ],
            errors="coerce"
        )

        for column in (
            required_weather_columns[
                1:
            ]
        ):

            future_weather[
                column
            ] = pd.to_numeric(
                future_weather[
                    column
                ],
                errors="coerce"
            )

        future_weather = (
            future_weather
            .dropna(
                subset=[
                    "TimeDK"
                ]
            )
            .sort_values(
                "TimeDK"
            )
            .drop_duplicates(
                subset=[
                    "TimeDK"
                ],
                keep="last"
            )
        )

        # --------------------------------------------------
        # Weather observations must actually lie after the
        # latest historical electricity-price observation.
        # --------------------------------------------------

        future_weather = (
            future_weather[
                future_weather[
                    "TimeDK"
                ]
                >
                latest_historical_time
            ]
            .head(
                horizon
            )
            .reset_index(
                drop=True
            )
        )

        if future_weather.empty:

            return (
                self._empty_forward_result(
                    (
                        "No future weather observations occur "
                        "after the latest historical price."
                    )
                )
            )

        if (
            future_weather[
                required_weather_columns[
                    1:
                ]
            ]
            .isna()
            .any()
            .any()
        ):

            return (
                self._empty_forward_result(
                    (
                        "The forward forecast could not be generated "
                        "because required future weather values "
                        "are missing."
                    )
                )
            )

        # ==================================================
        # Recursive multi-hour forecast
        # ==================================================

        forecast_times = []
        forecast_prices = []

        for _, weather_row in (
            future_weather.iterrows()
        ):

            target_time = pd.Timestamp(
                weather_row[
                    "TimeDK"
                ]
            )

            # --------------------------------------------------
            # For a training row at time t:
            #
            # DayAheadPriceEUR = price(t)
            # PriceLag1        = price(t-1)
            # PriceLag2        = price(t-2)
            # PriceLag24       = price(t-24)
            #
            # price_history[-1] is therefore t.
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Important:
            #
            # -24 would be t-23 because -1 is t.
            # The correct t-24 observation is -25.
            # --------------------------------------------------

            price_lag_24 = (
                price_history[
                    -25
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

            moving_average_6 = float(
                recent_6.mean()
            )

            moving_average_24 = float(
                recent_24.mean()
            )

            volatility_24 = float(
                recent_24.std(
                    ddof=1
                )
            )

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
                            float(
                                weather_row[
                                    "TemperatureC"
                                ]
                            ),

                        "TargetWeatherWindSpeedMS":
                            float(
                                weather_row[
                                    "WeatherWindSpeedMS"
                                ]
                            ),

                        "TargetCloudCoverPct":
                            float(
                                weather_row[
                                    "CloudCoverPct"
                                ]
                            ),

                        "TargetPrecipitationMM":
                            float(
                                weather_row[
                                    "PrecipitationMM"
                                ]
                            ),

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

                return (
                    self._empty_forward_result(
                        (
                            "The forecast could not be generated "
                            "because a recursive model input "
                            "contained missing values."
                        )
                    )
                )

            predicted_price = float(
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

            # --------------------------------------------------
            # Recursive forecast:
            #
            # prediction becomes the current price for the
            # following forecast step.
            # --------------------------------------------------

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
                    "The forward forecast could not be generated."
                )
            )

        forecast_prices = np.array(
            forecast_prices
        )

        forecast_times = np.array(
            forecast_times
        )

        ending_price = float(
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

        average_forecast_price = float(
            forecast_prices.mean()
        )

        maximum_forecast_price = float(
            forecast_prices.max()
        )

        minimum_forecast_price = float(
            forecast_prices.min()
        )

        history_chart = (
            historical_data.tail(
                48
            )
        )

        weather_location = ""

        if (
            hasattr(
                self.weather_api,
                "locations"
            )
            and
            price_area
            in self.weather_api.locations
        ):

            weather_location = (
                self.weather_api.locations[
                    price_area
                ].get(
                    "name",
                    ""
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
                weather_location,

            "training_observations":
                len(
                    model_data
                ),

            "n_splits":
                n_splits,

            "methodology":
                {
                    "validation":
                        "TimeSeriesSplit with one-observation gap",

                    "historical_weather":
                        (
                            "Realized next-hour weather is used as "
                            "a historical proxy for forecast weather"
                        ),

                    "historical_validation_caveat":
                        (
                            "Validation may be optimistic because "
                            "archived historical weather forecasts "
                            "are not currently used"
                        ),

                    "live_weather":
                        "Actual forward weather forecast",

                    "forecast_method":
                        "Recursive one-hour-ahead prediction",

                    "future_market_observations":
                        (
                            "Future realized electricity demand and "
                            "generation are not used"
                        )
                }
        }

    # ==================================================
    # Feature importance
    # ==================================================

    def _feature_importance_from_model(
        self,
        model,
        feature_columns
    ):

        # --------------------------------------------------
        # Tree models
        # --------------------------------------------------

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = np.array(
                model.feature_importances_,
                dtype=float
            )

        # --------------------------------------------------
        # Standardized Linear Regression pipeline
        # --------------------------------------------------

        elif (
            isinstance(
                model,
                Pipeline
            )
            and
            "regressor"
            in model.named_steps
            and
            hasattr(
                model.named_steps[
                    "regressor"
                ],
                "coef_"
            )
        ):

            importance = np.abs(
                np.asarray(
                    model.named_steps[
                        "regressor"
                    ].coef_,
                    dtype=float
                )
            )

            total = (
                importance.sum()
            )

            if total > 0:

                importance = (
                    importance
                    / total
                )

        elif hasattr(
            model,
            "coef_"
        ):

            importance = np.abs(
                np.asarray(
                    model.coef_,
                    dtype=float
                )
            )

            total = (
                importance.sum()
            )

            if total > 0:

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

        try:

            requested_splits = int(
                requested_splits
            )

        except (
            TypeError,
            ValueError
        ):

            return 0

        if requested_splits < 2:

            return requested_splits

        # --------------------------------------------------
        # TimeSeriesSplit with gap=1 needs slightly more
        # room than a splitter without a gap.
        # --------------------------------------------------

        maximum_splits = max(
            0,
            observations - 2
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

        return float(
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
                0,

            "methodology":
                {}
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
                0,

            "methodology":
                {}
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
                0,

            "methodology":
                {}
        }