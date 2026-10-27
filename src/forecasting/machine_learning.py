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


class MachineLearningForecaster:

    def __init__(self):

        # ==================================================
        # Feature groups
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
    # Prepare model data
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

        # --------------------------------------------------
        # Target:
        #
        # Features at time t
        # predict electricity price at time t + 1
        # --------------------------------------------------

        data[
            "NextPrice"
        ] = (
            data[
                "DayAheadPriceEUR"
            ].shift(-1)
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

        if len(
            model_data
        ) < 40:

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
    # Create one model
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

            return LinearRegression()

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

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

    # ==================================================
    # Create all models
    # ==================================================

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

        if error_message is not None:

            return self._empty_comparison_result(
                error_message
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

        maximum_splits = (
            len(model_data)
            - 1
        )

        n_splits = min(
            n_splits,
            maximum_splits
        )

        if n_splits < 2:

            return self._empty_comparison_result(
                "Not enough data for time-series validation."
            )

        time_series_split = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        model_predictions = {
            "Naive Baseline": [],
            "Linear Regression": [],
            "Random Forest": [],
            "Gradient Boosting": []
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
            time_series_split.split(
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
            # Naive baseline
            # --------------------------------------------------

            baseline_predictions = (
                X_test[
                    "DayAheadPriceEUR"
                ].to_numpy()
            )

            model_predictions[
                "Naive Baseline"
            ].extend(
                baseline_predictions
            )

            fold_scores = {}

            # --------------------------------------------------
            # ML models
            # --------------------------------------------------

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

                model_predictions[
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

        # ==================================================
        # Overall metrics
        # ==================================================

        actual_values = np.array(
            actual_values
        )

        prediction_times = np.array(
            prediction_times
        )

        comparison_rows = []

        baseline_mae = None

        for (
            model_name,
            predictions
        ) in model_predictions.items():

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

            if len(
                actual_values
            ) >= 2:

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

            if (
                baseline_mae is not None
                and
                baseline_mae > 0
            ):

                improvement = (
                    (
                        baseline_mae
                        - row["MAE"]
                    )
                    / baseline_mae
                ) * 100

            else:

                improvement = 0

            improvements.append(
                improvement
            )

        comparison_table[
            "ImprovementVsBaseline"
        ] = improvements

        comparison_table = (
            comparison_table.sort_values(
                by="MAE",
                ascending=True
            )
            .reset_index(
                drop=True
            )
        )

        best_model_name = (
            comparison_table.iloc[
                0
            ]["Model"]
        )

        # ==================================================
        # Final model training
        # ==================================================

        fitted_models = {}

        final_models = (
            self._create_models(
                random_state
            )
        )

        for (
            model_name,
            model
        ) in final_models.items():

            model.fit(
                X,
                y
            )

            fitted_models[
                model_name
            ] = model

        # ==================================================
        # Next prediction
        # ==================================================

        prediction_data = (
            feature_dataframe
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna(
                subset=self.feature_columns
            )
        )

        current_price = 0
        next_price_prediction = 0
        predicted_change = 0
        predicted_change_percent = 0

        if not prediction_data.empty:

            latest_row = (
                prediction_data.iloc[
                    [-1]
                ]
            )

            current_price = (
                latest_row[
                    "DayAheadPriceEUR"
                ].iloc[0]
            )

            latest_features = (
                latest_row[
                    self.feature_columns
                ]
            )

            if (
                best_model_name
                == "Naive Baseline"
            ):

                next_price_prediction = (
                    current_price
                )

            else:

                best_model = (
                    fitted_models[
                        best_model_name
                    ]
                )

                next_price_prediction = (
                    best_model.predict(
                        latest_features
                    )[0]
                )

            predicted_change = (
                next_price_prediction
                - current_price
            )

            if current_price != 0:

                predicted_change_percent = (
                    predicted_change
                    / abs(
                        current_price
                    )
                ) * 100

        # ==================================================
        # Feature importance
        # ==================================================

        feature_importance = (
            self._calculate_feature_importance(
                fitted_models,
                best_model_name,
                self.feature_columns
            )
        )

        best_predictions = np.array(
            model_predictions[
                best_model_name
            ]
        )

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
                best_predictions,

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
    # Feature ablation testing
    # ==================================================

    def compare_feature_sets(
        self,
        feature_dataframe,
        model_name="Random Forest",
        n_splits=5,
        random_state=42
    ):

        if feature_dataframe.empty:

            return self._empty_ablation_result(
                "No feature data available."
            )

        # --------------------------------------------------
        # To make the comparison fair, use only rows that
        # are complete for the FULL feature set.
        #
        # Otherwise Price History might get more observations
        # than the Weather model.
        # --------------------------------------------------

        (
            full_model_data,
            error_message
        ) = self._prepare_model_data(
            feature_dataframe,
            self.feature_columns
        )

        if error_message is not None:

            return self._empty_ablation_result(
                error_message
            )

        maximum_splits = (
            len(full_model_data)
            - 1
        )

        n_splits = min(
            n_splits,
            maximum_splits
        )

        if n_splits < 2:

            return self._empty_ablation_result(
                "Not enough data for feature ablation testing."
            )

        y = full_model_data[
            "NextPrice"
        ]

        times = full_model_data[
            "TimeDK"
        ]

        time_series_split = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        # --------------------------------------------------
        # Create the exact same folds once.
        # Every feature set will use these folds.
        # --------------------------------------------------

        folds = list(
            time_series_split.split(
                full_model_data
            )
        )

        results = []

        prediction_store = {}

        actual_values = None

        prediction_times = None

        # ==================================================
        # Test each feature set
        # ==================================================

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

                test_times = (
                    times.iloc[
                        test_index
                    ]
                )

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
                    test_times.to_numpy()
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

            if len(
                all_actual
            ) >= 2:

                r2 = (
                    r2_score(
                        all_actual,
                        all_predictions
                    )
                )

            else:

                r2 = 0

            fold_mae_mean = np.mean(
                fold_mae_values
            )

            fold_mae_std = np.std(
                fold_mae_values
            )

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
                        fold_mae_mean,

                    "FoldMAEStd":
                        fold_mae_std
                }
            )

            prediction_store[
                feature_set_name
            ] = all_predictions

            if actual_values is None:

                actual_values = (
                    all_actual
                )

                prediction_times = (
                    all_times
                )

        # ==================================================
        # Build comparison table
        # ==================================================

        comparison_table = (
            pd.DataFrame(
                results
            )
        )

        price_only_row = (
            comparison_table[
                comparison_table[
                    "FeatureSet"
                ]
                == "Price History"
            ]
        )

        if price_only_row.empty:

            price_only_mae = 0

        else:

            price_only_mae = (
                price_only_row.iloc[
                    0
                ]["MAE"]
            )

        improvement_values = []

        for _, row in (
            comparison_table.iterrows()
        ):

            if price_only_mae > 0:

                improvement = (
                    (
                        price_only_mae
                        - row["MAE"]
                    )
                    / price_only_mae
                ) * 100

            else:

                improvement = 0

            improvement_values.append(
                improvement
            )

        comparison_table[
            "ImprovementVsPriceOnly"
        ] = improvement_values

        comparison_table = (
            comparison_table.sort_values(
                by="MAE",
                ascending=True
            )
            .reset_index(
                drop=True
            )
        )

        best_feature_set = (
            comparison_table.iloc[
                0
            ]["FeatureSet"]
        )

        # ==================================================
        # Incremental market and weather value
        # ==================================================

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

        market_improvement = (
            self._calculate_improvement(
                price_mae,
                market_mae
            )
        )

        weather_improvement_vs_price = (
            self._calculate_improvement(
                price_mae,
                weather_mae
            )
        )

        weather_incremental_improvement = (
            self._calculate_improvement(
                market_mae,
                weather_mae
            )
        )

        # ==================================================
        # Final best feature-set model
        # ==================================================

        best_feature_columns = (
            self.feature_sets[
                best_feature_set
            ]
        )

        X_best = full_model_data[
            best_feature_columns
        ]

        final_model = (
            self._create_model(
                model_name,
                random_state
            )
        )

        final_model.fit(
            X_best,
            y
        )

        feature_importance = (
            self._feature_importance_from_model(
                final_model,
                best_feature_columns
            )
        )

        # ==================================================
        # Next-price prediction
        # ==================================================

        latest_data = (
            feature_dataframe
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna(
                subset=best_feature_columns
            )
        )

        current_price = 0

        next_price_prediction = 0

        predicted_change = 0

        predicted_change_percent = 0

        if not latest_data.empty:

            latest_row = (
                latest_data.iloc[
                    [-1]
                ]
            )

            current_price = (
                latest_row[
                    "DayAheadPriceEUR"
                ].iloc[0]
            )

            next_price_prediction = (
                final_model.predict(
                    latest_row[
                        best_feature_columns
                    ]
                )[0]
            )

            predicted_change = (
                next_price_prediction
                - current_price
            )

            if current_price != 0:

                predicted_change_percent = (
                    predicted_change
                    / abs(
                        current_price
                    )
                ) * 100

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
                market_improvement,

            "weather_improvement_vs_price":
                weather_improvement_vs_price,

            "weather_incremental_improvement":
                weather_incremental_improvement,

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
    # Feature importance helpers
    # ==================================================

    def _calculate_feature_importance(
        self,
        fitted_models,
        best_model_name,
        feature_columns
    ):

        if (
            best_model_name
            not in fitted_models
        ):

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance"
                ]
            )

        return (
            self._feature_importance_from_model(
                fitted_models[
                    best_model_name
                ],
                feature_columns
            )
        )

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
    # Ablation helpers
    # ==================================================

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

        return row.iloc[
            0
        ][metric]

    def _calculate_improvement(
        self,
        original_mae,
        new_mae
    ):

        if original_mae <= 0:

            return 0

        return (
            (
                original_mae
                - new_mae
            )
            / original_mae
        ) * 100

    # ==================================================
    # Random Forest compatibility
    # ==================================================

    def train_random_forest(
        self,
        feature_dataframe,
        test_size=0.20,
        n_estimators=300,
        random_state=42
    ):

        (
            model_data,
            error_message
        ) = self._prepare_model_data(
            feature_dataframe,
            self.feature_columns
        )

        if error_message is not None:

            return self._empty_random_forest_result(
                error_message
            )

        X = model_data[
            self.feature_columns
        ]

        y = model_data[
            "NextPrice"
        ]

        split_index = int(
            len(
                model_data
            )
            * (
                1
                - test_size
            )
        )

        split_index = max(
            1,
            min(
                split_index,
                len(
                    model_data
                ) - 1
            )
        )

        X_train = X.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_train = y.iloc[
            :split_index
        ]

        y_test = y.iloc[
            split_index:
        ]

        test_times = (
            model_data[
                "TimeDK"
            ].iloc[
                split_index:
            ]
        )

        model = (
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=12,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
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

        mae = (
            mean_absolute_error(
                y_test,
                predictions
            )
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        if len(
            y_test
        ) >= 2:

            r2 = (
                r2_score(
                    y_test,
                    predictions
                )
            )

        else:

            r2 = 0

        baseline_predictions = (
            X_test[
                "DayAheadPriceEUR"
            ].to_numpy()
        )

        baseline_mae = (
            mean_absolute_error(
                y_test,
                baseline_predictions
            )
        )

        if baseline_mae > 0:

            improvement = (
                (
                    baseline_mae
                    - mae
                )
                / baseline_mae
            ) * 100

        else:

            improvement = 0

        feature_importance = (
            self._feature_importance_from_model(
                model,
                self.feature_columns
            )
        )

        current_price = 0

        next_price_prediction = 0

        predicted_change = 0

        predicted_change_percent = 0

        prediction_data = (
            feature_dataframe
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna(
                subset=self.feature_columns
            )
        )

        if not prediction_data.empty:

            latest_row = (
                prediction_data.iloc[
                    [-1]
                ]
            )

            current_price = (
                latest_row[
                    "DayAheadPriceEUR"
                ].iloc[0]
            )

            next_price_prediction = (
                model.predict(
                    latest_row[
                        self.feature_columns
                    ]
                )[0]
            )

            predicted_change = (
                next_price_prediction
                - current_price
            )

            if current_price != 0:

                predicted_change_percent = (
                    predicted_change
                    / abs(
                        current_price
                    )
                ) * 100

        return {
            "success":
                True,

            "message":
                "Random Forest model trained successfully.",

            "model":
                model,

            "mae":
                mae,

            "rmse":
                rmse,

            "r2":
                r2,

            "baseline_mae":
                baseline_mae,

            "improvement_vs_baseline":
                improvement,

            "current_price":
                current_price,

            "next_price_prediction":
                next_price_prediction,

            "predicted_change":
                predicted_change,

            "predicted_change_percent":
                predicted_change_percent,

            "actual_prices":
                y_test.to_numpy(),

            "predicted_prices":
                predictions,

            "test_times":
                test_times.to_numpy(),

            "feature_importance":
                feature_importance,

            "training_observations":
                len(
                    X_train
                ),

            "testing_observations":
                len(
                    X_test
                ),

            "total_observations":
                len(
                    model_data
                )
        }

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

    def _empty_random_forest_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "model":
                None,

            "mae":
                0,

            "rmse":
                0,

            "r2":
                0,

            "baseline_mae":
                0,

            "improvement_vs_baseline":
                0,

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

            "predicted_prices":
                np.array([]),

            "test_times":
                np.array([]),

            "feature_importance":
                pd.DataFrame(),

            "training_observations":
                0,

            "testing_observations":
                0,

            "total_observations":
                0
        }