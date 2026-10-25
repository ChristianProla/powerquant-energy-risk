import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit


class MachineLearningForecaster:

    def __init__(self):

        self.feature_columns = [
            "DayAheadPriceEUR",
            "WindGenerationMW",
            "ElectricityDemandMW",
            "WindDemandRatio",
            "HourOfDay",
            "DayOfWeek",
            "Month",
            "IsWeekend",
            "PriceLag1",
            "PriceLag2",
            "PriceLag24",
            "WindLag1",
            "DemandLag1",
            "PriceMovingAverage6",
            "PriceMovingAverage24",
            "PriceVolatility24"
        ]

    # ==================================================
    # Prepare model data
    # ==================================================

    def _prepare_model_data(
        self,
        feature_dataframe
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
            for column in self.feature_columns
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
        # Forecast target
        #
        # Features at time t
        # predict price at time t + 1
        # --------------------------------------------------

        data["NextPrice"] = (
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
                    self.feature_columns
                    + ["NextPrice"]
                )
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(model_data) < 40:

            return None, (
                "Not enough observations for "
                "machine-learning validation. "
                "Try using 1 Month or 1 Year."
            )

        return (
            model_data,
            None
        )

    # ==================================================
    # Model factory
    # ==================================================

    def _create_models(
        self,
        random_state=42
    ):

        return {
            "Linear Regression":
                LinearRegression(),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state
                )
        }

    # ==================================================
    # Model comparison with time-series validation
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
            feature_dataframe
        )

        if error_message is not None:

            return self._empty_comparison_result(
                error_message
            )

        # --------------------------------------------------
        # Make sure number of folds is realistic
        # --------------------------------------------------

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

        X = model_data[
            self.feature_columns
        ]

        y = model_data[
            "NextPrice"
        ]

        times = model_data[
            "TimeDK"
        ]

        time_series_split = (
            TimeSeriesSplit(
                n_splits=n_splits
            )
        )

        models = (
            self._create_models(
                random_state=random_state
            )
        )

        # --------------------------------------------------
        # Containers for predictions
        # --------------------------------------------------

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
            time_series_split.split(X),
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

            # ----------------------------------------------
            # Naive baseline
            #
            # Predict:
            # next price = current price
            # ----------------------------------------------

            baseline_prediction = (
                X_test[
                    "DayAheadPriceEUR"
                ].to_numpy()
            )

            model_predictions[
                "Naive Baseline"
            ].extend(
                baseline_prediction
            )

            # ----------------------------------------------
            # Machine-learning models
            # ----------------------------------------------

            fold_model_scores = {}

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

                fold_mae = (
                    mean_absolute_error(
                        y_test,
                        predictions
                    )
                )

                fold_model_scores[
                    model_name
                ] = fold_mae

            # ----------------------------------------------
            # Baseline fold score
            # ----------------------------------------------

            baseline_fold_mae = (
                mean_absolute_error(
                    y_test,
                    baseline_prediction
                )
            )

            fold_results.append(
                {
                    "fold":
                        fold_number,

                    "train_size":
                        len(train_index),

                    "test_size":
                        len(test_index),

                    "baseline_mae":
                        baseline_fold_mae,

                    "linear_regression_mae":
                        fold_model_scores[
                            "Linear Regression"
                        ],

                    "random_forest_mae":
                        fold_model_scores[
                            "Random Forest"
                        ],

                    "gradient_boosting_mae":
                        fold_model_scores[
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
        # Overall model metrics
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

        # --------------------------------------------------
        # Improvement vs baseline
        # --------------------------------------------------

        improvements = []

        for _, row in (
            comparison_table.iterrows()
        ):

            if baseline_mae > 0:

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
            comparison_table.iloc[0][
                "Model"
            ]
        )

        # ==================================================
        # Train final models on all available observations
        # ==================================================

        final_models = (
            self._create_models(
                random_state=random_state
            )
        )

        fitted_models = {}

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

        # --------------------------------------------------
        # Latest row for next-price prediction
        # --------------------------------------------------

        prediction_data = (
            feature_dataframe.copy()
        )

        prediction_data = (
            prediction_data.replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        prediction_data = (
            prediction_data.dropna(
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
        # Feature importance for best ML model
        # ==================================================

        feature_importance = (
            pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance"
                ]
            )
        )

        if (
            best_model_name
            in fitted_models
        ):

            best_model = (
                fitted_models[
                    best_model_name
                ]
            )

            if hasattr(
                best_model,
                "feature_importances_"
            ):

                feature_importance = (
                    pd.DataFrame(
                        {
                            "Feature":
                                self.feature_columns,

                            "Importance":
                                best_model
                                .feature_importances_
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

            elif hasattr(
                best_model,
                "coef_"
            ):

                coefficients = (
                    np.abs(
                        best_model.coef_
                    )
                )

                coefficient_sum = (
                    coefficients.sum()
                )

                if coefficient_sum > 0:

                    coefficients = (
                        coefficients
                        / coefficient_sum
                    )

                feature_importance = (
                    pd.DataFrame(
                        {
                            "Feature":
                                self.feature_columns,

                            "Importance":
                                coefficients
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
        # Best-model validation predictions
        # ==================================================

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
                len(model_data),

            "validation_observations":
                len(actual_values),

            "n_splits":
                n_splits
        }

    # ==================================================
    # Existing Random Forest method
    #
    # Kept so older PowerQuant code still works.
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
            feature_dataframe
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
            len(model_data)
            * (
                1
                - test_size
            )
        )

        split_index = max(
            1,
            min(
                split_index,
                len(model_data) - 1
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

        test_times = model_data[
            "TimeDK"
        ].iloc[
            split_index:
        ]

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

        if len(y_test) >= 2:

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
            pd.DataFrame(
                {
                    "Feature":
                        self.feature_columns,

                    "Importance":
                        model.feature_importances_
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

            latest_features = (
                latest_row[
                    self.feature_columns
                ]
            )

            next_price_prediction = (
                model.predict(
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
                len(X_train),

            "testing_observations":
                len(X_test),

            "total_observations":
                len(model_data)
        }

    # ==================================================
    # Empty comparison result
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

    # ==================================================
    # Empty Random Forest result
    # ==================================================

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
                pd.DataFrame(
                    columns=[
                        "Feature",
                        "Importance"
                    ]
                ),

            "training_observations":
                0,

            "testing_observations":
                0,

            "total_observations":
                0
        }