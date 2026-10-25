import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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

    # --------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------

    def train_random_forest(
        self,
        feature_dataframe,
        test_size=0.20,
        n_estimators=300,
        random_state=42
    ):

        if feature_dataframe.empty:

            return self._empty_result(
                "No feature data available."
            )

        data = feature_dataframe.copy()

        # --------------------------------------------------
        # Check required columns
        # --------------------------------------------------

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in data.columns
        ]

        if missing_columns:

            return self._empty_result(
                (
                    "Missing required features: "
                    + ", ".join(
                        missing_columns
                    )
                )
            )

        # --------------------------------------------------
        # Target
        #
        # Current row:
        # features at time t
        #
        # Target:
        # electricity price at time t + 1
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

        # --------------------------------------------------
        # Dataset used for model training
        # --------------------------------------------------

        model_data = data.dropna(
            subset=(
                self.feature_columns
                + ["NextPrice"]
            )
        ).copy()

        minimum_observations = 30

        if len(
            model_data
        ) < minimum_observations:

            return self._empty_result(
                (
                    "Not enough observations to train "
                    "the machine-learning model. "
                    "Try using 1 Week, 1 Month or 1 Year."
                )
            )

        # --------------------------------------------------
        # Features and target
        # --------------------------------------------------

        X = model_data[
            self.feature_columns
        ]

        y = model_data[
            "NextPrice"
        ]

        # --------------------------------------------------
        # Chronological train/test split
        #
        # IMPORTANT:
        # We do not randomly shuffle time-series data.
        #
        # Earlier data = training
        # Later data   = testing
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Random Forest model
        # --------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # --------------------------------------------------
        # Test predictions
        # --------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # --------------------------------------------------
        # Model evaluation
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mse
        )

        if len(
            y_test
        ) >= 2:

            r2 = r2_score(
                y_test,
                predictions
            )

        else:

            r2 = 0

        # --------------------------------------------------
        # Naive baseline
        #
        # Baseline prediction:
        # next price = current price
        # --------------------------------------------------

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

            improvement_vs_baseline = (
                (
                    baseline_mae
                    - mae
                )
                / baseline_mae
            ) * 100

        else:

            improvement_vs_baseline = 0

        # --------------------------------------------------
        # Feature importance
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Predict next available price
        # --------------------------------------------------

        prediction_data = data.dropna(
            subset=self.feature_columns
        )

        if prediction_data.empty:

            next_price_prediction = 0
            current_price = 0
            predicted_change = 0
            predicted_change_percent = 0

        else:

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

            else:

                predicted_change_percent = 0

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

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
                improvement_vs_baseline,

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

    # --------------------------------------------------
    # Empty result
    # --------------------------------------------------

    def _empty_result(
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