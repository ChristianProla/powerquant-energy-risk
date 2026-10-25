import pandas as pd
import numpy as np


class FeatureEngineer:

    def create_market_features(
        self,
        price_dataframe,
        market_factors_dataframe
    ):

        if (
            price_dataframe.empty
            or
            market_factors_dataframe.empty
        ):

            return pd.DataFrame()

        prices = price_dataframe.copy()

        factors = (
            market_factors_dataframe.copy()
        )

        # --------------------------------------------------
        # Convert timestamps
        # --------------------------------------------------

        prices["TimeDK"] = pd.to_datetime(
            prices["TimeDK"],
            errors="coerce"
        )

        factors["TimeDK"] = pd.to_datetime(
            factors["TimeDK"],
            errors="coerce"
        )

        prices = prices.dropna(
            subset=[
                "TimeDK"
            ]
        )

        factors = factors.dropna(
            subset=[
                "TimeDK"
            ]
        )

        # --------------------------------------------------
        # Convert electricity prices to hourly data
        # --------------------------------------------------

        prices["Hour"] = (
            prices["TimeDK"].dt.floor("h")
        )

        hourly_prices = (
            prices.groupby(
                "Hour",
                as_index=False
            )["DayAheadPriceEUR"].mean()
        )

        hourly_prices = (
            hourly_prices.rename(
                columns={
                    "Hour": "TimeDK"
                }
            )
        )

        # --------------------------------------------------
        # Convert factors to hourly data
        # --------------------------------------------------

        factors["TimeDK"] = (
            factors[
                "TimeDK"
            ].dt.floor("h")
        )

        factor_columns = []

        if (
            "WindGenerationMW"
            in factors.columns
        ):

            factor_columns.append(
                "WindGenerationMW"
            )

        if (
            "ElectricityDemandMW"
            in factors.columns
        ):

            factor_columns.append(
                "ElectricityDemandMW"
            )

        if not factor_columns:

            return pd.DataFrame()

        hourly_factors = (
            factors.groupby(
                "TimeDK",
                as_index=False
            )[factor_columns].mean()
        )

        # --------------------------------------------------
        # Merge price + factors
        # --------------------------------------------------

        features = pd.merge(
            hourly_prices,
            hourly_factors,
            on="TimeDK",
            how="inner"
        )

        if features.empty:

            return features

        # --------------------------------------------------
        # Make sure columns exist
        # --------------------------------------------------

        if (
            "WindGenerationMW"
            not in features.columns
        ):

            features[
                "WindGenerationMW"
            ] = np.nan

        if (
            "ElectricityDemandMW"
            not in features.columns
        ):

            features[
                "ElectricityDemandMW"
            ] = np.nan

        features = features.sort_values(
            "TimeDK"
        )

        # --------------------------------------------------
        # Calendar features
        # --------------------------------------------------

        features["HourOfDay"] = (
            features["TimeDK"].dt.hour
        )

        features["DayOfWeek"] = (
            features[
                "TimeDK"
            ].dt.dayofweek
        )

        features["Month"] = (
            features[
                "TimeDK"
            ].dt.month
        )

        features["IsWeekend"] = (
            features[
                "DayOfWeek"
            ] >= 5
        ).astype(int)

        # --------------------------------------------------
        # Wind / demand ratio
        # --------------------------------------------------

        demand = features[
            "ElectricityDemandMW"
        ].replace(
            0,
            np.nan
        )

        features[
            "WindDemandRatio"
        ] = (
            features[
                "WindGenerationMW"
            ]
            / demand
        )

        # --------------------------------------------------
        # Price lag features
        # --------------------------------------------------

        features["PriceLag1"] = (
            features[
                "DayAheadPriceEUR"
            ].shift(1)
        )

        features["PriceLag2"] = (
            features[
                "DayAheadPriceEUR"
            ].shift(2)
        )

        features["PriceLag24"] = (
            features[
                "DayAheadPriceEUR"
            ].shift(24)
        )

        # --------------------------------------------------
        # Market factor lag features
        # --------------------------------------------------

        features["WindLag1"] = (
            features[
                "WindGenerationMW"
            ].shift(1)
        )

        features["DemandLag1"] = (
            features[
                "ElectricityDemandMW"
            ].shift(1)
        )

        # --------------------------------------------------
        # Rolling features
        # --------------------------------------------------

        features[
            "PriceMovingAverage6"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].rolling(
                window=6
            ).mean()
        )

        features[
            "PriceMovingAverage24"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].rolling(
                window=24
            ).mean()
        )

        features[
            "PriceVolatility24"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].rolling(
                window=24
            ).std()
        )

        features = features.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        features = features.reset_index(
            drop=True
        )

        return features

    # --------------------------------------------------
    # Market factor statistics
    # --------------------------------------------------

    def calculate_factor_statistics(
        self,
        feature_dataframe
    ):

        if feature_dataframe.empty:

            return self._empty_statistics()

        data = (
            feature_dataframe.copy()
        )

        price = data[
            "DayAheadPriceEUR"
        ]

        wind = data[
            "WindGenerationMW"
        ]

        demand = data[
            "ElectricityDemandMW"
        ]

        wind_price_data = data[
            [
                "DayAheadPriceEUR",
                "WindGenerationMW"
            ]
        ].dropna()

        demand_price_data = data[
            [
                "DayAheadPriceEUR",
                "ElectricityDemandMW"
            ]
        ].dropna()

        if len(
            wind_price_data
        ) >= 2:

            wind_price_correlation = (
                wind_price_data[
                    "DayAheadPriceEUR"
                ].corr(
                    wind_price_data[
                        "WindGenerationMW"
                    ]
                )
            )

        else:

            wind_price_correlation = 0

        if len(
            demand_price_data
        ) >= 2:

            demand_price_correlation = (
                demand_price_data[
                    "DayAheadPriceEUR"
                ].corr(
                    demand_price_data[
                        "ElectricityDemandMW"
                    ]
                )
            )

        else:

            demand_price_correlation = 0

        average_wind = (
            wind.dropna().mean()
        )

        average_demand = (
            demand.dropna().mean()
        )

        average_wind_ratio = (
            data[
                "WindDemandRatio"
            ].dropna().mean()
        )

        if pd.isna(
            wind_price_correlation
        ):

            wind_price_correlation = 0

        if pd.isna(
            demand_price_correlation
        ):

            demand_price_correlation = 0

        if pd.isna(
            average_wind
        ):

            average_wind = 0

        if pd.isna(
            average_demand
        ):

            average_demand = 0

        if pd.isna(
            average_wind_ratio
        ):

            average_wind_ratio = 0

        matched_observations = (
            data[
                [
                    "DayAheadPriceEUR",
                    "WindGenerationMW",
                    "ElectricityDemandMW"
                ]
            ]
            .dropna()
            .shape[0]
        )

        return {
            "wind_price_correlation":
                wind_price_correlation,

            "demand_price_correlation":
                demand_price_correlation,

            "average_wind":
                average_wind,

            "average_demand":
                average_demand,

            "average_wind_ratio":
                average_wind_ratio,

            "observations":
                matched_observations
        }

    def _empty_statistics(self):

        return {
            "wind_price_correlation": 0,
            "demand_price_correlation": 0,
            "average_wind": 0,
            "average_demand": 0,
            "average_wind_ratio": 0,
            "observations": 0
        }