import numpy as np
import pandas as pd


class FeatureEngineer:

    # ==================================================
    # Create features
    # ==================================================

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

        prices = (
            price_dataframe.copy()
        )

        factors = (
            market_factors_dataframe.copy()
        )

        # --------------------------------------------------
        # Timestamps
        # --------------------------------------------------

        prices[
            "TimeDK"
        ] = pd.to_datetime(
            prices["TimeDK"],
            errors="coerce"
        )

        factors[
            "TimeDK"
        ] = pd.to_datetime(
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
        # Hourly electricity price
        # --------------------------------------------------

        prices[
            "Hour"
        ] = (
            prices[
                "TimeDK"
            ].dt.floor("h")
        )

        hourly_prices = (
            prices.groupby(
                "Hour",
                as_index=False
            )[
                "DayAheadPriceEUR"
            ]
            .mean()
        )

        hourly_prices = (
            hourly_prices.rename(
                columns={
                    "Hour":
                        "TimeDK"
                }
            )
        )

        # --------------------------------------------------
        # Ensure all factor columns exist
        # --------------------------------------------------

        expected_factor_columns = [
            "WindGenerationMW",
            "ElectricityDemandMW",
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM"
        ]

        for column in (
            expected_factor_columns
        ):

            if column not in (
                factors.columns
            ):

                factors[
                    column
                ] = np.nan

        # --------------------------------------------------
        # Hourly factors
        # --------------------------------------------------

        factors[
            "TimeDK"
        ] = (
            factors[
                "TimeDK"
            ].dt.floor("h")
        )

        hourly_factors = (
            factors.groupby(
                "TimeDK",
                as_index=False
            )[
                expected_factor_columns
            ]
            .mean()
        )

        # --------------------------------------------------
        # Merge
        # --------------------------------------------------

        features = pd.merge(
            hourly_prices,
            hourly_factors,
            on="TimeDK",
            how="inner"
        )

        if features.empty:

            return features

        features = (
            features.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        # ==================================================
        # Calendar features
        # ==================================================

        features[
            "HourOfDay"
        ] = (
            features[
                "TimeDK"
            ].dt.hour
        )

        features[
            "DayOfWeek"
        ] = (
            features[
                "TimeDK"
            ].dt.dayofweek
        )

        features[
            "Month"
        ] = (
            features[
                "TimeDK"
            ].dt.month
        )

        features[
            "IsWeekend"
        ] = (
            features[
                "DayOfWeek"
            ] >= 5
        ).astype(int)

        # ==================================================
        # Energy system features
        # ==================================================

        demand = (
            features[
                "ElectricityDemandMW"
            ].replace(
                0,
                np.nan
            )
        )

        features[
            "WindDemandRatio"
        ] = (
            features[
                "WindGenerationMW"
            ]
            / demand
        )

        # ==================================================
        # Price lag features
        # ==================================================

        features[
            "PriceLag1"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].shift(1)
        )

        features[
            "PriceLag2"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].shift(2)
        )

        features[
            "PriceLag24"
        ] = (
            features[
                "DayAheadPriceEUR"
            ].shift(24)
        )

        # ==================================================
        # Energy factor lags
        # ==================================================

        features[
            "WindLag1"
        ] = (
            features[
                "WindGenerationMW"
            ].shift(1)
        )

        features[
            "DemandLag1"
        ] = (
            features[
                "ElectricityDemandMW"
            ].shift(1)
        )

        # ==================================================
        # Weather lag features
        # ==================================================

        features[
            "WeatherWindLag1"
        ] = (
            features[
                "WeatherWindSpeedMS"
            ].shift(1)
        )

        features[
            "TemperatureLag1"
        ] = (
            features[
                "TemperatureC"
            ].shift(1)
        )

        features[
            "CloudCoverLag1"
        ] = (
            features[
                "CloudCoverPct"
            ].shift(1)
        )

        # ==================================================
        # Rolling price features
        # ==================================================

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

        # ==================================================
        # Rolling weather features
        # ==================================================

        features[
            "WeatherWindAverage6"
        ] = (
            features[
                "WeatherWindSpeedMS"
            ].rolling(
                window=6
            ).mean()
        )

        features[
            "TemperatureAverage6"
        ] = (
            features[
                "TemperatureC"
            ].rolling(
                window=6
            ).mean()
        )

        # --------------------------------------------------
        # Clean
        # --------------------------------------------------

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

    # ==================================================
    # Factor statistics
    # ==================================================

    def calculate_factor_statistics(
        self,
        feature_dataframe
    ):

        if feature_dataframe.empty:

            return self._empty_statistics()

        data = (
            feature_dataframe.copy()
        )

        statistics = {
            "wind_price_correlation":
                self._correlation(
                    data,
                    "WindGenerationMW"
                ),

            "demand_price_correlation":
                self._correlation(
                    data,
                    "ElectricityDemandMW"
                ),

            "weather_wind_price_correlation":
                self._correlation(
                    data,
                    "WeatherWindSpeedMS"
                ),

            "temperature_price_correlation":
                self._correlation(
                    data,
                    "TemperatureC"
                ),

            "average_wind":
                self._mean(
                    data,
                    "WindGenerationMW"
                ),

            "average_demand":
                self._mean(
                    data,
                    "ElectricityDemandMW"
                ),

            "average_weather_wind":
                self._mean(
                    data,
                    "WeatherWindSpeedMS"
                ),

            "average_temperature":
                self._mean(
                    data,
                    "TemperatureC"
                ),

            "average_cloud_cover":
                self._mean(
                    data,
                    "CloudCoverPct"
                ),

            "average_wind_ratio":
                self._mean(
                    data,
                    "WindDemandRatio"
                )
        }

        required_columns = [
            "DayAheadPriceEUR",
            "WindGenerationMW",
            "ElectricityDemandMW"
        ]

        statistics[
            "observations"
        ] = (
            data[
                required_columns
            ]
            .dropna()
            .shape[0]
        )

        weather_columns = [
            "DayAheadPriceEUR",
            "WeatherWindSpeedMS",
            "TemperatureC"
        ]

        statistics[
            "weather_observations"
        ] = (
            data[
                weather_columns
            ]
            .dropna()
            .shape[0]
        )

        return statistics

    # ==================================================
    # Helpers
    # ==================================================

    def _correlation(
        self,
        dataframe,
        column
    ):

        if column not in (
            dataframe.columns
        ):

            return 0

        clean_data = (
            dataframe[
                [
                    "DayAheadPriceEUR",
                    column
                ]
            ]
            .dropna()
        )

        if len(
            clean_data
        ) < 2:

            return 0

        correlation = (
            clean_data[
                "DayAheadPriceEUR"
            ].corr(
                clean_data[
                    column
                ]
            )
        )

        if pd.isna(
            correlation
        ):

            return 0

        return correlation

    def _mean(
        self,
        dataframe,
        column
    ):

        if column not in (
            dataframe.columns
        ):

            return 0

        value = (
            dataframe[
                column
            ]
            .dropna()
            .mean()
        )

        if pd.isna(
            value
        ):

            return 0

        return value

    def _empty_statistics(
        self
    ):

        return {
            "wind_price_correlation": 0,
            "demand_price_correlation": 0,
            "weather_wind_price_correlation": 0,
            "temperature_price_correlation": 0,
            "average_wind": 0,
            "average_demand": 0,
            "average_weather_wind": 0,
            "average_temperature": 0,
            "average_cloud_cover": 0,
            "average_wind_ratio": 0,
            "observations": 0,
            "weather_observations": 0
        }