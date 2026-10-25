import requests
import pandas as pd
import numpy as np

from src.api.weather_api import WeatherAPI


class MarketFactorsAPI:

    def __init__(self):

        self.base_url = (
            "https://api.energidataservice.dk/"
            "dataset/GenerationProdTypeExchange"
        )

        self.weather_api = (
            WeatherAPI()
        )

    # ==================================================
    # Main method
    # ==================================================

    def get_market_factors(
        self,
        price_area="DK1",
        start_date="now-P7D",
        end_date="now"
    ):

        energy_data = (
            self._get_energy_factors(
                price_area,
                start_date,
                end_date
            )
        )

        if energy_data.empty:

            return energy_data

        weather_data = (
            self.weather_api.get_weather(
                price_area=price_area,
                start_date=start_date,
                end_date=end_date
            )
        )

        combined_data = (
            self._merge_weather_data(
                energy_data,
                weather_data
            )
        )

        return combined_data

    # ==================================================
    # Energinet market factors
    # ==================================================

    def _get_energy_factors(
        self,
        price_area,
        start_date,
        end_date
    ):

        params = {
            "filter":
                (
                    '{"PriceArea":["'
                    + price_area
                    + '"]}'
                ),

            "start":
                start_date,

            "end":
                end_date,

            "limit":
                0
        }

        response = requests.get(
            self.base_url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        records = data.get(
            "records",
            []
        )

        dataframe = pd.DataFrame(
            records
        )

        if dataframe.empty:

            return dataframe

        return self._prepare_energy_data(
            dataframe,
            price_area
        )

    # ==================================================
    # Prepare Energinet data
    # ==================================================

    def _prepare_energy_data(
        self,
        dataframe,
        price_area
    ):

        data = dataframe.copy()

        time_column = (
            self._find_time_column(
                data
            )
        )

        if time_column is None:

            raise ValueError(
                "Could not identify the time column "
                "in market factor data."
            )

        data[
            time_column
        ] = pd.to_datetime(
            data[time_column],
            errors="coerce"
        )

        result = pd.DataFrame()

        result[
            "TimeDK"
        ] = data[
            time_column
        ]

        result[
            "PriceArea"
        ] = price_area

        # --------------------------------------------------
        # Wind generation
        # --------------------------------------------------

        wind_column = (
            self._find_total_wind_column(
                data
            )
        )

        if wind_column is not None:

            result[
                "WindGenerationMW"
            ] = pd.to_numeric(
                data[
                    wind_column
                ],
                errors="coerce"
            )

        else:

            result[
                "WindGenerationMW"
            ] = self._sum_wind_columns(
                data
            )

        # --------------------------------------------------
        # Electricity demand
        # --------------------------------------------------

        demand_column = (
            self._find_demand_column(
                data
            )
        )

        if demand_column is not None:

            result[
                "ElectricityDemandMW"
            ] = pd.to_numeric(
                data[
                    demand_column
                ],
                errors="coerce"
            )

        else:

            result[
                "ElectricityDemandMW"
            ] = np.nan

        result = result.dropna(
            subset=[
                "TimeDK"
            ]
        )

        result[
            "TimeDK"
        ] = (
            result[
                "TimeDK"
            ].dt.floor("h")
        )

        result = (
            result.groupby(
                [
                    "TimeDK",
                    "PriceArea"
                ],
                as_index=False
            )[
                [
                    "WindGenerationMW",
                    "ElectricityDemandMW"
                ]
            ]
            .mean()
        )

        result = result.sort_values(
            "TimeDK"
        )

        result = result.reset_index(
            drop=True
        )

        return result

    # ==================================================
    # Merge weather
    # ==================================================

    def _merge_weather_data(
        self,
        energy_data,
        weather_data
    ):

        result = energy_data.copy()

        weather_columns = [
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM"
        ]

        if weather_data.empty:

            for column in weather_columns:

                result[
                    column
                ] = np.nan

            return result

        weather = weather_data.copy()

        weather[
            "TimeDK"
        ] = pd.to_datetime(
            weather["TimeDK"],
            errors="coerce"
        )

        weather[
            "TimeDK"
        ] = (
            weather[
                "TimeDK"
            ].dt.floor("h")
        )

        weather = (
            weather.groupby(
                "TimeDK",
                as_index=False
            )[weather_columns]
            .mean()
        )

        result = pd.merge(
            result,
            weather,
            on="TimeDK",
            how="left"
        )

        return result

    # ==================================================
    # Find timestamp
    # ==================================================

    def _find_time_column(
        self,
        dataframe
    ):

        candidates = [
            "HourDK",
            "TimeDK",
            "Minutes15DK",
            "Minutes5DK",
            "HourUTC",
            "TimeUTC"
        ]

        for candidate in candidates:

            if candidate in dataframe.columns:

                return candidate

        for column in dataframe.columns:

            name = (
                column.lower()
            )

            if (
                "time" in name
                or
                "hour" in name
            ):

                return column

        return None

    # ==================================================
    # Wind generation
    # ==================================================

    def _find_total_wind_column(
        self,
        dataframe
    ):

        preferred = [
            "WindPower",
            "WindGeneration",
            "TotalWindPower",
            "WindProduction"
        ]

        for candidate in preferred:

            if candidate in dataframe.columns:

                return candidate

        for column in dataframe.columns:

            name = (
                column.lower()
            )

            if (
                "wind" in name
                and
                "total" in name
            ):

                return column

        return None

    def _sum_wind_columns(
        self,
        dataframe
    ):

        wind_columns = []

        for column in dataframe.columns:

            name = (
                column.lower()
            )

            if "wind" not in name:

                continue

            if (
                "forecast" in name
                or
                "share" in name
                or
                "capacity" in name
            ):

                continue

            if (
                "onshore" in name
                or
                "offshore" in name
            ):

                wind_columns.append(
                    column
                )

        if not wind_columns:

            return pd.Series(
                np.nan,
                index=dataframe.index
            )

        numeric_wind = (
            dataframe[
                wind_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        return numeric_wind.sum(
            axis=1,
            min_count=1
        )

    # ==================================================
    # Electricity demand
    # ==================================================

    def _find_demand_column(
        self,
        dataframe
    ):

        preferred = [
            "TotalConsumption",
            "GrossConsumption",
            "GrossCon",
            "Consumption",
            "TotalLoad",
            "Load"
        ]

        for candidate in preferred:

            if candidate in dataframe.columns:

                return candidate

        for column in dataframe.columns:

            name = (
                column.lower()
            )

            if (
                "consumption" in name
                or
                "load" in name
            ):

                return column

        return None