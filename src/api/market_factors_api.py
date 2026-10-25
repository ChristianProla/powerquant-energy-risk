import requests
import pandas as pd


class MarketFactorsAPI:

    def __init__(self):

        self.base_url = (
            "https://api.energidataservice.dk/"
            "dataset/GenerationProdTypeExchange"
        )

    def get_market_factors(
        self,
        price_area="DK1",
        start_date="now-P7D",
        end_date="now"
    ):

        params = {
            "filter": (
                '{"PriceArea":["'
                + price_area
                + '"]}'
            ),
            "start": start_date,
            "end": end_date,
            "limit": 0
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

        return self._prepare_data(
            dataframe,
            price_area
        )

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    def _prepare_data(
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

        data[time_column] = pd.to_datetime(
            data[time_column],
            errors="coerce"
        )

        result = pd.DataFrame()

        result["TimeDK"] = data[
            time_column
        ]

        result["PriceArea"] = (
            price_area
        )

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
                data[wind_column],
                errors="coerce"
            )

        else:

            result[
                "WindGenerationMW"
            ] = self._sum_wind_columns(
                data
            )

        # --------------------------------------------------
        # Demand / consumption
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
                data[demand_column],
                errors="coerce"
            )

        else:

            result[
                "ElectricityDemandMW"
            ] = float("nan")

        result = result.dropna(
            subset=[
                "TimeDK"
            ]
        )

        result = result.sort_values(
            "TimeDK"
        )

        result = result.reset_index(
            drop=True
        )

        return result

    # --------------------------------------------------
    # Find timestamp
    # --------------------------------------------------

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

            name = column.lower()

            if (
                "time" in name
                or
                "hour" in name
            ):

                return column

        return None

    # --------------------------------------------------
    # Find total wind column
    # --------------------------------------------------

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

            name = column.lower()

            if (
                "wind" in name
                and
                "total" in name
            ):

                return column

        return None

    # --------------------------------------------------
    # Sum onshore + offshore wind
    # --------------------------------------------------

    def _sum_wind_columns(
        self,
        dataframe
    ):

        wind_columns = []

        for column in dataframe.columns:

            name = column.lower()

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
                float("nan"),
                index=dataframe.index
            )

        numeric_wind = dataframe[
            wind_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        return numeric_wind.sum(
            axis=1,
            min_count=1
        )

    # --------------------------------------------------
    # Find demand column
    # --------------------------------------------------

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

            name = column.lower()

            if (
                "consumption" in name
                or
                "load" in name
            ):

                return column

        return None