from datetime import datetime, timedelta

import pandas as pd
import requests


class WeatherAPI:

    def __init__(self):

        self.archive_url = (
            "https://archive-api.open-meteo.com/v1/archive"
        )

        self.forecast_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        # Representative weather locations
        # for the two Danish electricity areas.
        #
        # DK1 -> Aarhus
        # DK2 -> Copenhagen

        self.locations = {
            "DK1": {
                "latitude": 56.1629,
                "longitude": 10.2039,
                "name": "Aarhus"
            },
            "DK2": {
                "latitude": 55.6761,
                "longitude": 12.5683,
                "name": "Copenhagen"
            }
        }

    # ==================================================
    # Public method
    # ==================================================

    def get_weather(
        self,
        price_area="DK1",
        start_date="now-P7D",
        end_date="now"
    ):

        if price_area not in self.locations:

            raise ValueError(
                f"Unknown price area: {price_area}"
            )

        start_timestamp = (
            self._resolve_start_date(
                start_date
            )
        )

        end_timestamp = (
            self._resolve_end_date(
                end_date
            )
        )

        historical_data = (
            self._get_historical_weather(
                price_area,
                start_timestamp,
                end_timestamp
            )
        )

        recent_data = (
            self._get_recent_weather(
                price_area,
                start_timestamp,
                end_timestamp
            )
        )

        frames = []

        if not historical_data.empty:

            frames.append(
                historical_data
            )

        if not recent_data.empty:

            frames.append(
                recent_data
            )

        if not frames:

            return self._empty_dataframe()

        weather = pd.concat(
            frames,
            ignore_index=True
        )

        weather = (
            weather.drop_duplicates(
                subset=["TimeDK"],
                keep="last"
            )
        )

        weather = weather.sort_values(
            "TimeDK"
        )

        weather = weather[
            (
                weather["TimeDK"]
                >= start_timestamp
            )
            &
            (
                weather["TimeDK"]
                <= end_timestamp
            )
        ]

        weather = weather.reset_index(
            drop=True
        )

        return weather

    # ==================================================
    # Historical weather
    # ==================================================

    def _get_historical_weather(
        self,
        price_area,
        start_timestamp,
        end_timestamp
    ):

        # Historical archive data can lag behind
        # real time slightly.
        #
        # We therefore use the archive until
        # six days ago and the live forecast API
        # for the most recent period.

        archive_end = (
            pd.Timestamp(
                datetime.now()
                - timedelta(days=6)
            )
        )

        archive_end = archive_end.normalize()

        if start_timestamp > archive_end:

            return self._empty_dataframe()

        requested_end = min(
            end_timestamp,
            archive_end
        )

        location = self.locations[
            price_area
        ]

        params = {
            "latitude":
                location["latitude"],

            "longitude":
                location["longitude"],

            "start_date":
                start_timestamp.strftime(
                    "%Y-%m-%d"
                ),

            "end_date":
                requested_end.strftime(
                    "%Y-%m-%d"
                ),

            "hourly":
                (
                    "temperature_2m,"
                    "wind_speed_10m,"
                    "cloud_cover,"
                    "precipitation"
                ),

            "wind_speed_unit":
                "ms",

            "timezone":
                "Europe/Copenhagen"
        }

        try:

            response = requests.get(
                self.archive_url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

        except requests.exceptions.RequestException:

            return self._empty_dataframe()

        return self._convert_response(
            response.json(),
            price_area
        )

    # ==================================================
    # Recent weather
    # ==================================================

    def _get_recent_weather(
        self,
        price_area,
        start_timestamp,
        end_timestamp
    ):

        location = self.locations[
            price_area
        ]

        params = {
            "latitude":
                location["latitude"],

            "longitude":
                location["longitude"],

            "hourly":
                (
                    "temperature_2m,"
                    "wind_speed_10m,"
                    "cloud_cover,"
                    "precipitation"
                ),

            "wind_speed_unit":
                "ms",

            "past_days":
                7,

            "forecast_days":
                1,

            "timezone":
                "Europe/Copenhagen"
        }

        try:

            response = requests.get(
                self.forecast_url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

        except requests.exceptions.RequestException:

            return self._empty_dataframe()

        dataframe = self._convert_response(
            response.json(),
            price_area
        )

        if dataframe.empty:

            return dataframe

        dataframe = dataframe[
            (
                dataframe["TimeDK"]
                >= start_timestamp
            )
            &
            (
                dataframe["TimeDK"]
                <= end_timestamp
            )
        ]

        dataframe = dataframe.reset_index(
            drop=True
        )

        return dataframe

    # ==================================================
    # Convert API response
    # ==================================================

    def _convert_response(
        self,
        response_data,
        price_area
    ):

        hourly = response_data.get(
            "hourly",
            {}
        )

        times = hourly.get(
            "time",
            []
        )

        if not times:

            return self._empty_dataframe()

        dataframe = pd.DataFrame(
            {
                "TimeDK":
                    times,

                "TemperatureC":
                    hourly.get(
                        "temperature_2m",
                        []
                    ),

                "WeatherWindSpeedMS":
                    hourly.get(
                        "wind_speed_10m",
                        []
                    ),

                "CloudCoverPct":
                    hourly.get(
                        "cloud_cover",
                        []
                    ),

                "PrecipitationMM":
                    hourly.get(
                        "precipitation",
                        []
                    )
            }
        )

        dataframe[
            "TimeDK"
        ] = pd.to_datetime(
            dataframe["TimeDK"],
            errors="coerce"
        )

        numeric_columns = [
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM"
        ]

        for column in numeric_columns:

            dataframe[
                column
            ] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

        dataframe[
            "PriceArea"
        ] = price_area

        dataframe[
            "WeatherLocation"
        ] = self.locations[
            price_area
        ]["name"]

        dataframe = dataframe.dropna(
            subset=["TimeDK"]
        )

        dataframe = dataframe.sort_values(
            "TimeDK"
        )

        dataframe = dataframe.reset_index(
            drop=True
        )

        return dataframe

    # ==================================================
    # Relative dates
    # ==================================================

    def _resolve_start_date(
        self,
        start_date
    ):

        now = pd.Timestamp(
            datetime.now()
        )

        relative_ranges = {
            "now-PT1H":
                timedelta(hours=1),

            "now-P1D":
                timedelta(days=1),

            "now-P7D":
                timedelta(days=7),

            "now-P1W":
                timedelta(days=7),

            "now-P1M":
                timedelta(days=30),

            "now-P1Y":
                timedelta(days=365)
        }

        if start_date in relative_ranges:

            return (
                now
                - relative_ranges[
                    start_date
                ]
            )

        try:

            return pd.Timestamp(
                start_date
            )

        except Exception:

            return (
                now
                - timedelta(days=30)
            )

    def _resolve_end_date(
        self,
        end_date
    ):

        if end_date == "now":

            return pd.Timestamp(
                datetime.now()
            )

        try:

            return pd.Timestamp(
                end_date
            )

        except Exception:

            return pd.Timestamp(
                datetime.now()
            )

    # ==================================================
    # Empty dataframe
    # ==================================================

    def _empty_dataframe(
        self
    ):

        return pd.DataFrame(
            columns=[
                "TimeDK",
                "TemperatureC",
                "WeatherWindSpeedMS",
                "CloudCoverPct",
                "PrecipitationMM",
                "PriceArea",
                "WeatherLocation"
            ]
        )