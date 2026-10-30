from datetime import timedelta

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

        self.timezone = (
            "Europe/Copenhagen"
        )

        # ==================================================
        # Representative weather locations
        # for the Danish electricity areas
        #
        # DK1 -> Aarhus
        # DK2 -> Copenhagen
        #
        # These are geographic proxies for each price area,
        # not complete regional weather representations.
        # ==================================================

        self.locations = {
            "DK1": {
                "latitude":
                    56.1629,

                "longitude":
                    10.2039,

                "name":
                    "Aarhus"
            },

            "DK2": {
                "latitude":
                    55.6761,

                "longitude":
                    12.5683,

                "name":
                    "Copenhagen"
            }
        }

    # ==================================================
    # Copenhagen-local current time
    #
    # Open-Meteo returns local Danish timestamps because
    # requests use timezone="Europe/Copenhagen".
    #
    # We therefore compare those timestamps with a Danish
    # local clock rather than the computer's local timezone.
    #
    # tz_localize(None) deliberately keeps the Danish local
    # clock time while returning the naive timestamps used
    # elsewhere in PowerQuant.
    # ==================================================

    def _current_copenhagen_time(
        self
    ):

        return (
            pd.Timestamp.now(
                tz=self.timezone
            )
            .tz_localize(
                None
            )
        )

    # ==================================================
    # Historical + recent weather
    # ==================================================

    def get_weather(
        self,
        price_area="DK1",
        start_date="now-P7D",
        end_date="now"
    ):

        self._validate_price_area(
            price_area
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

        if (
            start_timestamp
            >
            end_timestamp
        ):

            return (
                self._empty_dataframe()
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

            return (
                self._empty_dataframe()
            )

        weather = pd.concat(
            frames,
            ignore_index=True
        )

        weather = (
            weather
            .drop_duplicates(
                subset=[
                    "TimeDK"
                ],
                keep="last"
            )
            .sort_values(
                "TimeDK"
            )
        )

        weather = weather[
            (
                weather[
                    "TimeDK"
                ]
                >= start_timestamp
            )
            &
            (
                weather[
                    "TimeDK"
                ]
                <= end_timestamp
            )
        ]

        weather = (
            weather.reset_index(
                drop=True
            )
        )

        return weather

    # ==================================================
    # Future weather forecast
    # ==================================================

    def get_forecast_weather(
        self,
        price_area="DK1",
        forecast_hours=24
    ):

        self._validate_price_area(
            price_area
        )

        try:

            requested_hours = int(
                forecast_hours
            )

        except (
            TypeError,
            ValueError
        ):

            return (
                self._empty_dataframe()
            )

        requested_hours = max(
            1,
            min(
                requested_hours,
                72
            )
        )

        location = self.locations[
            price_area
        ]

        # --------------------------------------------------
        # Request a few extra hours because the API response
        # can include the current local hour.
        # --------------------------------------------------

        params = {
            "latitude":
                location[
                    "latitude"
                ],

            "longitude":
                location[
                    "longitude"
                ],

            "hourly":
                (
                    "temperature_2m,"
                    "wind_speed_10m,"
                    "cloud_cover,"
                    "precipitation"
                ),

            "wind_speed_unit":
                "ms",

            "forecast_hours":
                requested_hours + 3,

            "timezone":
                self.timezone
        }

        try:

            response = requests.get(
                self.forecast_url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

        except (
            requests.exceptions
            .RequestException
        ):

            return (
                self._empty_dataframe()
            )

        dataframe = (
            self._convert_response(
                response.json(),
                price_area
            )
        )

        if dataframe.empty:

            return dataframe

        # ==================================================
        # Future hours only
        #
        # Critical:
        # compare Danish API timestamps with Danish current
        # local time, not the machine's timezone.
        # ==================================================

        current_hour = (
            self._current_copenhagen_time()
            .floor(
                "h"
            )
        )

        dataframe = (
            dataframe[
                dataframe[
                    "TimeDK"
                ]
                >
                current_hour
            ]
            .sort_values(
                "TimeDK"
            )
            .drop_duplicates(
                subset=[
                    "TimeDK"
                ],
                keep="last"
            )
            .head(
                requested_hours
            )
            .reset_index(
                drop=True
            )
        )

        return dataframe

    # ==================================================
    # Historical weather
    # ==================================================

    def _get_historical_weather(
        self,
        price_area,
        start_timestamp,
        end_timestamp
    ):

        # --------------------------------------------------
        # Open-Meteo archive data is available with a delay.
        #
        # Keep a conservative six-day boundary between the
        # archive endpoint and recent forecast endpoint.
        # --------------------------------------------------

        archive_end = (
            self._current_copenhagen_time()
            - timedelta(
                days=6
            )
        ).normalize()

        if (
            start_timestamp
            > archive_end
        ):

            return (
                self._empty_dataframe()
            )

        requested_end = min(
            end_timestamp,
            archive_end
        )

        location = self.locations[
            price_area
        ]

        params = {
            "latitude":
                location[
                    "latitude"
                ],

            "longitude":
                location[
                    "longitude"
                ],

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
                self.timezone
        }

        try:

            response = requests.get(
                self.archive_url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

        except (
            requests.exceptions
            .RequestException
        ):

            return (
                self._empty_dataframe()
            )

        return (
            self._convert_response(
                response.json(),
                price_area
            )
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
                location[
                    "latitude"
                ],

            "longitude":
                location[
                    "longitude"
                ],

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
                self.timezone
        }

        try:

            response = requests.get(
                self.forecast_url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

        except (
            requests.exceptions
            .RequestException
        ):

            return (
                self._empty_dataframe()
            )

        dataframe = (
            self._convert_response(
                response.json(),
                price_area
            )
        )

        if dataframe.empty:

            return dataframe

        dataframe = dataframe[
            (
                dataframe[
                    "TimeDK"
                ]
                >= start_timestamp
            )
            &
            (
                dataframe[
                    "TimeDK"
                ]
                <= end_timestamp
            )
        ]

        dataframe = (
            dataframe
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

        return dataframe

    # ==================================================
    # Convert Open-Meteo response
    # ==================================================

    def _convert_response(
        self,
        response_data,
        price_area
    ):

        if not isinstance(
            response_data,
            dict
        ):

            return (
                self._empty_dataframe()
            )

        hourly = response_data.get(
            "hourly",
            {}
        )

        if not isinstance(
            hourly,
            dict
        ):

            return (
                self._empty_dataframe()
            )

        times = hourly.get(
            "time",
            []
        )

        if not times:

            return (
                self._empty_dataframe()
            )

        number_of_rows = len(
            times
        )

        def safe_values(
            key
        ):

            values = hourly.get(
                key
            )

            if (
                values is None
                or
                len(
                    values
                )
                != number_of_rows
            ):

                return [
                    None
                ] * number_of_rows

            return values

        dataframe = pd.DataFrame(
            {
                "TimeDK":
                    times,

                "TemperatureC":
                    safe_values(
                        "temperature_2m"
                    ),

                "WeatherWindSpeedMS":
                    safe_values(
                        "wind_speed_10m"
                    ),

                "CloudCoverPct":
                    safe_values(
                        "cloud_cover"
                    ),

                "PrecipitationMM":
                    safe_values(
                        "precipitation"
                    )
            }
        )

        dataframe[
            "TimeDK"
        ] = pd.to_datetime(
            dataframe[
                "TimeDK"
            ],
            errors="coerce"
        )

        numeric_columns = [
            "TemperatureC",
            "WeatherWindSpeedMS",
            "CloudCoverPct",
            "PrecipitationMM"
        ]

        for column in (
            numeric_columns
        ):

            dataframe[
                column
            ] = pd.to_numeric(
                dataframe[
                    column
                ],
                errors="coerce"
            )

        dataframe[
            "PriceArea"
        ] = (
            price_area
        )

        dataframe[
            "WeatherLocation"
        ] = (
            self.locations[
                price_area
            ][
                "name"
            ]
        )

        dataframe = (
            dataframe
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
            .reset_index(
                drop=True
            )
        )

        return dataframe

    # ==================================================
    # Validate price area
    # ==================================================

    def _validate_price_area(
        self,
        price_area
    ):

        if (
            price_area
            not in self.locations
        ):

            raise ValueError(
                (
                    "Unknown price area: "
                    f"{price_area}"
                )
            )

    # ==================================================
    # Relative dates
    # ==================================================

    def _resolve_start_date(
        self,
        start_date
    ):

        now = (
            self._current_copenhagen_time()
        )

        relative_ranges = {
            "now-PT1H":
                timedelta(
                    hours=1
                ),

            "now-P1D":
                timedelta(
                    days=1
                ),

            "now-P7D":
                timedelta(
                    days=7
                ),

            "now-P1W":
                timedelta(
                    days=7
                ),

            "now-P1M":
                timedelta(
                    days=30
                ),

            "now-P1Y":
                timedelta(
                    days=365
                )
        }

        if (
            start_date
            in relative_ranges
        ):

            return (
                now
                - relative_ranges[
                    start_date
                ]
            )

        try:

            timestamp = pd.Timestamp(
                start_date
            )

        except Exception:

            return (
                now
                - timedelta(
                    days=30
                )
            )

        # --------------------------------------------------
        # If an explicit timezone-aware timestamp is supplied,
        # convert it to Copenhagen local time before removing
        # timezone information.
        # --------------------------------------------------

        if (
            timestamp.tzinfo
            is not None
        ):

            timestamp = (
                timestamp
                .tz_convert(
                    self.timezone
                )
                .tz_localize(
                    None
                )
            )

        return timestamp

    def _resolve_end_date(
        self,
        end_date
    ):

        now = (
            self._current_copenhagen_time()
        )

        if (
            end_date
            == "now"
        ):

            return now

        try:

            timestamp = pd.Timestamp(
                end_date
            )

        except Exception:

            return now

        if (
            timestamp.tzinfo
            is not None
        ):

            timestamp = (
                timestamp
                .tz_convert(
                    self.timezone
                )
                .tz_localize(
                    None
                )
            )

        return timestamp

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