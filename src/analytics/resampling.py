import pandas as pd


class MarketDataResampler:

    def resample_for_range(self, dataframe, time_range):
        """
        Automatically chooses a suitable data resolution
        based on the selected time range.
        """

        if dataframe.empty:
            return dataframe, "No data"

        data = dataframe.copy()

        data = data.set_index("TimeDK")

        if time_range in ["1H", "1D"]:
            # Keep the original market data
            resampled_data = data
            resolution = "15-minute / raw"

        elif time_range == "1W":
            # Average electricity price for each hour
            resampled_data = data.resample("h").mean(
                numeric_only=True
            )
            resolution = "Hourly"

        elif time_range == "1M":
            # Average electricity price for each day
            resampled_data = data.resample("D").mean(
                numeric_only=True
            )
            resolution = "Daily"

        elif time_range == "1Y":
            # Daily average gives approximately 365 points
            resampled_data = data.resample("D").mean(
                numeric_only=True
            )
            resolution = "Daily"

        elif time_range == "5Y":
            # Weekly average
            resampled_data = data.resample("W").mean(
                numeric_only=True
            )
            resolution = "Weekly"

        elif time_range == "10Y":
            # Monthly average
            resampled_data = data.resample("ME").mean(
                numeric_only=True
            )
            resolution = "Monthly"

        else:
            resampled_data = data
            resolution = "Raw"

        resampled_data = resampled_data.dropna()

        resampled_data = resampled_data.reset_index()

        return resampled_data, resolution