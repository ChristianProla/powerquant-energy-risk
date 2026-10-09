import numpy as np


class ReturnCalculator:

    def calculate_returns(self, dataframe):
        data = dataframe.copy()

        price = data["DayAheadPriceEUR"]

        # Percentage return
        data["PercentageReturn"] = (
            price.pct_change(fill_method=None) * 100
        )

        # Log return
        previous_price = price.shift(1)

        valid_log_prices = (
            (price > 0) &
            (previous_price > 0)
        )

        data["LogReturn"] = np.where(
            valid_log_prices,
            np.log(price / previous_price) * 100,
            np.nan
        )

        # Cumulative return relative to the first price
        first_price = price.iloc[0]

        if first_price != 0:
            data["CumulativeReturn"] = (
                (price / first_price) - 1
            ) * 100
        else:
            data["CumulativeReturn"] = np.nan

        # Remove the first row because it has no previous price
        data = data.iloc[1:].reset_index(drop=True)

        return data

    def get_return_statistics(self, returns_dataframe):
        returns = returns_dataframe[
            "PercentageReturn"
        ].dropna()

        if returns.empty:
            return {
                "mean": 0,
                "minimum": 0,
                "maximum": 0
            }

        return {
            "mean": returns.mean(),
            "minimum": returns.min(),
            "maximum": returns.max()
        }