import numpy as np
import pandas as pd


class PriceForecaster:

    def moving_average_forecast(
        self,
        price_dataframe,
        horizon=10,
        window=10
    ):

        prices = price_dataframe[
            "DayAheadPriceEUR"
        ].dropna()

        if prices.empty:

            return {
                "method": "Moving Average",
                "forecast_prices": np.array([]),
                "forecast_periods": np.array([]),
                "current_price": 0,
                "forecast_price": 0,
                "price_change": 0,
                "price_change_percent": 0,
                "horizon": horizon
            }

        current_price = prices.iloc[-1]

        effective_window = min(
            window,
            len(prices)
        )

        moving_average = prices.tail(
            effective_window
        ).mean()

        forecast_prices = np.full(
            horizon,
            moving_average
        )

        forecast_periods = np.arange(
            1,
            horizon + 1
        )

        forecast_price = (
            forecast_prices[-1]
        )

        price_change = (
            forecast_price
            - current_price
        )

        if current_price != 0:

            price_change_percent = (
                price_change
                / abs(current_price)
            ) * 100

        else:

            price_change_percent = 0

        return {
            "method":
                "Moving Average",

            "forecast_prices":
                forecast_prices,

            "forecast_periods":
                forecast_periods,

            "current_price":
                current_price,

            "forecast_price":
                forecast_price,

            "price_change":
                price_change,

            "price_change_percent":
                price_change_percent,

            "horizon":
                horizon
        }

    def linear_trend_forecast(
        self,
        price_dataframe,
        horizon=10
    ):

        prices = price_dataframe[
            "DayAheadPriceEUR"
        ].dropna()

        if prices.empty:

            return {
                "method": "Linear Trend",
                "forecast_prices": np.array([]),
                "forecast_periods": np.array([]),
                "current_price": 0,
                "forecast_price": 0,
                "price_change": 0,
                "price_change_percent": 0,
                "slope": 0,
                "horizon": horizon
            }

        current_price = prices.iloc[-1]

        x_values = np.arange(
            len(prices)
        )

        if len(prices) < 2:

            slope = 0
            intercept = current_price

        else:

            slope, intercept = np.polyfit(
                x_values,
                prices.to_numpy(),
                1
            )

        future_x = np.arange(
            len(prices),
            len(prices) + horizon
        )

        forecast_prices = (
            slope * future_x
            + intercept
        )

        forecast_periods = np.arange(
            1,
            horizon + 1
        )

        forecast_price = (
            forecast_prices[-1]
        )

        price_change = (
            forecast_price
            - current_price
        )

        if current_price != 0:

            price_change_percent = (
                price_change
                / abs(current_price)
            ) * 100

        else:

            price_change_percent = 0

        return {
            "method":
                "Linear Trend",

            "forecast_prices":
                forecast_prices,

            "forecast_periods":
                forecast_periods,

            "current_price":
                current_price,

            "forecast_price":
                forecast_price,

            "price_change":
                price_change,

            "price_change_percent":
                price_change_percent,

            "slope":
                slope,

            "horizon":
                horizon
        }

    def forecast(
        self,
        price_dataframe,
        method="Moving Average",
        horizon=10
    ):

        if method == "Linear Trend":

            return self.linear_trend_forecast(
                price_dataframe,
                horizon=horizon
            )

        return self.moving_average_forecast(
            price_dataframe,
            horizon=horizon
        )