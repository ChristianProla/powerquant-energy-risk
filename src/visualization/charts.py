from matplotlib.figure import Figure
import matplotlib.dates as mdates


class ChartBuilder:

    def create_price_chart(
        self,
        dataframe,
        price_area,
        time_range,
        resolution
    ):

        figure = Figure(
            figsize=(10, 5),
            dpi=100
        )

        axis = figure.add_subplot(111)

        axis.plot(
            dataframe["TimeDK"],
            dataframe["DayAheadPriceEUR"]
        )

        axis.set_title(
            f"{price_area} Day-Ahead Electricity Prices - {time_range}"
        )

        axis.set_ylabel("Price (EUR/MWh)")

        axis.grid(
            True,
            alpha=0.3
        )

        # -----------------------------------------
        # X-axis formatting based on selected range
        # -----------------------------------------

        if time_range == "1 Hour":

            locator = mdates.MinuteLocator(
                interval=15
            )

            formatter = mdates.DateFormatter(
                "%H:%M"
            )

            axis.set_xlabel("Time")

        elif time_range == "1 Day":

            locator = mdates.HourLocator(
                interval=3
            )

            formatter = mdates.DateFormatter(
                "%H:%M"
            )

            axis.set_xlabel("Time")

        elif time_range == "1 Week":

            locator = mdates.DayLocator()

            formatter = mdates.DateFormatter(
                "%d %b"
            )

            axis.set_xlabel("Date")

        elif time_range == "1 Month":

            locator = mdates.DayLocator(
                interval=4
            )

            formatter = mdates.DateFormatter(
                "%d %b"
            )

            axis.set_xlabel("Date")

        elif time_range == "1 Year":

            locator = mdates.MonthLocator()

            formatter = mdates.DateFormatter(
                "%b %Y"
            )

            axis.set_xlabel("Month")

        else:

            locator = mdates.AutoDateLocator()

            formatter = mdates.ConciseDateFormatter(
                locator
            )

            axis.set_xlabel("Time")

        axis.xaxis.set_major_locator(
            locator
        )

        axis.xaxis.set_major_formatter(
            formatter
        )

        if time_range in [
            "1 Week",
            "1 Month",
            "1 Year"
        ]:

            figure.autofmt_xdate(
                rotation=45
            )

        figure.tight_layout()

        return figure