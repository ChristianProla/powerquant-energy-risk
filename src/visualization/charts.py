from matplotlib.figure import Figure
import matplotlib.dates as mdates


class ChartBuilder:

    def _format_x_axis(
        self,
        axis,
        figure,
        time_range
    ):

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

        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

        if time_range in [
            "1 Week",
            "1 Month",
            "1 Year"
        ]:

            figure.autofmt_xdate(
                rotation=45
            )

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

        axis.set_ylabel(
            "Price (EUR/MWh)"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        self._format_x_axis(
            axis,
            figure,
            time_range
        )

        figure.tight_layout()

        return figure

    def create_returns_chart(
        self,
        dataframe,
        price_area,
        time_range
    ):

        figure = Figure(
            figsize=(10, 5),
            dpi=100
        )

        axis = figure.add_subplot(111)

        axis.plot(
            dataframe["TimeDK"],
            dataframe["PercentageReturn"]
        )

        # Zero line
        axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.6
        )

        axis.set_title(
            f"{price_area} Electricity Price Returns - {time_range}"
        )

        axis.set_ylabel(
            "Return (%)"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        self._format_x_axis(
            axis,
            figure,
            time_range
        )

        figure.tight_layout()

        return figure