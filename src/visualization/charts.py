from matplotlib.figure import Figure


class ChartBuilder:

    def create_price_chart(self, dataframe, price_area, time_range):

        figure = Figure(figsize=(10, 5), dpi=100)

        axis = figure.add_subplot(111)

        axis.plot(
            dataframe["TimeDK"],
            dataframe["DayAheadPriceEUR"]
        )

        axis.set_title(
            f"{price_area} Day-Ahead Electricity Prices - {time_range}"
        )

        axis.set_xlabel("Time")
        axis.set_ylabel("Price (EUR/MWh)")

        axis.grid(True, alpha=0.3)

        figure.autofmt_xdate()

        figure.tight_layout()

        return figure