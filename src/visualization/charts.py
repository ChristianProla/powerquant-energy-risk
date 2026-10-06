import matplotlib.pyplot as plt


class ChartBuilder:
    def plot_price_chart(self, dataframe):
        plt.figure(figsize=(12, 6))

        plt.plot(
            dataframe["TimeDK"],
            dataframe["DayAheadPriceEUR"]
        )

        plt.title("DK1 Day-Ahead Electricity Prices")
        plt.xlabel("Time")
        plt.ylabel("Price (EUR/MWh)")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()