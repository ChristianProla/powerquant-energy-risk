from src.api.electricity_api import ElectricityAPI
from src.visualization.charts import ChartBuilder


def main():
    api = ElectricityAPI()

    prices = api.get_prices(
        price_area="DK1",
        start_date="2016-01-01",
        end_date="2026-01-01"
    )

    print(prices)

    print()
    print("Number of data points:", len(prices))

    if not prices.empty:
        print("First date:", prices["TimeDK"].min())
        print("Last date:", prices["TimeDK"].max())

        charts = ChartBuilder()

        charts.plot_price_chart(prices)


if __name__ == "__main__":
    main()