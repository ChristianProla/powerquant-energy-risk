from src.api.electricity_api import ElectricityAPI


def main():
    api = ElectricityAPI()

    prices = api.get_prices(
        price_area="DK1",
        limit=20
    )

    print(prices)


if __name__ == "__main__":
    main()