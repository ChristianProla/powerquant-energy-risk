import requests
import pandas as pd


class ElectricityAPI:
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset/DayAheadPrices"

    def get_prices(self, price_area="DK1", limit=20):
        params = {
            "filter": '{"PriceArea":["' + price_area + '"]}',
            "columns": "TimeDK,PriceArea,DayAheadPriceEUR",
            "limit": limit
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        data = response.json()
        records = data["records"]

        dataframe = pd.DataFrame(records)

        # Convert TimeDK into a real datetime value
        dataframe["TimeDK"] = pd.to_datetime(dataframe["TimeDK"])

        # Make sure electricity prices are numeric
        dataframe["DayAheadPriceEUR"] = pd.to_numeric(
            dataframe["DayAheadPriceEUR"]
        )

        # Sort data from oldest to newest
        dataframe = dataframe.sort_values(by="TimeDK")

        # Reset row numbers after sorting
        dataframe = dataframe.reset_index(drop=True)

        return dataframe