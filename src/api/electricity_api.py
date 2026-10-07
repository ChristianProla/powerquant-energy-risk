import requests
import pandas as pd


class ElectricityAPI:
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset/DayAheadPrices"

    def get_prices(
        self,
        price_area="DK1",
        start_date="now-P1D",
        end_date="now"
    ):
        params = {
            "filter": '{"PriceArea":["' + price_area + '"]}',
            "columns": "TimeDK,PriceArea,DayAheadPriceEUR",
            "start": start_date,
            "end": end_date,
            "limit": 0
        }

        response = requests.get(
            self.base_url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()
        records = data["records"]

        dataframe = pd.DataFrame(records)

        if dataframe.empty:
            return dataframe

        dataframe["TimeDK"] = pd.to_datetime(
            dataframe["TimeDK"]
        )

        dataframe["DayAheadPriceEUR"] = pd.to_numeric(
            dataframe["DayAheadPriceEUR"]
        )

        dataframe = dataframe.sort_values(
            by="TimeDK"
        )

        dataframe = dataframe.reset_index(
            drop=True
        )

        return dataframe