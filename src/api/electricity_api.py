import requests
import pandas as pd


class ElectricityAPI:
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset/DayAheadPrices"

    def get_prices(
        self,
        price_area="DK1",
        start_date=None,
        end_date=None
    ):
        params = {
            "filter": '{"PriceArea":["' + price_area + '"]}',
            "columns": "TimeDK,PriceArea,DayAheadPriceEUR"
        }

        if start_date is not None:
            params["start"] = start_date

        if end_date is not None:
            params["end"] = end_date

        response = requests.get(
            self.base_url,
            params=params
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