class VolatilityAnalyzer:

    def calculate_volatility(self, returns_dataframe):

        data = returns_dataframe.copy()

        returns = data["PercentageReturn"]

        # Rolling volatility over 7 observations
        data["Volatility7"] = returns.rolling(
            window=7
        ).std()

        # Rolling volatility over 30 observations
        data["Volatility30"] = returns.rolling(
            window=30
        ).std()

        return data

    def get_volatility_statistics(self, volatility_dataframe):

        volatility_7 = volatility_dataframe[
            "Volatility7"
        ].dropna()

        volatility_30 = volatility_dataframe[
            "Volatility30"
        ].dropna()

        current_7 = 0
        current_30 = 0
        average_30 = 0
        maximum_30 = 0

        if not volatility_7.empty:
            current_7 = volatility_7.iloc[-1]

        if not volatility_30.empty:
            current_30 = volatility_30.iloc[-1]
            average_30 = volatility_30.mean()
            maximum_30 = volatility_30.max()

        return {
            "current_7": current_7,
            "current_30": current_30,
            "average_30": average_30,
            "maximum_30": maximum_30
        }