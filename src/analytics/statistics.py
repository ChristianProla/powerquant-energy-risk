class StatisticsAnalyzer:

    def calculate_statistics(self, returns_dataframe):

        returns = returns_dataframe[
            "PercentageReturn"
        ].dropna()

        if returns.empty:
            return {
                "mean": 0,
                "std": 0,
                "minimum": 0,
                "maximum": 0,
                "skewness": 0,
                "kurtosis": 0
            }

        statistics = {
            "mean": returns.mean(),
            "std": returns.std(),
            "minimum": returns.min(),
            "maximum": returns.max(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurt()
        }

        return statistics