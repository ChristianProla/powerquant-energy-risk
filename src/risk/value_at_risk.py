import numpy as np


class ValueAtRiskAnalyzer:

    def calculate_historical_var(
        self,
        returns_dataframe
    ):

        returns = returns_dataframe[
            "PercentageReturn"
        ].dropna()

        if returns.empty:

            return {
                "var_95": 0,
                "var_99": 0,
                "worst_return": 0,
                "average_loss": 0
            }

        percentile_5 = np.percentile(
            returns,
            5
        )

        percentile_1 = np.percentile(
            returns,
            1
        )

        losses = returns[
            returns < 0
        ]

        if losses.empty:
            average_loss = 0
        else:
            average_loss = losses.mean()

        return {
            "var_95": max(0, -percentile_5),
            "var_99": max(0, -percentile_1),
            "worst_return": returns.min(),
            "average_loss": average_loss
        }