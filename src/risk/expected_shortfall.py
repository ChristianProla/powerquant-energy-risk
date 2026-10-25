import numpy as np


class ExpectedShortfallAnalyzer:

    def calculate_historical_expected_shortfall(
        self,
        returns_dataframe
    ):

        returns = returns_dataframe[
            "PercentageReturn"
        ].dropna()

        if returns.empty:

            return {
                "es_95": 0,
                "es_99": 0
            }

        # -----------------------------------------
        # VaR thresholds
        # -----------------------------------------

        percentile_5 = np.percentile(
            returns,
            5
        )

        percentile_1 = np.percentile(
            returns,
            1
        )

        # -----------------------------------------
        # Tail losses beyond VaR
        # -----------------------------------------

        losses_95 = returns[
            returns <= percentile_5
        ]

        losses_99 = returns[
            returns <= percentile_1
        ]

        # -----------------------------------------
        # Expected Shortfall
        # -----------------------------------------

        if losses_95.empty:
            es_95 = 0
        else:
            es_95 = -losses_95.mean()

        if losses_99.empty:
            es_99 = 0
        else:
            es_99 = -losses_99.mean()

        return {
            "es_95": max(0, es_95),
            "es_99": max(0, es_99)
        }