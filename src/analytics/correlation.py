import pandas as pd


class CorrelationAnalyzer:

    def prepare_correlation_data(
        self,
        dk1_returns,
        dk2_returns
    ):

        dk1 = dk1_returns[
            ["TimeDK", "PercentageReturn"]
        ].copy()

        dk2 = dk2_returns[
            ["TimeDK", "PercentageReturn"]
        ].copy()

        dk1 = dk1.rename(
            columns={
                "PercentageReturn": "DK1Return"
            }
        )

        dk2 = dk2.rename(
            columns={
                "PercentageReturn": "DK2Return"
            }
        )

        correlation_data = pd.merge(
            dk1,
            dk2,
            on="TimeDK",
            how="inner"
        )

        correlation_data = correlation_data.dropna()

        correlation_data = correlation_data.reset_index(
            drop=True
        )

        return correlation_data

    def calculate_correlation(
        self,
        correlation_data
    ):

        if len(correlation_data) < 2:

            return {
                "correlation": 0,
                "strength": "Not enough data",
                "observations": len(correlation_data)
            }

        correlation = correlation_data[
            "DK1Return"
        ].corr(
            correlation_data["DK2Return"]
        )

        strength = self.interpret_correlation(
            correlation
        )

        return {
            "correlation": correlation,
            "strength": strength,
            "observations": len(correlation_data)
        }

    def interpret_correlation(
        self,
        correlation
    ):

        absolute_correlation = abs(correlation)

        if absolute_correlation >= 0.80:
            strength = "Very Strong"

        elif absolute_correlation >= 0.60:
            strength = "Strong"

        elif absolute_correlation >= 0.40:
            strength = "Moderate"

        elif absolute_correlation >= 0.20:
            strength = "Weak"

        else:
            strength = "Very Weak"

        if correlation > 0:
            direction = "Positive"

        elif correlation < 0:
            direction = "Negative"

        else:
            direction = "No"

        return f"{strength} {direction} Correlation"