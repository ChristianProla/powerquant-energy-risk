import numpy as np


class MonteCarloSimulator:

    def run_simulation(
        self,
        returns_dataframe,
        simulations=10000,
        horizon=30,
        seed=42
    ):

        returns = returns_dataframe[
            "PercentageReturn"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if returns.empty:

            return {
                "paths": np.empty((0, 0)),
                "final_returns": np.array([]),
                "expected_return": 0,
                "median_return": 0,
                "percentile_5": 0,
                "percentile_95": 0,
                "probability_of_loss": 0,
                "historical_mean": 0,
                "historical_volatility": 0,
                "simulations": simulations,
                "horizon": horizon
            }

        historical_mean = returns.mean()
        historical_volatility = returns.std()

        if np.isnan(historical_volatility):
            historical_volatility = 0

        random_generator = np.random.default_rng(
            seed
        )

        simulated_returns = (
            random_generator.normal(
                loc=historical_mean,
                scale=historical_volatility,
                size=(
                    simulations,
                    horizon
                )
            )
        )

        # Cumulative simulated return in percentage points.
        #
        # We use cumulative returns rather than a traditional
        # geometric stock-price model because electricity prices
        # can become negative.

        cumulative_paths = np.cumsum(
            simulated_returns,
            axis=1
        )

        final_returns = (
            cumulative_paths[:, -1]
        )

        expected_return = (
            final_returns.mean()
        )

        median_return = (
            np.median(final_returns)
        )

        percentile_5 = (
            np.percentile(
                final_returns,
                5
            )
        )

        percentile_95 = (
            np.percentile(
                final_returns,
                95
            )
        )

        probability_of_loss = (
            np.mean(
                final_returns < 0
            ) * 100
        )

        return {
            "paths": cumulative_paths,
            "final_returns": final_returns,
            "expected_return": expected_return,
            "median_return": median_return,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95,
            "probability_of_loss": probability_of_loss,
            "historical_mean": historical_mean,
            "historical_volatility": historical_volatility,
            "simulations": simulations,
            "horizon": horizon
        }