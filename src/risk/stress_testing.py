import numpy as np


class StressTester:

    def __init__(self):

        self.scenarios = {
            "Moderate Downside": {
                "market_shock": -10.0,
                "volatility_multiplier": 1.25
            },

            "Severe Downside": {
                "market_shock": -20.0,
                "volatility_multiplier": 1.50
            },

            "Extreme Crisis": {
                "market_shock": -40.0,
                "volatility_multiplier": 2.00
            },

            "Upside Spike": {
                "market_shock": 20.0,
                "volatility_multiplier": 1.40
            }
        }

    def get_scenarios(self):

        return list(
            self.scenarios.keys()
        )

    def run_stress_test(
        self,
        returns_dataframe,
        scenario_name
    ):

        returns = returns_dataframe[
            "PercentageReturn"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if returns.empty:

            return {
                "scenario": scenario_name,
                "market_shock": 0,
                "volatility_multiplier": 0,
                "baseline_mean": 0,
                "baseline_volatility": 0,
                "stressed_return": 0,
                "stressed_volatility": 0,
                "estimated_loss": 0,
                "volatility_change": 0
            }

        scenario = self.scenarios[
            scenario_name
        ]

        market_shock = scenario[
            "market_shock"
        ]

        volatility_multiplier = scenario[
            "volatility_multiplier"
        ]

        baseline_mean = returns.mean()

        baseline_volatility = returns.std()

        if np.isnan(
            baseline_volatility
        ):

            baseline_volatility = 0

        # -----------------------------------------
        # Stress scenario
        # -----------------------------------------
        #
        # The market shock is treated as an
        # additional one-period return shock.
        #
        # Example:
        #
        # historical mean = 1%
        # market shock    = -20%
        #
        # stressed return = -19%
        # -----------------------------------------

        stressed_return = (
            baseline_mean
            + market_shock
        )

        stressed_volatility = (
            baseline_volatility
            * volatility_multiplier
        )

        estimated_loss = max(
            0,
            -stressed_return
        )

        volatility_change = (
            stressed_volatility
            - baseline_volatility
        )

        return {
            "scenario":
                scenario_name,

            "market_shock":
                market_shock,

            "volatility_multiplier":
                volatility_multiplier,

            "baseline_mean":
                baseline_mean,

            "baseline_volatility":
                baseline_volatility,

            "stressed_return":
                stressed_return,

            "stressed_volatility":
                stressed_volatility,

            "estimated_loss":
                estimated_loss,

            "volatility_change":
                volatility_change
        }