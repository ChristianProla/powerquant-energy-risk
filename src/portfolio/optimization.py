from scipy.optimize import minimize_scalar


class PortfolioOptimizer:

    def find_minimum_volatility_portfolio(
        self,
        correlation_data
    ):

        if correlation_data.empty:

            return {
                "dk1_weight": 0.50,
                "dk2_weight": 0.50,
                "mean_return": 0,
                "volatility": 0
            }

        dk1_returns = correlation_data[
            "DK1Return"
        ]

        dk2_returns = correlation_data[
            "DK2Return"
        ]

        # -----------------------------------------
        # Objective function
        # -----------------------------------------

        def objective_function(dk1_weight):

            dk2_weight = 1 - dk1_weight

            portfolio_returns = (
                dk1_weight * dk1_returns
                + dk2_weight * dk2_returns
            )

            portfolio_volatility = (
                portfolio_returns.std()
            )

            return portfolio_volatility

        # -----------------------------------------
        # Optimization
        #
        # Constraint:
        # 0 <= DK1 weight <= 1
        #
        # DK2 weight automatically becomes:
        # 1 - DK1 weight
        # -----------------------------------------

        result = minimize_scalar(
            objective_function,
            bounds=(0, 1),
            method="bounded"
        )

        optimal_dk1_weight = result.x

        optimal_dk2_weight = (
            1 - optimal_dk1_weight
        )

        optimal_portfolio_returns = (
            optimal_dk1_weight * dk1_returns
            + optimal_dk2_weight * dk2_returns
        )

        mean_return = (
            optimal_portfolio_returns.mean()
        )

        volatility = (
            optimal_portfolio_returns.std()
        )

        return {
            "dk1_weight":
                optimal_dk1_weight,

            "dk2_weight":
                optimal_dk2_weight,

            "mean_return":
                mean_return,

            "volatility":
                volatility
        }