class PortfolioAnalyzer:

    def build_portfolio(
        self,
        correlation_data,
        dk1_weight=0.50
    ):

        data = correlation_data.copy()

        dk2_weight = 1 - dk1_weight

        data["PortfolioReturn"] = (
            dk1_weight * data["DK1Return"]
            + dk2_weight * data["DK2Return"]
        )

        data["CumulativePortfolioReturn"] = (
            (
                1
                + data["PortfolioReturn"] / 100
            ).cumprod()
            - 1
        ) * 100

        return data

    def calculate_portfolio_statistics(
        self,
        portfolio_data,
        dk1_weight
    ):

        dk2_weight = 1 - dk1_weight

        if portfolio_data.empty:

            return {
                "dk1_weight": dk1_weight,
                "dk2_weight": dk2_weight,
                "mean_return": 0,
                "volatility": 0,
                "dk1_volatility": 0,
                "dk2_volatility": 0,
                "weighted_individual_volatility": 0,
                "diversification_benefit": 0
            }

        dk1_returns = portfolio_data[
            "DK1Return"
        ]

        dk2_returns = portfolio_data[
            "DK2Return"
        ]

        portfolio_returns = portfolio_data[
            "PortfolioReturn"
        ]

        dk1_volatility = dk1_returns.std()
        dk2_volatility = dk2_returns.std()

        portfolio_volatility = (
            portfolio_returns.std()
        )

        weighted_individual_volatility = (
            dk1_weight * dk1_volatility
            + dk2_weight * dk2_volatility
        )

        diversification_benefit = (
            weighted_individual_volatility
            - portfolio_volatility
        )

        return {
            "dk1_weight": dk1_weight,
            "dk2_weight": dk2_weight,
            "mean_return": portfolio_returns.mean(),
            "volatility": portfolio_volatility,
            "dk1_volatility": dk1_volatility,
            "dk2_volatility": dk2_volatility,
            "weighted_individual_volatility":
                weighted_individual_volatility,
            "diversification_benefit":
                diversification_benefit
        }