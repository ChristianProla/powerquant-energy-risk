import numpy as np
import pandas as pd


class StrategyRobustnessAnalyzer:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.spread_analyzer = (
            spread_analyzer
        )

        self.backtester = (
            backtester
        )

    # ==================================================
    # Run full robustness test
    # ==================================================

    def run_robustness_test(
        self,
        spread_dataframe
    ):

        if spread_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        # ==================================================
        # Parameter grid
        # ==================================================

        rolling_windows = [
            12,
            24,
            48,
            72,
            168
        ]

        entry_thresholds = [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0
        ]

        exit_thresholds = [
            0.0,
            0.5,
            1.0
        ]

        transaction_costs = [
            0.00,
            0.25,
            0.50,
            1.00
        ]

        results = []

        # ==================================================
        # Test all parameter combinations
        # ==================================================

        for window in rolling_windows:

            signal_data = (
                self.spread_analyzer
                .calculate_signals(
                    spread_dataframe,
                    window=window,
                    threshold=1.0
                )
            )

            for entry_threshold in (
                entry_thresholds
            ):

                for exit_threshold in (
                    exit_thresholds
                ):

                    for transaction_cost in (
                        transaction_costs
                    ):

                        backtest = (
                            self.backtester
                            .run_backtest(
                                signal_data,
                                entry_threshold=(
                                    entry_threshold
                                ),
                                exit_threshold=(
                                    exit_threshold
                                ),
                                transaction_cost_per_leg=(
                                    transaction_cost
                                )
                            )
                        )

                        if not backtest[
                            "success"
                        ]:

                            continue

                        statistics = (
                            backtest[
                                "statistics"
                            ]
                        )

                        results.append(
                            {
                                "Window":
                                    window,

                                "EntryZ":
                                    entry_threshold,

                                "ExitZ":
                                    exit_threshold,

                                "CostPerLeg":
                                    transaction_cost,

                                "NetPnL":
                                    statistics[
                                        "total_net_pnl"
                                    ],

                                "GrossPnL":
                                    statistics[
                                        "total_gross_pnl"
                                    ],

                                "TradingCosts":
                                    statistics[
                                        "total_transaction_cost"
                                    ],

                                "Trades":
                                    statistics[
                                        "number_of_trades"
                                    ],

                                "WinRate":
                                    statistics[
                                        "win_rate"
                                    ],

                                "AverageTrade":
                                    statistics[
                                        "average_trade"
                                    ],

                                "SharpeProxy":
                                    statistics[
                                        "sharpe_proxy"
                                    ],

                                "MaxDrawdown":
                                    statistics[
                                        "maximum_drawdown"
                                    ],

                                "ProfitFactor":
                                    statistics[
                                        "profit_factor"
                                    ]
                            }
                        )

        results_dataframe = (
            pd.DataFrame(
                results
            )
        )

        if results_dataframe.empty:

            return self._empty_result(
                "No valid parameter combinations were generated."
            )

        # ==================================================
        # Replace infinite values
        # ==================================================

        results_dataframe = (
            results_dataframe.replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        # ==================================================
        # Minimum trade filter
        #
        # A parameter combination with one lucky trade
        # is not useful evidence of robustness.
        # ==================================================

        minimum_trades = 3

        valid_results = (
            results_dataframe[
                results_dataframe[
                    "Trades"
                ] >= minimum_trades
            ]
            .copy()
        )

        if valid_results.empty:

            valid_results = (
                results_dataframe.copy()
            )

        # ==================================================
        # Summary statistics
        # ==================================================

        profitable_percentage = (
            (
                valid_results[
                    "NetPnL"
                ] > 0
            ).mean()
            * 100
        )

        positive_sharpe_percentage = (
            (
                valid_results[
                    "SharpeProxy"
                ] > 0
            ).mean()
            * 100
        )

        median_net_pnl = (
            valid_results[
                "NetPnL"
            ].median()
        )

        median_sharpe = (
            valid_results[
                "SharpeProxy"
            ].median()
        )

        average_net_pnl = (
            valid_results[
                "NetPnL"
            ].mean()
        )

        average_sharpe = (
            valid_results[
                "SharpeProxy"
            ].mean()
        )

        # ==================================================
        # Best individual configuration
        #
        # We report this, but do NOT automatically assume
        # it is the strategy we should use.
        # ==================================================

        best_configuration = (
            valid_results.sort_values(
                by=[
                    "SharpeProxy",
                    "NetPnL"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .iloc[
                0
            ]
        )

        # ==================================================
        # Robust window / entry combinations
        #
        # Here we average across:
        # - exit threshold
        # - transaction costs
        #
        # This is more useful than looking at one isolated
        # best parameter combination.
        # ==================================================

        parameter_surface = (
            valid_results.groupby(
                [
                    "Window",
                    "EntryZ"
                ],
                as_index=False
            )
            .agg(
                AverageSharpe=(
                    "SharpeProxy",
                    "mean"
                ),
                AverageNetPnL=(
                    "NetPnL",
                    "mean"
                ),
                MedianNetPnL=(
                    "NetPnL",
                    "median"
                ),
                ProfitableShare=(
                    "NetPnL",
                    lambda values:
                    (
                        values > 0
                    ).mean()
                    * 100
                ),
                AverageTrades=(
                    "Trades",
                    "mean"
                ),
                AverageDrawdown=(
                    "MaxDrawdown",
                    "mean"
                )
            )
        )

        parameter_surface = (
            parameter_surface.sort_values(
                by=[
                    "AverageSharpe",
                    "ProfitableShare",
                    "AverageNetPnL"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )

        robust_configuration = (
            parameter_surface.iloc[
                0
            ]
        )

        # ==================================================
        # Transaction-cost sensitivity
        # ==================================================

        cost_sensitivity = (
            valid_results.groupby(
                "CostPerLeg",
                as_index=False
            )
            .agg(
                AverageNetPnL=(
                    "NetPnL",
                    "mean"
                ),
                MedianNetPnL=(
                    "NetPnL",
                    "median"
                ),
                AverageSharpe=(
                    "SharpeProxy",
                    "mean"
                ),
                ProfitableShare=(
                    "NetPnL",
                    lambda values:
                    (
                        values > 0
                    ).mean()
                    * 100
                )
            )
        )

        # ==================================================
        # Top configurations
        # ==================================================

        top_configurations = (
            valid_results.sort_values(
                by=[
                    "SharpeProxy",
                    "NetPnL"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .head(
                10
            )
            .reset_index(
                drop=True
            )
        )

        # ==================================================
        # Heatmap data
        #
        # Average Sharpe for every:
        # window x entry threshold
        # ==================================================

        heatmap_data = (
            valid_results.pivot_table(
                index="Window",
                columns="EntryZ",
                values="SharpeProxy",
                aggfunc="mean"
            )
        )

        heatmap_data = (
            heatmap_data.sort_index()
        )

        heatmap_data = (
            heatmap_data.reindex(
                sorted(
                    heatmap_data.columns
                ),
                axis=1
            )
        )

        # ==================================================
        # Robustness classification
        # ==================================================

        robustness_label = (
            self._classify_robustness(
                profitable_percentage,
                positive_sharpe_percentage,
                median_sharpe
            )
        )

        return {
            "success":
                True,

            "message":
                "Strategy robustness testing completed.",

            "all_results":
                results_dataframe,

            "valid_results":
                valid_results,

            "top_configurations":
                top_configurations,

            "parameter_surface":
                parameter_surface,

            "cost_sensitivity":
                cost_sensitivity,

            "heatmap_data":
                heatmap_data,

            "best_configuration":
                best_configuration,

            "robust_configuration":
                robust_configuration,

            "total_combinations":
                len(
                    results_dataframe
                ),

            "valid_combinations":
                len(
                    valid_results
                ),

            "minimum_trades":
                minimum_trades,

            "profitable_percentage":
                profitable_percentage,

            "positive_sharpe_percentage":
                positive_sharpe_percentage,

            "median_net_pnl":
                median_net_pnl,

            "median_sharpe":
                median_sharpe,

            "average_net_pnl":
                average_net_pnl,

            "average_sharpe":
                average_sharpe,

            "robustness_label":
                robustness_label
        }

    # ==================================================
    # Robustness interpretation
    # ==================================================

    def _classify_robustness(
        self,
        profitable_percentage,
        positive_sharpe_percentage,
        median_sharpe
    ):

        if (
            profitable_percentage >= 70
            and
            positive_sharpe_percentage >= 70
            and
            median_sharpe > 0.75
        ):

            return "Strong"

        if (
            profitable_percentage >= 55
            and
            positive_sharpe_percentage >= 55
            and
            median_sharpe > 0
        ):

            return "Moderate"

        if (
            profitable_percentage >= 40
            or
            positive_sharpe_percentage >= 40
        ):

            return "Weak"

        return "Poor"

    # ==================================================
    # Empty result
    # ==================================================

    def _empty_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "all_results":
                pd.DataFrame(),

            "valid_results":
                pd.DataFrame(),

            "top_configurations":
                pd.DataFrame(),

            "parameter_surface":
                pd.DataFrame(),

            "cost_sensitivity":
                pd.DataFrame(),

            "heatmap_data":
                pd.DataFrame(),

            "best_configuration":
                None,

            "robust_configuration":
                None,

            "total_combinations":
                0,

            "valid_combinations":
                0,

            "minimum_trades":
                0,

            "profitable_percentage":
                0,

            "positive_sharpe_percentage":
                0,

            "median_net_pnl":
                0,

            "median_sharpe":
                0,

            "average_net_pnl":
                0,

            "average_sharpe":
                0,

            "robustness_label":
                "No Data"
        }