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

        self.base_transaction_cost = 0.25

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
        # Test all valid parameter combinations
        # ==================================================

        for window in rolling_windows:

            if window >= len(
                spread_dataframe
            ):

                continue

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

                    # ------------------------------------------
                    # A mean-reversion exit must be closer to
                    # zero than the entry threshold.
                    # ------------------------------------------

                    if (
                        exit_threshold
                        >= entry_threshold
                    ):

                        continue

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
        # Baseline-cost parameter analysis
        #
        # Parameter robustness is evaluated at one fixed,
        # realistic transaction-cost assumption.
        #
        # Other cost levels are reserved for cost sensitivity.
        # ==================================================

        baseline_results = (
            results_dataframe[
                np.isclose(
                    results_dataframe[
                        "CostPerLeg"
                    ],
                    self.base_transaction_cost
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if baseline_results.empty:

            return self._empty_result(
                "No baseline-cost parameter results were generated."
            )

        # ==================================================
        # Minimum trade filter
        # ==================================================

        minimum_trades = 3

        valid_results = (
            baseline_results[
                baseline_results[
                    "Trades"
                ] >= minimum_trades
            ]
            .copy()
        )

        # --------------------------------------------------
        # If three-trade evidence is unavailable, accept
        # configurations with at least one completed trade.
        #
        # Zero-trade configurations are not considered valid
        # evidence of strategy robustness.
        # --------------------------------------------------

        if valid_results.empty:

            valid_results = (
                baseline_results[
                    baseline_results[
                        "Trades"
                    ] >= 1
                ]
                .copy()
            )

        if valid_results.empty:

            return self._empty_result(
                (
                    "No baseline parameter configuration "
                    "generated a completed trade."
                )
            )

        valid_results = (
            valid_results.reset_index(
                drop=True
            )
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
        # Best individual baseline-cost configuration
        #
        # Reported for reference only.
        # It is not automatically assumed to be robust.
        # ==================================================

        best_configuration = (
            valid_results.sort_values(
                by=[
                    "SharpeProxy",
                    "NetPnL",
                    "Trades"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .iloc[
                0
            ]
        )

        # ==================================================
        # Robust window / entry parameter surface
        #
        # Exit thresholds are averaged while transaction cost
        # remains fixed at the baseline assumption.
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
        #
        # Cost sensitivity intentionally uses the complete
        # cost grid.
        #
        # To keep comparisons fair, we only use parameter
        # configurations that generate at least one trade.
        # ==================================================

        cost_results = (
            results_dataframe[
                results_dataframe[
                    "Trades"
                ] >= 1
            ]
            .copy()
        )

        cost_sensitivity = (
            cost_results.groupby(
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
                ),

                AverageTrades=(
                    "Trades",
                    "mean"
                )
            )
        )

        # ==================================================
        # Top baseline configurations
        # ==================================================

        top_configurations = (
            valid_results.sort_values(
                by=[
                    "SharpeProxy",
                    "NetPnL",
                    "Trades"
                ],
                ascending=[
                    False,
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
        # Average baseline-cost Sharpe for every:
        #
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

            "baseline_results":
                baseline_results,

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

            "base_transaction_cost":
                self.base_transaction_cost,

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

            "baseline_results":
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

            "base_transaction_cost":
                self.base_transaction_cost,

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