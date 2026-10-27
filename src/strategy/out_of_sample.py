import numpy as np
import pandas as pd


class OutOfSampleValidator:

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

        self.windows = [
            12,
            24,
            48,
            72,
            168
        ]

        self.entry_thresholds = [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0
        ]

        self.exit_thresholds = [
            0.0,
            0.5,
            1.0
        ]

    # ==================================================
    # Main validation
    # ==================================================

    def run_validation(
        self,
        spread_dataframe,
        train_ratio=0.70,
        transaction_cost_per_leg=0.25
    ):

        if spread_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        data = (
            spread_dataframe.copy()
        )

        data = (
            data.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        train_ratio = float(
            train_ratio
        )

        if (
            train_ratio <= 0.50
            or
            train_ratio >= 0.95
        ):

            return self._empty_result(
                "Train ratio must be between 50% and 95%."
            )

        transaction_cost_per_leg = max(
            0.0,
            float(
                transaction_cost_per_leg
            )
        )

        # ==================================================
        # Chronological train / test split
        # ==================================================

        split_index = int(
            len(data)
            * train_ratio
        )

        if (
            split_index < 30
            or
            (
                len(data)
                - split_index
            ) < 20
        ):

            return self._empty_result(
                (
                    "Not enough observations for "
                    "out-of-sample validation. "
                    "Try using 1 Month or 1 Year."
                )
            )

        train_data = (
            data.iloc[
                :split_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_data = (
            data.iloc[
                split_index:
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_start_time = (
            test_data[
                "TimeDK"
            ].iloc[
                0
            ]
        )

        # ==================================================
        # Parameter selection
        #
        # IMPORTANT:
        # Only training data is used here.
        # ==================================================

        training_results = (
            self._search_training_parameters(
                train_data,
                transaction_cost_per_leg
            )
        )

        if training_results.empty:

            return self._empty_result(
                (
                    "No valid strategy parameters "
                    "could be selected from the training period."
                )
            )

        # --------------------------------------------------
        # Prefer configurations with several trades.
        # One lucky trade is weak evidence.
        # --------------------------------------------------

        minimum_trades = 3

        eligible_results = (
            training_results[
                training_results[
                    "Trades"
                ] >= minimum_trades
            ]
            .copy()
        )

        if eligible_results.empty:

            eligible_results = (
                training_results[
                    training_results[
                        "Trades"
                    ] >= 1
                ]
                .copy()
            )

        if eligible_results.empty:

            return self._empty_result(
                (
                    "The training period generated "
                    "no completed trades."
                )
            )

        # --------------------------------------------------
        # Selection rule:
        #
        # 1. Highest Sharpe Proxy
        # 2. Highest Net PnL
        # 3. More trades
        #
        # All selection happens before test data is used.
        # --------------------------------------------------

        eligible_results = (
            eligible_results.sort_values(
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
            .reset_index(
                drop=True
            )
        )

        selected = (
            eligible_results.iloc[
                0
            ]
        )

        selected_window = int(
            selected[
                "Window"
            ]
        )

        selected_entry = float(
            selected[
                "EntryZ"
            ]
        )

        selected_exit = float(
            selected[
                "ExitZ"
            ]
        )

        # ==================================================
        # Run selected strategy on training period
        # ==================================================

        train_signals = (
            self.spread_analyzer
            .calculate_signals(
                train_data,
                window=selected_window,
                threshold=selected_entry
            )
        )

        train_backtest = (
            self.backtester
            .run_backtest(
                train_signals,
                entry_threshold=selected_entry,
                exit_threshold=selected_exit,
                transaction_cost_per_leg=(
                    transaction_cost_per_leg
                )
            )
        )

        if not train_backtest[
            "success"
        ]:

            return self._empty_result(
                train_backtest[
                    "message"
                ]
            )

        # ==================================================
        # Prepare unseen test period
        #
        # We include historical training observations only
        # as a warm-up for the rolling z-score.
        #
        # Trading still begins flat at the first test row.
        # ==================================================

        warmup_length = max(
            selected_window,
            2
        )

        warmup_data = (
            train_data.tail(
                warmup_length
            )
        )

        combined_test_data = (
            pd.concat(
                [
                    warmup_data,
                    test_data
                ],
                ignore_index=True
            )
        )

        combined_test_signals = (
            self.spread_analyzer
            .calculate_signals(
                combined_test_data,
                window=selected_window,
                threshold=selected_entry
            )
        )

        test_signals = (
            combined_test_signals[
                combined_test_signals[
                    "TimeDK"
                ] >= test_start_time
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # ==================================================
        # Completely unseen test backtest
        # ==================================================

        test_backtest = (
            self.backtester
            .run_backtest(
                test_signals,
                entry_threshold=selected_entry,
                exit_threshold=selected_exit,
                transaction_cost_per_leg=(
                    transaction_cost_per_leg
                )
            )
        )

        if not test_backtest[
            "success"
        ]:

            return self._empty_result(
                test_backtest[
                    "message"
                ]
            )

        train_statistics = (
            train_backtest[
                "statistics"
            ]
        )

        test_statistics = (
            test_backtest[
                "statistics"
            ]
        )

        # ==================================================
        # Generalization metrics
        # ==================================================

        train_sharpe = (
            train_statistics[
                "sharpe_proxy"
            ]
        )

        test_sharpe = (
            test_statistics[
                "sharpe_proxy"
            ]
        )

        if train_sharpe > 0:

            sharpe_retention = (
                test_sharpe
                / train_sharpe
            ) * 100

        else:

            sharpe_retention = 0

        train_average_trade = (
            train_statistics[
                "average_trade"
            ]
        )

        test_average_trade = (
            test_statistics[
                "average_trade"
            ]
        )

        if train_average_trade != 0:

            average_trade_retention = (
                test_average_trade
                / abs(
                    train_average_trade
                )
            ) * 100

        else:

            average_trade_retention = 0

        generalization_label = (
            self._classify_generalization(
                test_statistics,
                sharpe_retention
            )
        )

        # ==================================================
        # Top training configurations
        # ==================================================

        top_training_configurations = (
            eligible_results.head(
                10
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        return {
            "success":
                True,

            "message":
                (
                    "Out-of-sample strategy "
                    "validation completed."
                ),

            "train_ratio":
                train_ratio,

            "test_ratio":
                1 - train_ratio,

            "train_observations":
                len(
                    train_data
                ),

            "test_observations":
                len(
                    test_data
                ),

            "train_start":
                train_data[
                    "TimeDK"
                ].iloc[
                    0
                ],

            "train_end":
                train_data[
                    "TimeDK"
                ].iloc[
                    -1
                ],

            "test_start":
                test_data[
                    "TimeDK"
                ].iloc[
                    0
                ],

            "test_end":
                test_data[
                    "TimeDK"
                ].iloc[
                    -1
                ],

            "selected_window":
                selected_window,

            "selected_entry":
                selected_entry,

            "selected_exit":
                selected_exit,

            "transaction_cost_per_leg":
                transaction_cost_per_leg,

            "train_statistics":
                train_statistics,

            "test_statistics":
                test_statistics,

            "train_backtest_data":
                train_backtest[
                    "backtest_data"
                ],

            "test_backtest_data":
                test_backtest[
                    "backtest_data"
                ],

            "train_trades":
                train_backtest[
                    "trades"
                ],

            "test_trades":
                test_backtest[
                    "trades"
                ],

            "parameter_results":
                training_results,

            "top_training_configurations":
                top_training_configurations,

            "sharpe_retention":
                sharpe_retention,

            "average_trade_retention":
                average_trade_retention,

            "generalization_label":
                generalization_label,

            "minimum_trades":
                minimum_trades
        }

    # ==================================================
    # Search parameters using training data only
    # ==================================================

    def _search_training_parameters(
        self,
        train_data,
        transaction_cost_per_leg
    ):

        results = []

        available_windows = [
            window
            for window in self.windows
            if window < len(
                train_data
            )
        ]

        for window in (
            available_windows
        ):

            for entry_threshold in (
                self.entry_thresholds
            ):

                signal_data = (
                    self.spread_analyzer
                    .calculate_signals(
                        train_data,
                        window=window,
                        threshold=entry_threshold
                    )
                )

                for exit_threshold in (
                    self.exit_thresholds
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
                                transaction_cost_per_leg
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

                    profit_factor = (
                        statistics[
                            "profit_factor"
                        ]
                    )

                    if np.isinf(
                        profit_factor
                    ):

                        profit_factor = np.nan

                    results.append(
                        {
                            "Window":
                                window,

                            "EntryZ":
                                entry_threshold,

                            "ExitZ":
                                exit_threshold,

                            "NetPnL":
                                statistics[
                                    "total_net_pnl"
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
                                profit_factor
                        }
                    )

        results_dataframe = (
            pd.DataFrame(
                results
            )
        )

        if not results_dataframe.empty:

            results_dataframe = (
                results_dataframe.replace(
                    [
                        np.inf,
                        -np.inf
                    ],
                    np.nan
                )
            )

        return results_dataframe

    # ==================================================
    # Generalization classification
    # ==================================================

    def _classify_generalization(
        self,
        test_statistics,
        sharpe_retention
    ):

        test_pnl = (
            test_statistics[
                "total_net_pnl"
            ]
        )

        test_sharpe = (
            test_statistics[
                "sharpe_proxy"
            ]
        )

        number_of_trades = (
            test_statistics[
                "number_of_trades"
            ]
        )

        if (
            number_of_trades >= 3
            and
            test_pnl > 0
            and
            test_sharpe >= 0.50
            and
            sharpe_retention >= 50
        ):

            return "Strong"

        if (
            test_pnl > 0
            and
            test_sharpe > 0
        ):

            return "Moderate"

        if (
            test_pnl > 0
            or
            test_sharpe > 0
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

            "train_ratio":
                0,

            "test_ratio":
                0,

            "train_observations":
                0,

            "test_observations":
                0,

            "train_start":
                None,

            "train_end":
                None,

            "test_start":
                None,

            "test_end":
                None,

            "selected_window":
                0,

            "selected_entry":
                0,

            "selected_exit":
                0,

            "transaction_cost_per_leg":
                0,

            "train_statistics":
                {},

            "test_statistics":
                {},

            "train_backtest_data":
                pd.DataFrame(),

            "test_backtest_data":
                pd.DataFrame(),

            "train_trades":
                pd.DataFrame(),

            "test_trades":
                pd.DataFrame(),

            "parameter_results":
                pd.DataFrame(),

            "top_training_configurations":
                pd.DataFrame(),

            "sharpe_retention":
                0,

            "average_trade_retention":
                0,

            "generalization_label":
                "No Data",

            "minimum_trades":
                0
        }