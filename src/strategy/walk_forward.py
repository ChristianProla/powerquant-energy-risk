import numpy as np
import pandas as pd


class WalkForwardValidator:

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
    # Main walk-forward validation
    # ==================================================

    def run_validation(
        self,
        spread_dataframe,
        n_folds=4,
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

        n_folds = int(
            n_folds
        )

        if n_folds < 2:

            return self._empty_result(
                "At least two walk-forward folds are required."
            )

        transaction_cost_per_leg = max(
            0.0,
            float(
                transaction_cost_per_leg
            )
        )

        # ==================================================
        # Walk-forward structure
        #
        # Initial training period = 50% of observations.
        # Remaining observations are divided into
        # sequential unseen test folds.
        # ==================================================

        initial_train_size = int(
            len(data)
            * 0.50
        )

        remaining_observations = (
            len(data)
            - initial_train_size
        )

        test_size = (
            remaining_observations
            // n_folds
        )

        if (
            initial_train_size < 50
            or
            test_size < 10
        ):

            return self._empty_result(
                (
                    "Not enough observations for walk-forward "
                    "validation. Try using 1 Year."
                )
            )

        fold_results = []

        all_test_backtests = []

        all_test_trades = []

        parameter_selections = []

        # ==================================================
        # Run folds
        # ==================================================

        for fold_number in range(
            1,
            n_folds + 1
        ):

            train_end_index = (
                initial_train_size
                + (
                    fold_number
                    - 1
                )
                * test_size
            )

            test_start_index = (
                train_end_index
            )

            if (
                fold_number
                == n_folds
            ):

                test_end_index = (
                    len(data)
                )

            else:

                test_end_index = (
                    test_start_index
                    + test_size
                )

            train_data = (
                data.iloc[
                    :train_end_index
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            test_data = (
                data.iloc[
                    test_start_index:
                    test_end_index
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if test_data.empty:

                continue

            # ==================================================
            # Select parameters using TRAIN data only
            # ==================================================

            parameter_results = (
                self._search_parameters(
                    train_data,
                    transaction_cost_per_leg
                )
            )

            if parameter_results.empty:

                continue

            eligible_results = (
                parameter_results[
                    parameter_results[
                        "Trades"
                    ] >= 3
                ]
                .copy()
            )

            if eligible_results.empty:

                eligible_results = (
                    parameter_results[
                        parameter_results[
                            "Trades"
                        ] >= 1
                    ]
                    .copy()
                )

            if eligible_results.empty:

                continue

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

            parameter_selections.append(
                {
                    "Fold":
                        fold_number,

                    "Window":
                        selected_window,

                    "EntryZ":
                        selected_entry,

                    "ExitZ":
                        selected_exit
                }
            )

            # ==================================================
            # Training performance using selected parameters
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
                    entry_threshold=(
                        selected_entry
                    ),
                    exit_threshold=(
                        selected_exit
                    ),
                    transaction_cost_per_leg=(
                        transaction_cost_per_leg
                    )
                )
            )

            if not train_backtest[
                "success"
            ]:

                continue

            # ==================================================
            # Warm-up before unseen test
            #
            # Historical rows are used only to calculate
            # the rolling z-score.
            #
            # Trading begins flat on the first test row.
            # ==================================================

            warmup_size = max(
                selected_window,
                2
            )

            warmup_data = (
                train_data.tail(
                    warmup_size
                )
            )

            combined_data = (
                pd.concat(
                    [
                        warmup_data,
                        test_data
                    ],
                    ignore_index=True
                )
            )

            combined_signals = (
                self.spread_analyzer
                .calculate_signals(
                    combined_data,
                    window=selected_window,
                    threshold=selected_entry
                )
            )

            test_start_time = (
                test_data[
                    "TimeDK"
                ].iloc[
                    0
                ]
            )

            test_signals = (
                combined_signals[
                    combined_signals[
                        "TimeDK"
                    ] >= test_start_time
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            # ==================================================
            # Unseen test backtest
            # ==================================================

            test_backtest = (
                self.backtester
                .run_backtest(
                    test_signals,
                    entry_threshold=(
                        selected_entry
                    ),
                    exit_threshold=(
                        selected_exit
                    ),
                    transaction_cost_per_leg=(
                        transaction_cost_per_leg
                    )
                )
            )

            if not test_backtest[
                "success"
            ]:

                continue

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

            fold_results.append(
                {
                    "Fold":
                        fold_number,

                    "TrainStart":
                        train_data[
                            "TimeDK"
                        ].iloc[
                            0
                        ],

                    "TrainEnd":
                        train_data[
                            "TimeDK"
                        ].iloc[
                            -1
                        ],

                    "TestStart":
                        test_data[
                            "TimeDK"
                        ].iloc[
                            0
                        ],

                    "TestEnd":
                        test_data[
                            "TimeDK"
                        ].iloc[
                            -1
                        ],

                    "Window":
                        selected_window,

                    "EntryZ":
                        selected_entry,

                    "ExitZ":
                        selected_exit,

                    "TrainPnL":
                        train_statistics[
                            "total_net_pnl"
                        ],

                    "TrainSharpe":
                        train_statistics[
                            "sharpe_proxy"
                        ],

                    "TestPnL":
                        test_statistics[
                            "total_net_pnl"
                        ],

                    "TestSharpe":
                        test_statistics[
                            "sharpe_proxy"
                        ],

                    "TestWinRate":
                        test_statistics[
                            "win_rate"
                        ],

                    "TestDrawdown":
                        test_statistics[
                            "maximum_drawdown"
                        ],

                    "TestTrades":
                        test_statistics[
                            "number_of_trades"
                        ]
                }
            )

            fold_backtest_data = (
                test_backtest[
                    "backtest_data"
                ].copy()
            )

            fold_backtest_data[
                "Fold"
            ] = fold_number

            all_test_backtests.append(
                fold_backtest_data
            )

            fold_trades = (
                test_backtest[
                    "trades"
                ].copy()
            )

            if not fold_trades.empty:

                fold_trades[
                    "Fold"
                ] = fold_number

                all_test_trades.append(
                    fold_trades
                )

        # ==================================================
        # Validate result
        # ==================================================

        if not fold_results:

            return self._empty_result(
                (
                    "Walk-forward validation could not "
                    "generate any valid test folds."
                )
            )

        fold_results_dataframe = (
            pd.DataFrame(
                fold_results
            )
        )

        parameter_selections_dataframe = (
            pd.DataFrame(
                parameter_selections
            )
        )

        if all_test_backtests:

            combined_test_data = (
                pd.concat(
                    all_test_backtests,
                    ignore_index=True
                )
            )

        else:

            combined_test_data = (
                pd.DataFrame()
            )

        if all_test_trades:

            combined_trades = (
                pd.concat(
                    all_test_trades,
                    ignore_index=True
                )
            )

        else:

            combined_trades = (
                pd.DataFrame()
            )

        # ==================================================
        # Create continuous walk-forward PnL
        # ==================================================

        if not combined_test_data.empty:

            combined_test_data = (
                combined_test_data.sort_values(
                    "TimeDK"
                )
                .reset_index(
                    drop=True
                )
            )

            combined_test_data[
                "WalkForwardCumulativePnL"
            ] = (
                combined_test_data[
                    "NetPnL"
                ].cumsum()
            )

        # ==================================================
        # Aggregate OOS statistics
        # ==================================================

        total_oos_pnl = (
            fold_results_dataframe[
                "TestPnL"
            ].sum()
        )

        average_oos_pnl = (
            fold_results_dataframe[
                "TestPnL"
            ].mean()
        )

        median_oos_pnl = (
            fold_results_dataframe[
                "TestPnL"
            ].median()
        )

        average_oos_sharpe = (
            fold_results_dataframe[
                "TestSharpe"
            ].mean()
        )

        median_oos_sharpe = (
            fold_results_dataframe[
                "TestSharpe"
            ].median()
        )

        average_win_rate = (
            fold_results_dataframe[
                "TestWinRate"
            ].mean()
        )

        total_trades = int(
            fold_results_dataframe[
                "TestTrades"
            ].sum()
        )

        profitable_folds = int(
            (
                fold_results_dataframe[
                    "TestPnL"
                ] > 0
            ).sum()
        )

        profitable_fold_percentage = (
            profitable_folds
            / len(
                fold_results_dataframe
            )
        ) * 100

        positive_sharpe_folds = int(
            (
                fold_results_dataframe[
                    "TestSharpe"
                ] > 0
            ).sum()
        )

        positive_sharpe_percentage = (
            positive_sharpe_folds
            / len(
                fold_results_dataframe
            )
        ) * 100

        worst_fold_pnl = (
            fold_results_dataframe[
                "TestPnL"
            ].min()
        )

        best_fold_pnl = (
            fold_results_dataframe[
                "TestPnL"
            ].max()
        )

        # ==================================================
        # Aggregate drawdown from full walk-forward curve
        # ==================================================

        maximum_drawdown = (
            self._calculate_max_drawdown(
                combined_test_data
            )
        )

        # ==================================================
        # Parameter stability
        # ==================================================

        parameter_stability = (
            self._calculate_parameter_stability(
                parameter_selections_dataframe
            )
        )

        validation_label = (
            self._classify_validation(
                profitable_fold_percentage,
                positive_sharpe_percentage,
                average_oos_sharpe,
                total_oos_pnl
            )
        )

        return {
            "success":
                True,

            "message":
                (
                    "Walk-forward strategy "
                    "validation completed."
                ),

            "n_folds":
                len(
                    fold_results_dataframe
                ),

            "transaction_cost_per_leg":
                transaction_cost_per_leg,

            "fold_results":
                fold_results_dataframe,

            "combined_test_data":
                combined_test_data,

            "combined_trades":
                combined_trades,

            "parameter_selections":
                parameter_selections_dataframe,

            "parameter_stability":
                parameter_stability,

            "total_oos_pnl":
                total_oos_pnl,

            "average_oos_pnl":
                average_oos_pnl,

            "median_oos_pnl":
                median_oos_pnl,

            "average_oos_sharpe":
                average_oos_sharpe,

            "median_oos_sharpe":
                median_oos_sharpe,

            "average_win_rate":
                average_win_rate,

            "total_trades":
                total_trades,

            "profitable_folds":
                profitable_folds,

            "profitable_fold_percentage":
                profitable_fold_percentage,

            "positive_sharpe_folds":
                positive_sharpe_folds,

            "positive_sharpe_percentage":
                positive_sharpe_percentage,

            "worst_fold_pnl":
                worst_fold_pnl,

            "best_fold_pnl":
                best_fold_pnl,

            "maximum_drawdown":
                maximum_drawdown,

            "validation_label":
                validation_label
        }

    # ==================================================
    # Parameter search
    #
    # TRAIN DATA ONLY
    # ==================================================

    def _search_parameters(
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

                            "SharpeProxy":
                                statistics[
                                    "sharpe_proxy"
                                ],

                            "Trades":
                                statistics[
                                    "number_of_trades"
                                ],

                            "WinRate":
                                statistics[
                                    "win_rate"
                                ],

                            "MaxDrawdown":
                                statistics[
                                    "maximum_drawdown"
                                ]
                        }
                    )

        if not results:

            return pd.DataFrame()

        result_dataframe = (
            pd.DataFrame(
                results
            )
        )

        result_dataframe = (
            result_dataframe.replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        return result_dataframe

    # ==================================================
    # Maximum drawdown
    # ==================================================

    def _calculate_max_drawdown(
        self,
        combined_test_data
    ):

        if combined_test_data.empty:

            return 0

        cumulative = (
            combined_test_data[
                "WalkForwardCumulativePnL"
            ]
        )

        running_peak = (
            cumulative.cummax()
        )

        drawdown = (
            cumulative
            - running_peak
        )

        if drawdown.empty:

            return 0

        return (
            drawdown.min()
        )

    # ==================================================
    # Parameter stability
    # ==================================================

    def _calculate_parameter_stability(
        self,
        selections
    ):

        if selections.empty:

            return {
                "most_common_window":
                    0,

                "window_share":
                    0,

                "most_common_entry":
                    0,

                "entry_share":
                    0,

                "most_common_exit":
                    0,

                "exit_share":
                    0,

                "most_common_configuration":
                    "None",

                "configuration_share":
                    0
            }

        total = len(
            selections
        )

        # --------------------------------------------------
        # Individual parameters
        # --------------------------------------------------

        window_counts = (
            selections[
                "Window"
            ].value_counts()
        )

        entry_counts = (
            selections[
                "EntryZ"
            ].value_counts()
        )

        exit_counts = (
            selections[
                "ExitZ"
            ].value_counts()
        )

        most_common_window = (
            window_counts.index[
                0
            ]
        )

        most_common_entry = (
            entry_counts.index[
                0
            ]
        )

        most_common_exit = (
            exit_counts.index[
                0
            ]
        )

        # --------------------------------------------------
        # Full configuration
        # --------------------------------------------------

        configuration_counts = (
            selections.groupby(
                [
                    "Window",
                    "EntryZ",
                    "ExitZ"
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        best_configuration = (
            configuration_counts.index[
                0
            ]
        )

        best_configuration_count = (
            configuration_counts.iloc[
                0
            ]
        )

        configuration_text = (
            f"{int(best_configuration[0])}h / "
            f"{best_configuration[1]:.1f} / "
            f"{best_configuration[2]:.1f}"
        )

        return {
            "most_common_window":
                int(
                    most_common_window
                ),

            "window_share":
                (
                    window_counts.iloc[
                        0
                    ]
                    / total
                ) * 100,

            "most_common_entry":
                float(
                    most_common_entry
                ),

            "entry_share":
                (
                    entry_counts.iloc[
                        0
                    ]
                    / total
                ) * 100,

            "most_common_exit":
                float(
                    most_common_exit
                ),

            "exit_share":
                (
                    exit_counts.iloc[
                        0
                    ]
                    / total
                ) * 100,

            "most_common_configuration":
                configuration_text,

            "configuration_share":
                (
                    best_configuration_count
                    / total
                ) * 100
        }

    # ==================================================
    # Overall interpretation
    # ==================================================

    def _classify_validation(
        self,
        profitable_fold_percentage,
        positive_sharpe_percentage,
        average_oos_sharpe,
        total_oos_pnl
    ):

        if (
            profitable_fold_percentage >= 75
            and
            positive_sharpe_percentage >= 75
            and
            average_oos_sharpe >= 0.50
            and
            total_oos_pnl > 0
        ):

            return "Strong"

        if (
            profitable_fold_percentage >= 50
            and
            positive_sharpe_percentage >= 50
            and
            average_oos_sharpe > 0
            and
            total_oos_pnl > 0
        ):

            return "Moderate"

        if (
            total_oos_pnl > 0
            or
            average_oos_sharpe > 0
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

            "n_folds":
                0,

            "transaction_cost_per_leg":
                0,

            "fold_results":
                pd.DataFrame(),

            "combined_test_data":
                pd.DataFrame(),

            "combined_trades":
                pd.DataFrame(),

            "parameter_selections":
                pd.DataFrame(),

            "parameter_stability":
                {},

            "total_oos_pnl":
                0,

            "average_oos_pnl":
                0,

            "median_oos_pnl":
                0,

            "average_oos_sharpe":
                0,

            "median_oos_sharpe":
                0,

            "average_win_rate":
                0,

            "total_trades":
                0,

            "profitable_folds":
                0,

            "profitable_fold_percentage":
                0,

            "positive_sharpe_folds":
                0,

            "positive_sharpe_percentage":
                0,

            "worst_fold_pnl":
                0,

            "best_fold_pnl":
                0,

            "maximum_drawdown":
                0,

            "validation_label":
                "No Data"
        }