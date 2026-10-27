import numpy as np
import pandas as pd


class SpreadBacktester:

    # ==================================================
    # Run mean-reversion backtest
    # ==================================================

    def run_backtest(
        self,
        signal_dataframe,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0.25
    ):

        if signal_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        data = (
            signal_dataframe.copy()
        )

        required_columns = [
            "TimeDK",
            "Spread",
            "ZScore"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            return self._empty_result(
                (
                    "Missing required columns: "
                    + ", ".join(
                        missing_columns
                    )
                )
            )

        data = data.sort_values(
            "TimeDK"
        )

        data = data.reset_index(
            drop=True
        )

        entry_threshold = max(
            0.5,
            float(
                entry_threshold
            )
        )

        exit_threshold = max(
            0.0,
            float(
                exit_threshold
            )
        )

        transaction_cost_per_leg = max(
            0.0,
            float(
                transaction_cost_per_leg
            )
        )

        # --------------------------------------------------
        # Position definitions
        #
        # +1 = Long spread
        #      Long DK1 / Short DK2
        #
        # -1 = Short spread
        #      Short DK1 / Long DK2
        #
        #  0 = No position
        # --------------------------------------------------

        position = 0

        entry_spread = None

        entry_time = None

        entry_z_score = None

        trade_direction = None

        trades = []

        positions = []

        gross_pnl_values = []

        transaction_cost_values = []

        net_pnl_values = []

        entry_markers = []

        exit_markers = []

        # ==================================================
        # Backtest loop
        # ==================================================

        for index in range(
            len(data)
        ):

            current_spread = (
                data.loc[
                    index,
                    "Spread"
                ]
            )

            current_z_score = (
                data.loc[
                    index,
                    "ZScore"
                ]
            )

            current_time = (
                data.loc[
                    index,
                    "TimeDK"
                ]
            )

            # --------------------------------------------------
            # PnL from position held during previous interval
            #
            # Important:
            # We calculate PnL BEFORE making a new trading
            # decision at this timestamp.
            #
            # This avoids using the current signal to profit
            # from a price move that already happened.
            # --------------------------------------------------

            if index == 0:

                spread_change = 0

            else:

                previous_spread = (
                    data.loc[
                        index - 1,
                        "Spread"
                    ]
                )

                spread_change = (
                    current_spread
                    - previous_spread
                )

            gross_pnl = (
                position
                * spread_change
            )

            transaction_cost = 0

            entry_marker = np.nan

            exit_marker = np.nan

            # ==================================================
            # Existing position
            # ==================================================

            if (
                position != 0
                and
                not pd.isna(
                    current_z_score
                )
            ):

                should_exit = False

                # ----------------------------------------------
                # Long spread:
                # entered when z-score was very negative.
                #
                # Exit when z-score has reverted toward zero.
                # ----------------------------------------------

                if position == 1:

                    if (
                        current_z_score
                        >= -exit_threshold
                    ):

                        should_exit = True

                # ----------------------------------------------
                # Short spread:
                # entered when z-score was very positive.
                #
                # Exit when z-score has reverted toward zero.
                # ----------------------------------------------

                elif position == -1:

                    if (
                        current_z_score
                        <= exit_threshold
                    ):

                        should_exit = True

                if should_exit:

                    # Two legs:
                    # DK1 + DK2

                    exit_cost = (
                        transaction_cost_per_leg
                        * 2
                    )

                    transaction_cost += (
                        exit_cost
                    )

                    # ------------------------------------------
                    # Trade-level gross PnL
                    # ------------------------------------------

                    if position == 1:

                        trade_gross_pnl = (
                            current_spread
                            - entry_spread
                        )

                    else:

                        trade_gross_pnl = (
                            entry_spread
                            - current_spread
                        )

                    # Entry + exit both involve two legs

                    total_trade_cost = (
                        transaction_cost_per_leg
                        * 4
                    )

                    trade_net_pnl = (
                        trade_gross_pnl
                        - total_trade_cost
                    )

                    trades.append(
                        {
                            "EntryTime":
                                entry_time,

                            "ExitTime":
                                current_time,

                            "Direction":
                                trade_direction,

                            "EntrySpread":
                                entry_spread,

                            "ExitSpread":
                                current_spread,

                            "EntryZScore":
                                entry_z_score,

                            "ExitZScore":
                                current_z_score,

                            "GrossPnL":
                                trade_gross_pnl,

                            "TransactionCost":
                                total_trade_cost,

                            "NetPnL":
                                trade_net_pnl
                        }
                    )

                    exit_marker = (
                        current_spread
                    )

                    position = 0

                    entry_spread = None
                    entry_time = None
                    entry_z_score = None
                    trade_direction = None

            # ==================================================
            # New position
            #
            # Only enter when currently flat.
            # ==================================================

            if (
                position == 0
                and
                not pd.isna(
                    current_z_score
                )
            ):

                # ----------------------------------------------
                # Spread unusually high
                #
                # Short DK1 / Long DK2
                # ----------------------------------------------

                if (
                    current_z_score
                    >= entry_threshold
                ):

                    position = -1

                    entry_spread = (
                        current_spread
                    )

                    entry_time = (
                        current_time
                    )

                    entry_z_score = (
                        current_z_score
                    )

                    trade_direction = (
                        "Short DK1 / Long DK2"
                    )

                    entry_cost = (
                        transaction_cost_per_leg
                        * 2
                    )

                    transaction_cost += (
                        entry_cost
                    )

                    entry_marker = (
                        current_spread
                    )

                # ----------------------------------------------
                # Spread unusually low
                #
                # Long DK1 / Short DK2
                # ----------------------------------------------

                elif (
                    current_z_score
                    <= -entry_threshold
                ):

                    position = 1

                    entry_spread = (
                        current_spread
                    )

                    entry_time = (
                        current_time
                    )

                    entry_z_score = (
                        current_z_score
                    )

                    trade_direction = (
                        "Long DK1 / Short DK2"
                    )

                    entry_cost = (
                        transaction_cost_per_leg
                        * 2
                    )

                    transaction_cost += (
                        entry_cost
                    )

                    entry_marker = (
                        current_spread
                    )

            # --------------------------------------------------
            # Net PnL
            # --------------------------------------------------

            net_pnl = (
                gross_pnl
                - transaction_cost
            )

            positions.append(
                position
            )

            gross_pnl_values.append(
                gross_pnl
            )

            transaction_cost_values.append(
                transaction_cost
            )

            net_pnl_values.append(
                net_pnl
            )

            entry_markers.append(
                entry_marker
            )

            exit_markers.append(
                exit_marker
            )

        # ==================================================
        # Force-close any open position at the end
        # ==================================================

        if (
            position != 0
            and
            entry_spread is not None
        ):

            final_index = (
                len(data)
                - 1
            )

            final_spread = (
                data.loc[
                    final_index,
                    "Spread"
                ]
            )

            final_time = (
                data.loc[
                    final_index,
                    "TimeDK"
                ]
            )

            final_z_score = (
                data.loc[
                    final_index,
                    "ZScore"
                ]
            )

            exit_cost = (
                transaction_cost_per_leg
                * 2
            )

            transaction_cost_values[
                final_index
            ] += (
                exit_cost
            )

            net_pnl_values[
                final_index
            ] -= (
                exit_cost
            )

            if position == 1:

                trade_gross_pnl = (
                    final_spread
                    - entry_spread
                )

            else:

                trade_gross_pnl = (
                    entry_spread
                    - final_spread
                )

            total_trade_cost = (
                transaction_cost_per_leg
                * 4
            )

            trade_net_pnl = (
                trade_gross_pnl
                - total_trade_cost
            )

            trades.append(
                {
                    "EntryTime":
                        entry_time,

                    "ExitTime":
                        final_time,

                    "Direction":
                        trade_direction,

                    "EntrySpread":
                        entry_spread,

                    "ExitSpread":
                        final_spread,

                    "EntryZScore":
                        entry_z_score,

                    "ExitZScore":
                        final_z_score,

                    "GrossPnL":
                        trade_gross_pnl,

                    "TransactionCost":
                        total_trade_cost,

                    "NetPnL":
                        trade_net_pnl
                }
            )

            exit_markers[
                final_index
            ] = (
                final_spread
            )

            positions[
                final_index
            ] = 0

        # ==================================================
        # Add backtest columns
        # ==================================================

        data[
            "Position"
        ] = positions

        data[
            "GrossPnL"
        ] = gross_pnl_values

        data[
            "TransactionCost"
        ] = transaction_cost_values

        data[
            "NetPnL"
        ] = net_pnl_values

        data[
            "CumulativeGrossPnL"
        ] = (
            data[
                "GrossPnL"
            ].cumsum()
        )

        data[
            "CumulativeNetPnL"
        ] = (
            data[
                "NetPnL"
            ].cumsum()
        )

        data[
            "EntryMarker"
        ] = entry_markers

        data[
            "ExitMarker"
        ] = exit_markers

        # ==================================================
        # Trade dataframe
        # ==================================================

        trades_dataframe = (
            pd.DataFrame(
                trades
            )
        )

        # ==================================================
        # Performance statistics
        # ==================================================

        statistics = (
            self._calculate_statistics(
                data,
                trades_dataframe
            )
        )

        return {
            "success":
                True,

            "message":
                "Backtest completed successfully.",

            "backtest_data":
                data,

            "trades":
                trades_dataframe,

            "statistics":
                statistics,

            "entry_threshold":
                entry_threshold,

            "exit_threshold":
                exit_threshold,

            "transaction_cost_per_leg":
                transaction_cost_per_leg
        }

    # ==================================================
    # Statistics
    # ==================================================

    def _calculate_statistics(
        self,
        backtest_data,
        trades_dataframe
    ):

        total_gross_pnl = (
            backtest_data[
                "GrossPnL"
            ].sum()
        )

        total_transaction_cost = (
            backtest_data[
                "TransactionCost"
            ].sum()
        )

        total_net_pnl = (
            backtest_data[
                "NetPnL"
            ].sum()
        )

        number_of_trades = (
            len(
                trades_dataframe
            )
        )

        # --------------------------------------------------
        # Trade statistics
        # --------------------------------------------------

        if (
            number_of_trades
            > 0
        ):

            winning_trades = (
                trades_dataframe[
                    trades_dataframe[
                        "NetPnL"
                    ] > 0
                ]
            )

            losing_trades = (
                trades_dataframe[
                    trades_dataframe[
                        "NetPnL"
                    ] < 0
                ]
            )

            win_rate = (
                len(
                    winning_trades
                )
                / number_of_trades
            ) * 100

            average_trade = (
                trades_dataframe[
                    "NetPnL"
                ].mean()
            )

            average_winner = (
                winning_trades[
                    "NetPnL"
                ].mean()
                if not winning_trades.empty
                else 0
            )

            average_loser = (
                losing_trades[
                    "NetPnL"
                ].mean()
                if not losing_trades.empty
                else 0
            )

            best_trade = (
                trades_dataframe[
                    "NetPnL"
                ].max()
            )

            worst_trade = (
                trades_dataframe[
                    "NetPnL"
                ].min()
            )

        else:

            win_rate = 0
            average_trade = 0
            average_winner = 0
            average_loser = 0
            best_trade = 0
            worst_trade = 0

        # --------------------------------------------------
        # Sharpe proxy
        #
        # Uses hourly PnL observations.
        # This is a simplified research metric because
        # the strategy does not yet define invested capital.
        # --------------------------------------------------

        pnl_series = (
            backtest_data[
                "NetPnL"
            ]
        )

        pnl_mean = (
            pnl_series.mean()
        )

        pnl_std = (
            pnl_series.std()
        )

        if (
            pd.isna(
                pnl_std
            )
            or
            pnl_std == 0
        ):

            sharpe_proxy = 0

        else:

            periods_per_year = (
                24
                * 365
            )

            sharpe_proxy = (
                pnl_mean
                / pnl_std
            ) * np.sqrt(
                periods_per_year
            )

        # --------------------------------------------------
        # Maximum drawdown
        # --------------------------------------------------

        cumulative_pnl = (
            backtest_data[
                "CumulativeNetPnL"
            ]
        )

        running_peak = (
            cumulative_pnl.cummax()
        )

        drawdown = (
            cumulative_pnl
            - running_peak
        )

        if drawdown.empty:

            maximum_drawdown = 0

        else:

            maximum_drawdown = (
                drawdown.min()
            )

        # --------------------------------------------------
        # Profit factor
        # --------------------------------------------------

        if (
            number_of_trades
            > 0
        ):

            gross_profit = (
                trades_dataframe.loc[
                    trades_dataframe[
                        "NetPnL"
                    ] > 0,
                    "NetPnL"
                ].sum()
            )

            gross_loss = abs(
                trades_dataframe.loc[
                    trades_dataframe[
                        "NetPnL"
                    ] < 0,
                    "NetPnL"
                ].sum()
            )

            if gross_loss > 0:

                profit_factor = (
                    gross_profit
                    / gross_loss
                )

            elif gross_profit > 0:

                profit_factor = (
                    float(
                        "inf"
                    )
                )

            else:

                profit_factor = 0

        else:

            profit_factor = 0

        return {
            "total_gross_pnl":
                total_gross_pnl,

            "total_transaction_cost":
                total_transaction_cost,

            "total_net_pnl":
                total_net_pnl,

            "number_of_trades":
                number_of_trades,

            "win_rate":
                win_rate,

            "average_trade":
                average_trade,

            "average_winner":
                average_winner,

            "average_loser":
                average_loser,

            "best_trade":
                best_trade,

            "worst_trade":
                worst_trade,

            "sharpe_proxy":
                sharpe_proxy,

            "maximum_drawdown":
                maximum_drawdown,

            "profit_factor":
                profit_factor
        }

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

            "backtest_data":
                pd.DataFrame(),

            "trades":
                pd.DataFrame(),

            "statistics":
                {
                    "total_gross_pnl": 0,
                    "total_transaction_cost": 0,
                    "total_net_pnl": 0,
                    "number_of_trades": 0,
                    "win_rate": 0,
                    "average_trade": 0,
                    "average_winner": 0,
                    "average_loser": 0,
                    "best_trade": 0,
                    "worst_trade": 0,
                    "sharpe_proxy": 0,
                    "maximum_drawdown": 0,
                    "profit_factor": 0
                },

            "entry_threshold":
                0,

            "exit_threshold":
                0,

            "transaction_cost_per_leg":
                0
        }