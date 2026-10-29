import numpy as np
import pandas as pd


class PerformanceAttributionAnalyzer:

    # ==================================================
    # Main analysis
    # ==================================================

    def analyze(
        self,
        simulation_data,
        equity_curve,
        starting_capital
    ):

        if (
            simulation_data.empty
            or
            equity_curve.empty
        ):

            return self._empty_result(
                "No portfolio simulation data available."
            )

        trades = (
            simulation_data.copy()
        )

        equity = (
            equity_curve.copy()
        )

        trades[
            "EntryTime"
        ] = pd.to_datetime(
            trades[
                "EntryTime"
            ]
        )

        trades[
            "ExitTime"
        ] = pd.to_datetime(
            trades[
                "ExitTime"
            ]
        )

        equity[
            "TimeDK"
        ] = pd.to_datetime(
            equity[
                "TimeDK"
            ]
        )

        # ==================================================
        # Overall performance
        # ==================================================

        ending_capital = float(
            equity[
                "PortfolioValue"
            ].iloc[
                -1
            ]
        )

        total_net_pnl = (
            trades[
                "NetPnLEUR"
            ].sum()
        )

        total_gross_pnl = (
            trades[
                "GrossPnLEUR"
            ].sum()
        )

        total_costs = (
            trades[
                "TransactionCostEUR"
            ].sum()
        )

        total_return_percent = (
            (
                ending_capital
                - starting_capital
            )
            / starting_capital
        ) * 100

        # ==================================================
        # Trade performance
        # ==================================================

        number_of_trades = len(
            trades
        )

        winning_trades = (
            trades[
                trades[
                    "NetPnLEUR"
                ] > 0
            ]
        )

        losing_trades = (
            trades[
                trades[
                    "NetPnLEUR"
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
            trades[
                "NetPnLEUR"
            ].mean()
        )

        best_trade = (
            trades[
                "NetPnLEUR"
            ].max()
        )

        worst_trade = (
            trades[
                "NetPnLEUR"
            ].min()
        )

        gross_profit = (
            winning_trades[
                "NetPnLEUR"
            ].sum()
        )

        gross_loss = abs(
            losing_trades[
                "NetPnLEUR"
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

        # ==================================================
        # Transaction-cost drag
        # ==================================================

        if total_gross_pnl != 0:

            cost_drag_percent = (
                total_costs
                / abs(
                    total_gross_pnl
                )
            ) * 100

        else:

            cost_drag_percent = 0

        # ==================================================
        # Direction attribution
        # ==================================================

        direction_attribution = (
            self._calculate_direction_attribution(
                trades
            )
        )

        # ==================================================
        # Monthly performance
        # ==================================================

        monthly_performance = (
            self._calculate_monthly_performance(
                trades
            )
        )

        if not monthly_performance.empty:

            best_month_row = (
                monthly_performance.loc[
                    monthly_performance[
                        "MonthlyReturnPercent"
                    ].idxmax()
                ]
            )

            worst_month_row = (
                monthly_performance.loc[
                    monthly_performance[
                        "MonthlyReturnPercent"
                    ].idxmin()
                ]
            )

            best_month = (
                best_month_row[
                    "Month"
                ]
            )

            best_month_return = (
                best_month_row[
                    "MonthlyReturnPercent"
                ]
            )

            worst_month = (
                worst_month_row[
                    "Month"
                ]
            )

            worst_month_return = (
                worst_month_row[
                    "MonthlyReturnPercent"
                ]
            )

            profitable_month_percentage = (
                (
                    monthly_performance[
                        "MonthlyReturnPercent"
                    ] > 0
                ).mean()
                * 100
            )

        else:

            best_month = "N/A"
            best_month_return = 0
            worst_month = "N/A"
            worst_month_return = 0
            profitable_month_percentage = 0

        # ==================================================
        # Drawdown analysis
        # ==================================================

        drawdown_analysis = (
            self._calculate_drawdown_analysis(
                equity
            )
        )

        maximum_drawdown_percent = (
            drawdown_analysis[
                "maximum_drawdown_percent"
            ]
        )

        # ==================================================
        # Risk-adjusted metrics
        # ==================================================

        if (
            maximum_drawdown_percent
            < 0
        ):

            return_to_drawdown = (
                total_return_percent
                / abs(
                    maximum_drawdown_percent
                )
            )

        else:

            return_to_drawdown = 0

        trade_returns = (
            trades[
                "TradeReturnPercent"
            ]
            .dropna()
        )

        if (
            len(
                trade_returns
            ) >= 2
            and
            trade_returns.std()
            > 0
        ):

            trade_sharpe_proxy = (
                trade_returns.mean()
                / trade_returns.std()
            ) * np.sqrt(
                len(
                    trade_returns
                )
            )

        else:

            trade_sharpe_proxy = 0

        # ==================================================
        # Exposure
        # ==================================================

        average_exposure = (
            trades[
                "GrossExposureEUR"
            ].mean()
        )

        peak_exposure = (
            trades[
                "GrossExposureEUR"
            ].max()
        )

        average_position = (
            trades[
                "PositionSizeMWh"
            ].mean()
        )

        maximum_position = (
            trades[
                "PositionSizeMWh"
            ].max()
        )

        average_leverage = (
            trades[
                "Leverage"
            ].mean()
        )

        maximum_leverage = (
            trades[
                "Leverage"
            ].max()
        )

        # ==================================================
        # Binding constraints
        # ==================================================

        constraint_attribution = (
            trades[
                "BindingConstraint"
            ]
            .value_counts()
            .rename_axis(
                "Constraint"
            )
            .reset_index(
                name="Trades"
            )
        )

        constraint_attribution[
            "SharePercent"
        ] = (
            constraint_attribution[
                "Trades"
            ]
            / number_of_trades
        ) * 100

        return {
            "success":
                True,

            "message":
                (
                    "Performance attribution "
                    "completed successfully."
                ),

            "starting_capital":
                starting_capital,

            "ending_capital":
                ending_capital,

            "total_net_pnl":
                total_net_pnl,

            "total_gross_pnl":
                total_gross_pnl,

            "total_costs":
                total_costs,

            "total_return_percent":
                total_return_percent,

            "cost_drag_percent":
                cost_drag_percent,

            "number_of_trades":
                number_of_trades,

            "win_rate":
                win_rate,

            "average_trade":
                average_trade,

            "best_trade":
                best_trade,

            "worst_trade":
                worst_trade,

            "profit_factor":
                profit_factor,

            "return_to_drawdown":
                return_to_drawdown,

            "trade_sharpe_proxy":
                trade_sharpe_proxy,

            "maximum_drawdown_percent":
                maximum_drawdown_percent,

            "best_month":
                best_month,

            "best_month_return":
                best_month_return,

            "worst_month":
                worst_month,

            "worst_month_return":
                worst_month_return,

            "profitable_month_percentage":
                profitable_month_percentage,

            "average_exposure":
                average_exposure,

            "peak_exposure":
                peak_exposure,

            "average_position":
                average_position,

            "maximum_position":
                maximum_position,

            "average_leverage":
                average_leverage,

            "maximum_leverage":
                maximum_leverage,

            "direction_attribution":
                direction_attribution,

            "monthly_performance":
                monthly_performance,

            "drawdown_analysis":
                drawdown_analysis,

            "constraint_attribution":
                constraint_attribution
        }

    # ==================================================
    # Direction attribution
    # ==================================================

    def _calculate_direction_attribution(
        self,
        trades
    ):

        results = []

        directions = (
            trades[
                "Direction"
            ].dropna().unique()
        )

        total_net_pnl = (
            trades[
                "NetPnLEUR"
            ].sum()
        )

        for direction in directions:

            direction_data = (
                trades[
                    trades[
                        "Direction"
                    ] == direction
                ]
            )

            trade_count = len(
                direction_data
            )

            net_pnl = (
                direction_data[
                    "NetPnLEUR"
                ].sum()
            )

            gross_pnl = (
                direction_data[
                    "GrossPnLEUR"
                ].sum()
            )

            costs = (
                direction_data[
                    "TransactionCostEUR"
                ].sum()
            )

            wins = (
                direction_data[
                    direction_data[
                        "NetPnLEUR"
                    ] > 0
                ]
            )

            win_rate = (
                len(
                    wins
                )
                / trade_count
            ) * 100

            average_trade = (
                direction_data[
                    "NetPnLEUR"
                ].mean()
            )

            if total_net_pnl != 0:

                contribution_percent = (
                    net_pnl
                    / total_net_pnl
                ) * 100

            else:

                contribution_percent = 0

            results.append(
                {
                    "Direction":
                        direction,

                    "Trades":
                        trade_count,

                    "NetPnL":
                        net_pnl,

                    "GrossPnL":
                        gross_pnl,

                    "Costs":
                        costs,

                    "WinRate":
                        win_rate,

                    "AverageTrade":
                        average_trade,

                    "ContributionPercent":
                        contribution_percent
                }
            )

        return (
            pd.DataFrame(
                results
            )
            .sort_values(
                by="NetPnL",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # Monthly performance
    # ==================================================

    def _calculate_monthly_performance(
        self,
        trades
    ):

        if trades.empty:

            return pd.DataFrame()

        monthly_data = (
            trades.copy()
        )

        monthly_data[
            "MonthPeriod"
        ] = (
            monthly_data[
                "ExitTime"
            ].dt.to_period(
                "M"
            )
        )

        monthly_rows = []

        for (
            month,
            group
        ) in monthly_data.groupby(
            "MonthPeriod"
        ):

            group = (
                group.sort_values(
                    "ExitTime"
                )
            )

            starting_capital = float(
                group[
                    "CapitalBefore"
                ].iloc[
                    0
                ]
            )

            ending_capital = float(
                group[
                    "CapitalAfter"
                ].iloc[
                    -1
                ]
            )

            net_pnl = (
                group[
                    "NetPnLEUR"
                ].sum()
            )

            costs = (
                group[
                    "TransactionCostEUR"
                ].sum()
            )

            if starting_capital != 0:

                monthly_return = (
                    net_pnl
                    / starting_capital
                ) * 100

            else:

                monthly_return = 0

            monthly_rows.append(
                {
                    "Month":
                        str(
                            month
                        ),

                    "NetPnL":
                        net_pnl,

                    "MonthlyReturnPercent":
                        monthly_return,

                    "Trades":
                        len(
                            group
                        ),

                    "Costs":
                        costs,

                    "StartingCapital":
                        starting_capital,

                    "EndingCapital":
                        ending_capital
                }
            )

        return (
            pd.DataFrame(
                monthly_rows
            )
            .sort_values(
                "Month"
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # Drawdown analysis
    # ==================================================

    def _calculate_drawdown_analysis(
        self,
        equity
    ):

        data = (
            equity.copy()
        )

        if (
            "RunningPeak"
            not in data.columns
        ):

            data[
                "RunningPeak"
            ] = (
                data[
                    "PortfolioValue"
                ].cummax()
            )

        if (
            "DrawdownEUR"
            not in data.columns
        ):

            data[
                "DrawdownEUR"
            ] = (
                data[
                    "PortfolioValue"
                ]
                - data[
                    "RunningPeak"
                ]
            )

        if (
            "DrawdownPercent"
            not in data.columns
        ):

            data[
                "DrawdownPercent"
            ] = np.where(
                data[
                    "RunningPeak"
                ] != 0,
                (
                    data[
                        "DrawdownEUR"
                    ]
                    / data[
                        "RunningPeak"
                    ]
                ) * 100,
                0
            )

        trough_index = (
            data[
                "DrawdownPercent"
            ].idxmin()
        )

        trough_row = (
            data.loc[
                trough_index
            ]
        )

        maximum_drawdown_percent = (
            trough_row[
                "DrawdownPercent"
            ]
        )

        maximum_drawdown_eur = (
            trough_row[
                "DrawdownEUR"
            ]
        )

        before_trough = (
            data.loc[
                :trough_index
            ]
        )

        peak_index = (
            before_trough[
                "PortfolioValue"
            ].idxmax()
        )

        peak_row = (
            data.loc[
                peak_index
            ]
        )

        peak_value = (
            peak_row[
                "PortfolioValue"
            ]
        )

        after_trough = (
            data.loc[
                trough_index:
            ]
        )

        recovered = (
            after_trough[
                after_trough[
                    "PortfolioValue"
                ] >= peak_value
            ]
        )

        if recovered.empty:

            recovery_time = None
            recovery_status = (
                "Not Recovered"
            )

        else:

            recovery_time = (
                recovered[
                    "TimeDK"
                ].iloc[
                    0
                ]
            )

            recovery_status = (
                "Recovered"
            )

        return {
            "maximum_drawdown_percent":
                maximum_drawdown_percent,

            "maximum_drawdown_eur":
                maximum_drawdown_eur,

            "peak_time":
                peak_row[
                    "TimeDK"
                ],

            "peak_value":
                peak_value,

            "trough_time":
                trough_row[
                    "TimeDK"
                ],

            "trough_value":
                trough_row[
                    "PortfolioValue"
                ],

            "recovery_time":
                recovery_time,

            "recovery_status":
                recovery_status
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

            "starting_capital":
                0,

            "ending_capital":
                0,

            "total_net_pnl":
                0,

            "total_gross_pnl":
                0,

            "total_costs":
                0,

            "total_return_percent":
                0,

            "cost_drag_percent":
                0,

            "number_of_trades":
                0,

            "win_rate":
                0,

            "average_trade":
                0,

            "best_trade":
                0,

            "worst_trade":
                0,

            "profit_factor":
                0,

            "return_to_drawdown":
                0,

            "trade_sharpe_proxy":
                0,

            "maximum_drawdown_percent":
                0,

            "best_month":
                "N/A",

            "best_month_return":
                0,

            "worst_month":
                "N/A",

            "worst_month_return":
                0,

            "profitable_month_percentage":
                0,

            "average_exposure":
                0,

            "peak_exposure":
                0,

            "average_position":
                0,

            "maximum_position":
                0,

            "average_leverage":
                0,

            "maximum_leverage":
                0,

            "direction_attribution":
                pd.DataFrame(),

            "monthly_performance":
                pd.DataFrame(),

            "drawdown_analysis":
                {},

            "constraint_attribution":
                pd.DataFrame()
        }