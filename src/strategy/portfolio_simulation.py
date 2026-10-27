import numpy as np
import pandas as pd


class CapitalPortfolioSimulator:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.spread_analyzer = spread_analyzer
        self.backtester = backtester

    # ==================================================
    # Main simulation
    # ==================================================

    def run_simulation(
        self,
        spread_dataframe,
        window=48,
        entry_threshold=2.0,
        exit_threshold=0.5,
        transaction_cost_per_leg=0.25,
        starting_capital=100000,
        risk_per_trade_percent=1.0,
        max_position_mwh=10.0,
        max_gross_exposure_percent=50.0
    ):

        if spread_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        starting_capital = max(
            100.0,
            float(
                starting_capital
            )
        )

        risk_per_trade_percent = max(
            0.1,
            float(
                risk_per_trade_percent
            )
        )

        max_position_mwh = max(
            0.1,
            float(
                max_position_mwh
            )
        )

        max_gross_exposure_percent = max(
            1.0,
            float(
                max_gross_exposure_percent
            )
        )

        transaction_cost_per_leg = max(
            0.0,
            float(
                transaction_cost_per_leg
            )
        )

        window = max(
            5,
            int(
                window
            )
        )

        # ==================================================
        # Generate strategy signals
        # ==================================================

        signal_data = (
            self.spread_analyzer
            .calculate_signals(
                spread_dataframe,
                window=window,
                threshold=entry_threshold
            )
        )

        if signal_data.empty:

            return self._empty_result(
                "No strategy signals could be generated."
            )

        # ==================================================
        # Standard strategy backtest
        # ==================================================

        backtest = (
            self.backtester
            .run_backtest(
                signal_data,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost_per_leg=(
                    transaction_cost_per_leg
                )
            )
        )

        if not backtest[
            "success"
        ]:

            return self._empty_result(
                backtest[
                    "message"
                ]
            )

        trades = (
            backtest[
                "trades"
            ]
            .copy()
        )

        if trades.empty:

            return self._no_trades_result(
                starting_capital,
                window,
                entry_threshold,
                exit_threshold,
                transaction_cost_per_leg,
                risk_per_trade_percent,
                max_position_mwh,
                max_gross_exposure_percent
            )

        # ==================================================
        # Prepare entry data lookup
        # ==================================================

        market_data = (
            signal_data.copy()
        )

        market_data[
            "TimeDK"
        ] = pd.to_datetime(
            market_data[
                "TimeDK"
            ]
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

        market_lookup = (
            market_data.set_index(
                "TimeDK"
            )
        )

        # ==================================================
        # Capital simulation
        # ==================================================

        current_capital = (
            starting_capital
        )

        simulated_trades = []

        equity_times = []

        equity_values = []

        first_trade_time = (
            trades[
                "EntryTime"
            ].iloc[
                0
            ]
        )

        equity_times.append(
            first_trade_time
        )

        equity_values.append(
            starting_capital
        )

        # ==================================================
        # Process each completed spread trade
        # ==================================================

        for _, trade in (
            trades.iterrows()
        ):

            if current_capital <= 0:

                break

            entry_time = (
                trade[
                    "EntryTime"
                ]
            )

            if (
                entry_time
                not in market_lookup.index
            ):

                continue

            entry_row = (
                market_lookup.loc[
                    entry_time
                ]
            )

            # --------------------------------------------------
            # In case duplicate timestamps ever appear,
            # take the first matching row.
            # --------------------------------------------------

            if isinstance(
                entry_row,
                pd.DataFrame
            ):

                entry_row = (
                    entry_row.iloc[
                        0
                    ]
                )

            rolling_std = (
                entry_row.get(
                    "RollingStd",
                    np.nan
                )
            )

            dk1_price = float(
                entry_row.get(
                    "DK1Price",
                    0
                )
            )

            dk2_price = float(
                entry_row.get(
                    "DK2Price",
                    0
                )
            )

            # ==================================================
            # Risk budget
            # ==================================================

            capital_before = (
                current_capital
            )

            risk_budget = (
                capital_before
                * (
                    risk_per_trade_percent
                    / 100
                )
            )

            # --------------------------------------------------
            # Estimated adverse spread move
            #
            # 2 rolling standard deviations is used as
            # a simple volatility-risk estimate.
            #
            # It is NOT a guaranteed maximum loss.
            # --------------------------------------------------

            if (
                pd.isna(
                    rolling_std
                )
                or
                rolling_std <= 0
            ):

                spread_risk = 1.0

            else:

                spread_risk = (
                    2.0
                    * abs(
                        float(
                            rolling_std
                        )
                    )
                )

            round_trip_cost_per_mwh = (
                transaction_cost_per_leg
                * 4
            )

            estimated_risk_per_mwh = max(
                1.0,
                (
                    spread_risk
                    + round_trip_cost_per_mwh
                )
            )

            # ==================================================
            # Risk-based position size
            # ==================================================

            risk_based_size = (
                risk_budget
                / estimated_risk_per_mwh
            )

            # ==================================================
            # Gross exposure limit
            #
            # Each spread position has two legs:
            # DK1 and DK2.
            # ==================================================

            gross_price_per_mwh = (
                abs(
                    dk1_price
                )
                + abs(
                    dk2_price
                )
            )

            maximum_gross_exposure = (
                capital_before
                * (
                    max_gross_exposure_percent
                    / 100
                )
            )

            if gross_price_per_mwh > 0:

                exposure_based_size = (
                    maximum_gross_exposure
                    / gross_price_per_mwh
                )

            else:

                exposure_based_size = (
                    max_position_mwh
                )

            # ==================================================
            # Final position size
            # ==================================================

            position_size = min(
                risk_based_size,
                exposure_based_size,
                max_position_mwh
            )

            position_size = max(
                0,
                position_size
            )

            # ==================================================
            # Capital PnL
            # ==================================================

            gross_pnl_eur = (
                float(
                    trade[
                        "GrossPnL"
                    ]
                )
                * position_size
            )

            transaction_cost_eur = (
                float(
                    trade[
                        "TransactionCost"
                    ]
                )
                * position_size
            )

            net_pnl_eur = (
                float(
                    trade[
                        "NetPnL"
                    ]
                )
                * position_size
            )

            current_capital = (
                capital_before
                + net_pnl_eur
            )

            # ==================================================
            # Exposure and leverage
            # ==================================================

            gross_exposure = (
                gross_price_per_mwh
                * position_size
            )

            if capital_before > 0:

                leverage = (
                    gross_exposure
                    / capital_before
                )

                trade_return_percent = (
                    net_pnl_eur
                    / capital_before
                ) * 100

                estimated_risk_percent = (
                    (
                        estimated_risk_per_mwh
                        * position_size
                    )
                    / capital_before
                ) * 100

            else:

                leverage = 0
                trade_return_percent = 0
                estimated_risk_percent = 0

            # ==================================================
            # Determine what constrained the position
            # ==================================================

            size_candidates = {
                "Risk Budget":
                    risk_based_size,

                "Max Position":
                    max_position_mwh,

                "Exposure Limit":
                    exposure_based_size
            }

            binding_constraint = min(
                size_candidates,
                key=size_candidates.get
            )

            simulated_trades.append(
                {
                    "EntryTime":
                        trade[
                            "EntryTime"
                        ],

                    "ExitTime":
                        trade[
                            "ExitTime"
                        ],

                    "Direction":
                        trade[
                            "Direction"
                        ],

                    "EntrySpread":
                        trade[
                            "EntrySpread"
                        ],

                    "ExitSpread":
                        trade[
                            "ExitSpread"
                        ],

                    "PositionSizeMWh":
                        position_size,

                    "RiskBudgetEUR":
                        risk_budget,

                    "EstimatedRiskPerMWh":
                        estimated_risk_per_mwh,

                    "EstimatedRiskPercent":
                        estimated_risk_percent,

                    "GrossExposureEUR":
                        gross_exposure,

                    "Leverage":
                        leverage,

                    "GrossPnLEUR":
                        gross_pnl_eur,

                    "TransactionCostEUR":
                        transaction_cost_eur,

                    "NetPnLEUR":
                        net_pnl_eur,

                    "TradeReturnPercent":
                        trade_return_percent,

                    "CapitalBefore":
                        capital_before,

                    "CapitalAfter":
                        current_capital,

                    "BindingConstraint":
                        binding_constraint
                }
            )

            equity_times.append(
                trade[
                    "ExitTime"
                ]
            )

            equity_values.append(
                current_capital
            )

        # ==================================================
        # Simulation dataframe
        # ==================================================

        simulation_data = (
            pd.DataFrame(
                simulated_trades
            )
        )

        if simulation_data.empty:

            return self._empty_result(
                (
                    "The portfolio simulation could not "
                    "generate any valid sized trades."
                )
            )

        # ==================================================
        # Equity curve
        # ==================================================

        equity_curve = pd.DataFrame(
            {
                "TimeDK":
                    equity_times,

                "PortfolioValue":
                    equity_values
            }
        )

        equity_curve = (
            equity_curve.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        equity_curve[
            "RunningPeak"
        ] = (
            equity_curve[
                "PortfolioValue"
            ].cummax()
        )

        equity_curve[
            "DrawdownEUR"
        ] = (
            equity_curve[
                "PortfolioValue"
            ]
            - equity_curve[
                "RunningPeak"
            ]
        )

        equity_curve[
            "DrawdownPercent"
        ] = np.where(
            equity_curve[
                "RunningPeak"
            ] != 0,
            (
                equity_curve[
                    "DrawdownEUR"
                ]
                / equity_curve[
                    "RunningPeak"
                ]
            ) * 100,
            0
        )

        # ==================================================
        # Statistics
        # ==================================================

        ending_capital = (
            simulation_data[
                "CapitalAfter"
            ].iloc[
                -1
            ]
        )

        total_pnl = (
            ending_capital
            - starting_capital
        )

        total_return_percent = (
            total_pnl
            / starting_capital
        ) * 100

        winning_trades = (
            simulation_data[
                simulation_data[
                    "NetPnLEUR"
                ] > 0
            ]
        )

        win_rate = (
            len(
                winning_trades
            )
            / len(
                simulation_data
            )
        ) * 100

        average_position_size = (
            simulation_data[
                "PositionSizeMWh"
            ].mean()
        )

        maximum_position_size = (
            simulation_data[
                "PositionSizeMWh"
            ].max()
        )

        average_gross_exposure = (
            simulation_data[
                "GrossExposureEUR"
            ].mean()
        )

        peak_gross_exposure = (
            simulation_data[
                "GrossExposureEUR"
            ].max()
        )

        maximum_leverage = (
            simulation_data[
                "Leverage"
            ].max()
        )

        average_leverage = (
            simulation_data[
                "Leverage"
            ].mean()
        )

        maximum_drawdown_percent = (
            equity_curve[
                "DrawdownPercent"
            ].min()
        )

        maximum_drawdown_eur = (
            equity_curve[
                "DrawdownEUR"
            ].min()
        )

        average_trade_pnl = (
            simulation_data[
                "NetPnLEUR"
            ].mean()
        )

        best_trade = (
            simulation_data[
                "NetPnLEUR"
            ].max()
        )

        worst_trade = (
            simulation_data[
                "NetPnLEUR"
            ].min()
        )

        total_costs = (
            simulation_data[
                "TransactionCostEUR"
            ].sum()
        )

        average_risk_utilization = (
            simulation_data[
                "EstimatedRiskPercent"
            ].mean()
        )

        return {
            "success":
                True,

            "message":
                (
                    "Capital-based portfolio "
                    "simulation completed."
                ),

            "simulation_data":
                simulation_data,

            "equity_curve":
                equity_curve,

            "statistics":
                {
                    "starting_capital":
                        starting_capital,

                    "ending_capital":
                        ending_capital,

                    "total_pnl":
                        total_pnl,

                    "total_return_percent":
                        total_return_percent,

                    "number_of_trades":
                        len(
                            simulation_data
                        ),

                    "win_rate":
                        win_rate,

                    "average_trade_pnl":
                        average_trade_pnl,

                    "best_trade":
                        best_trade,

                    "worst_trade":
                        worst_trade,

                    "total_costs":
                        total_costs,

                    "average_position_size":
                        average_position_size,

                    "maximum_position_size":
                        maximum_position_size,

                    "average_gross_exposure":
                        average_gross_exposure,

                    "peak_gross_exposure":
                        peak_gross_exposure,

                    "average_leverage":
                        average_leverage,

                    "maximum_leverage":
                        maximum_leverage,

                    "maximum_drawdown_percent":
                        maximum_drawdown_percent,

                    "maximum_drawdown_eur":
                        maximum_drawdown_eur,

                    "average_risk_utilization":
                        average_risk_utilization
                },

            "strategy_parameters":
                {
                    "window":
                        window,

                    "entry_threshold":
                        entry_threshold,

                    "exit_threshold":
                        exit_threshold,

                    "transaction_cost_per_leg":
                        transaction_cost_per_leg
                },

            "portfolio_parameters":
                {
                    "starting_capital":
                        starting_capital,

                    "risk_per_trade_percent":
                        risk_per_trade_percent,

                    "max_position_mwh":
                        max_position_mwh,

                    "max_gross_exposure_percent":
                        max_gross_exposure_percent
                }
        }

    # ==================================================
    # No trades
    # ==================================================

    def _no_trades_result(
        self,
        starting_capital,
        window,
        entry_threshold,
        exit_threshold,
        transaction_cost_per_leg,
        risk_per_trade_percent,
        max_position_mwh,
        max_gross_exposure_percent
    ):

        equity_curve = pd.DataFrame(
            {
                "TimeDK":
                    [],

                "PortfolioValue":
                    []
            }
        )

        return {
            "success":
                True,

            "message":
                (
                    "No completed trades were generated "
                    "for the selected strategy parameters."
                ),

            "simulation_data":
                pd.DataFrame(),

            "equity_curve":
                equity_curve,

            "statistics":
                {
                    "starting_capital":
                        starting_capital,

                    "ending_capital":
                        starting_capital,

                    "total_pnl":
                        0,

                    "total_return_percent":
                        0,

                    "number_of_trades":
                        0,

                    "win_rate":
                        0,

                    "average_trade_pnl":
                        0,

                    "best_trade":
                        0,

                    "worst_trade":
                        0,

                    "total_costs":
                        0,

                    "average_position_size":
                        0,

                    "maximum_position_size":
                        0,

                    "average_gross_exposure":
                        0,

                    "peak_gross_exposure":
                        0,

                    "average_leverage":
                        0,

                    "maximum_leverage":
                        0,

                    "maximum_drawdown_percent":
                        0,

                    "maximum_drawdown_eur":
                        0,

                    "average_risk_utilization":
                        0
                },

            "strategy_parameters":
                {
                    "window":
                        window,

                    "entry_threshold":
                        entry_threshold,

                    "exit_threshold":
                        exit_threshold,

                    "transaction_cost_per_leg":
                        transaction_cost_per_leg
                },

            "portfolio_parameters":
                {
                    "starting_capital":
                        starting_capital,

                    "risk_per_trade_percent":
                        risk_per_trade_percent,

                    "max_position_mwh":
                        max_position_mwh,

                    "max_gross_exposure_percent":
                        max_gross_exposure_percent
                }
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

            "simulation_data":
                pd.DataFrame(),

            "equity_curve":
                pd.DataFrame(),

            "statistics":
                {},

            "strategy_parameters":
                {},

            "portfolio_parameters":
                {}
        }