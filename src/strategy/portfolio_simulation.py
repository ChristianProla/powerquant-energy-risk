import numpy as np
import pandas as pd


class CapitalPortfolioSimulator:

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

        # ==================================================
        # Parameter conversion
        # ==================================================

        try:

            starting_capital = float(
                starting_capital
            )

            risk_per_trade_percent = float(
                risk_per_trade_percent
            )

            max_position_mwh = float(
                max_position_mwh
            )

            max_gross_exposure_percent = float(
                max_gross_exposure_percent
            )

            transaction_cost_per_leg = float(
                transaction_cost_per_leg
            )

            entry_threshold = float(
                entry_threshold
            )

            exit_threshold = float(
                exit_threshold
            )

            window = int(
                window
            )

        except (
            TypeError,
            ValueError
        ):

            return self._empty_result(
                "Portfolio parameters must be numeric."
            )

        # ==================================================
        # Parameter validation
        # ==================================================

        if starting_capital <= 0:

            return self._empty_result(
                "Starting capital must be positive."
            )

        if (
            risk_per_trade_percent <= 0
            or
            risk_per_trade_percent > 100
        ):

            return self._empty_result(
                (
                    "Risk per trade must be greater than "
                    "0% and no greater than 100%."
                )
            )

        if max_position_mwh <= 0:

            return self._empty_result(
                "Maximum position must be positive."
            )

        if max_gross_exposure_percent <= 0:

            return self._empty_result(
                "Maximum gross exposure must be positive."
            )

        if transaction_cost_per_leg < 0:

            return self._empty_result(
                "Transaction cost cannot be negative."
            )

        if window < 5:

            return self._empty_result(
                "Signal window must be at least 5 observations."
            )

        if entry_threshold < 0.5:

            return self._empty_result(
                "Entry threshold must be at least 0.5."
            )

        if exit_threshold < 0:

            return self._empty_result(
                "Exit threshold cannot be negative."
            )

        if exit_threshold >= entry_threshold:

            return self._empty_result(
                (
                    "Exit threshold must be lower "
                    "than entry threshold."
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

        signal_data = (
            signal_data.copy()
        )

        signal_data[
            "TimeDK"
        ] = pd.to_datetime(
            signal_data[
                "TimeDK"
            ]
        )

        signal_data = (
            signal_data.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
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
        # Prepare timestamps
        # ==================================================

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

        trades = (
            trades.sort_values(
                [
                    "EntryTime",
                    "ExitTime"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        market_lookup = (
            signal_data.set_index(
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
        # Process each completed trade
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
            # Protect against duplicate market timestamps
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

            dk1_price = (
                entry_row.get(
                    "DK1Price",
                    np.nan
                )
            )

            dk2_price = (
                entry_row.get(
                    "DK2Price",
                    np.nan
                )
            )

            # --------------------------------------------------
            # Price observations are necessary to calculate
            # gross two-leg exposure.
            # --------------------------------------------------

            try:

                dk1_price = float(
                    dk1_price
                )

                dk2_price = float(
                    dk2_price
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            if (
                not np.isfinite(
                    dk1_price
                )
                or
                not np.isfinite(
                    dk2_price
                )
            ):

                continue

            # ==================================================
            # Capital before trade
            # ==================================================

            capital_before = (
                current_capital
            )

            # ==================================================
            # Risk budget
            # ==================================================

            risk_budget = (
                capital_before
                * (
                    risk_per_trade_percent
                    / 100
                )
            )

            # ==================================================
            # Estimated adverse spread move
            #
            # Two rolling standard deviations are used as
            # a simple volatility-risk estimate.
            #
            # This is a research heuristic rather than a
            # guaranteed maximum loss.
            # ==================================================

            try:

                rolling_std_value = float(
                    rolling_std
                )

            except (
                TypeError,
                ValueError
            ):

                rolling_std_value = np.nan

            if (
                not np.isfinite(
                    rolling_std_value
                )
                or
                rolling_std_value <= 0
            ):

                spread_risk = 1.0

            else:

                spread_risk = (
                    2.0
                    * abs(
                        rolling_std_value
                    )
                )

            # --------------------------------------------------
            # One complete spread round trip:
            #
            # Entry DK1
            # Entry DK2
            # Exit DK1
            # Exit DK2
            # --------------------------------------------------

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
            # Gross notional uses absolute prices because a
            # spread consists of one long and one short leg.
            # ==================================================

            gross_price_per_mwh = (
                abs(
                    dk1_price
                )
                +
                abs(
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

                # Both power prices can theoretically be zero.
                # In that situation gross notional does not
                # constrain position size.
                exposure_based_size = (
                    max_position_mwh
                )

            # ==================================================
            # Final position size
            # ==================================================

            size_candidates = {
                "Risk Budget":
                    risk_based_size,

                "Exposure Limit":
                    exposure_based_size,

                "Max Position":
                    max_position_mwh
            }

            binding_constraint = min(
                size_candidates,
                key=size_candidates.get
            )

            position_size = min(
                size_candidates.values()
            )

            position_size = max(
                0.0,
                float(
                    position_size
                )
            )

            if (
                not np.isfinite(
                    position_size
                )
                or
                position_size <= 0
            ):

                continue

            # ==================================================
            # Capital PnL
            # ==================================================

            gross_pnl_per_mwh = float(
                trade[
                    "GrossPnL"
                ]
            )

            transaction_cost_per_mwh = float(
                trade[
                    "TransactionCost"
                ]
            )

            net_pnl_per_mwh = float(
                trade[
                    "NetPnL"
                ]
            )

            gross_pnl_eur = (
                gross_pnl_per_mwh
                * position_size
            )

            transaction_cost_eur = (
                transaction_cost_per_mwh
                * position_size
            )

            net_pnl_eur = (
                net_pnl_per_mwh
                * position_size
            )

            # ==================================================
            # Risk utilization
            # ==================================================

            estimated_risk_eur = (
                estimated_risk_per_mwh
                * position_size
            )

            if capital_before > 0:

                estimated_risk_percent = (
                    estimated_risk_eur
                    / capital_before
                ) * 100

            else:

                estimated_risk_percent = 0

            if risk_budget > 0:

                risk_utilization_percent = (
                    estimated_risk_eur
                    / risk_budget
                ) * 100

            else:

                risk_utilization_percent = 0

            # Small floating point differences should never
            # produce values such as 100.0000000002%.

            risk_utilization_percent = min(
                100.0,
                max(
                    0.0,
                    risk_utilization_percent
                )
            )

            # ==================================================
            # Update capital
            # ==================================================

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

            else:

                leverage = 0
                trade_return_percent = 0

            # ==================================================
            # Store sized trade
            # ==================================================

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

                    "RiskBasedSizeMWh":
                        risk_based_size,

                    "ExposureBasedSizeMWh":
                        exposure_based_size,

                    "MaxPositionMWh":
                        max_position_mwh,

                    "RiskBudgetEUR":
                        risk_budget,

                    "EstimatedRiskPerMWh":
                        estimated_risk_per_mwh,

                    "EstimatedRiskEUR":
                        estimated_risk_eur,

                    "EstimatedRiskPercent":
                        estimated_risk_percent,

                    "RiskUtilizationPercent":
                        risk_utilization_percent,

                    "GrossPricePerMWh":
                        gross_price_per_mwh,

                    "MaximumGrossExposureEUR":
                        maximum_gross_exposure,

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
        #
        # Current methodology records portfolio value at
        # trade exits rather than hourly mark-to-market.
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
            -
            equity_curve[
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
                /
                equity_curve[
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

        average_estimated_risk_percent = (
            simulation_data[
                "EstimatedRiskPercent"
            ].mean()
        )

        average_risk_utilization = (
            simulation_data[
                "RiskUtilizationPercent"
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

                    "average_estimated_risk_percent":
                        average_estimated_risk_percent,

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
                },

            "methodology":
                {
                    "risk_model":
                        "2x rolling spread volatility plus round-trip costs",

                    "equity_curve":
                        "Trade-exit equity, not hourly mark-to-market",

                    "gross_exposure":
                        "Absolute DK1 plus DK2 notional"
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

                    "average_estimated_risk_percent":
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
                },

            "methodology":
                {
                    "risk_model":
                        "2x rolling spread volatility plus round-trip costs",

                    "equity_curve":
                        "Trade-exit equity, not hourly mark-to-market",

                    "gross_exposure":
                        "Absolute DK1 plus DK2 notional"
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
                {},

            "methodology":
                {}
        }