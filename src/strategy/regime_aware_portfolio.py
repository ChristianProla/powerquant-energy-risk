import numpy as np
import pandas as pd

from src.strategy.portfolio_simulation import (
    CapitalPortfolioSimulator
)

from src.strategy.regime_detection import (
    MarketRegimeDetector
)


class RegimeAwarePortfolioSimulator:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.base_simulator = (
            CapitalPortfolioSimulator(
                spread_analyzer,
                backtester
            )
        )

        self.regime_detector = (
            MarketRegimeDetector()
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
        # First run the ordinary static-risk portfolio
        #
        # This gives us:
        # - completed strategy trades
        # - per-trade volatility risk
        # - exposure information
        #
        # It also provides the benchmark we compare against.
        # ==================================================

        base_results = (
            self.base_simulator
            .run_simulation(
                spread_dataframe,
                window=window,
                entry_threshold=(
                    entry_threshold
                ),
                exit_threshold=(
                    exit_threshold
                ),
                transaction_cost_per_leg=(
                    transaction_cost_per_leg
                ),
                starting_capital=(
                    starting_capital
                ),
                risk_per_trade_percent=(
                    risk_per_trade_percent
                ),
                max_position_mwh=(
                    max_position_mwh
                ),
                max_gross_exposure_percent=(
                    max_gross_exposure_percent
                )
            )
        )

        if not base_results[
            "success"
        ]:

            return self._empty_result(
                base_results[
                    "message"
                ]
            )

        base_trades = (
            base_results[
                "simulation_data"
            ].copy()
        )

        if base_trades.empty:

            return self._empty_result(
                (
                    "No completed strategy trades "
                    "were available."
                )
            )

        # ==================================================
        # Build market regimes
        # ==================================================

        regime_results = (
            self.regime_detector
            .analyze(
                spread_dataframe,
                pd.DataFrame(),
                signal_window=window
            )
        )

        if not regime_results[
            "success"
        ]:

            return self._empty_result(
                regime_results[
                    "message"
                ]
            )

        regime_data = (
            regime_results[
                "regime_data"
            ].copy()
        )

        if regime_data.empty:

            return self._empty_result(
                "No regime data could be generated."
            )

        # ==================================================
        # Match every trade with the regime known at entry
        # ==================================================

        base_trades[
            "EntryTime"
        ] = pd.to_datetime(
            base_trades[
                "EntryTime"
            ]
        )

        base_trades[
            "ExitTime"
        ] = pd.to_datetime(
            base_trades[
                "ExitTime"
            ]
        )

        regime_data[
            "TimeDK"
        ] = pd.to_datetime(
            regime_data[
                "TimeDK"
            ]
        )

        base_trades = (
            base_trades.sort_values(
                "EntryTime"
            )
        )

        regime_lookup = (
            regime_data[
                [
                    "TimeDK",
                    "VolatilityRegime",
                    "StructureRegime",
                    "SpreadState",
                    "RelativePriceState",
                    "RollingSpreadVolatility",
                    "TrendStrength",
                    "ZScore"
                ]
            ]
            .sort_values(
                "TimeDK"
            )
        )

        trades = (
            pd.merge_asof(
                base_trades,
                regime_lookup,
                left_on="EntryTime",
                right_on="TimeDK",
                direction="backward"
            )
        )

        # ==================================================
        # Sequential regime-aware simulation
        #
        # IMPORTANT:
        #
        # Historical performance for a regime is updated
        # only AFTER each trade outcome.
        #
        # Future trades therefore never influence earlier
        # position sizing.
        # ==================================================

        current_capital = float(
            starting_capital
        )

        regime_history = {}

        simulated_trades = []

        equity_times = [
            trades[
                "EntryTime"
            ].iloc[
                0
            ]
        ]

        equity_values = [
            current_capital
        ]

        for _, trade in (
            trades.iterrows()
        ):

            if current_capital <= 0:

                break

            original_size = float(
                trade[
                    "PositionSizeMWh"
                ]
            )

            if original_size <= 0:

                continue

            # ==================================================
            # Recover per-MWh trade economics
            # ==================================================

            gross_pnl_per_mwh = (
                float(
                    trade[
                        "GrossPnLEUR"
                    ]
                )
                / original_size
            )

            cost_per_mwh = (
                float(
                    trade[
                        "TransactionCostEUR"
                    ]
                )
                / original_size
            )

            net_pnl_per_mwh = (
                float(
                    trade[
                        "NetPnLEUR"
                    ]
                )
                / original_size
            )

            gross_price_per_mwh = (
                float(
                    trade[
                        "GrossExposureEUR"
                    ]
                )
                / original_size
            )

            estimated_risk_per_mwh = float(
                trade[
                    "EstimatedRiskPerMWh"
                ]
            )

            # ==================================================
            # Current regime
            # ==================================================

            volatility_regime = (
                trade.get(
                    "VolatilityRegime",
                    "Unknown"
                )
            )

            structure_regime = (
                trade.get(
                    "StructureRegime",
                    "Unknown"
                )
            )

            spread_state = (
                trade.get(
                    "SpreadState",
                    "Unknown"
                )
            )

            if pd.isna(
                volatility_regime
            ):

                volatility_regime = (
                    "Unknown"
                )

            if pd.isna(
                structure_regime
            ):

                structure_regime = (
                    "Unknown"
                )

            if pd.isna(
                spread_state
            ):

                spread_state = (
                    "Unknown"
                )

            regime_key = (
                volatility_regime,
                structure_regime
            )

            # ==================================================
            # Structural regime multiplier
            #
            # These are conservative predefined risk controls.
            # They do not use future strategy performance.
            # ==================================================

            structural_multiplier = (
                self._get_structural_multiplier(
                    volatility_regime,
                    structure_regime
                )
            )

            # ==================================================
            # Adaptive multiplier
            #
            # Uses only PREVIOUS completed signal outcomes
            # from the same regime.
            # ==================================================

            previous_outcomes = (
                regime_history.get(
                    regime_key,
                    []
                )
            )

            (
                adaptive_multiplier,
                historical_regime_label,
                historical_trade_count,
                historical_average_pnl,
                historical_win_rate
            ) = self._get_adaptive_multiplier(
                previous_outcomes
            )

            # ==================================================
            # Final multiplier
            #
            # Regime-aware sizing never increases risk above
            # the user's original risk-per-trade setting.
            # ==================================================

            final_multiplier = min(
                structural_multiplier,
                adaptive_multiplier
            )

            final_multiplier = max(
                0,
                min(
                    1,
                    final_multiplier
                )
            )

            effective_risk_percent = (
                float(
                    risk_per_trade_percent
                )
                * final_multiplier
            )

            capital_before = (
                current_capital
            )

            base_risk_budget = (
                capital_before
                * (
                    float(
                        risk_per_trade_percent
                    )
                    / 100
                )
            )

            adjusted_risk_budget = (
                base_risk_budget
                * final_multiplier
            )

            # ==================================================
            # Risk-based position size
            # ==================================================

            if estimated_risk_per_mwh > 0:

                risk_based_size = (
                    adjusted_risk_budget
                    / estimated_risk_per_mwh
                )

            else:

                risk_based_size = 0

            # ==================================================
            # Exposure-based position size
            # ==================================================

            maximum_gross_exposure = (
                capital_before
                * (
                    float(
                        max_gross_exposure_percent
                    )
                    / 100
                )
            )

            if gross_price_per_mwh > 0:

                exposure_based_size = (
                    maximum_gross_exposure
                    / gross_price_per_mwh
                )

            else:

                exposure_based_size = float(
                    max_position_mwh
                )

            # ==================================================
            # Final position
            # ==================================================

            if final_multiplier <= 0:

                position_size = 0

            else:

                position_size = min(
                    risk_based_size,
                    exposure_based_size,
                    float(
                        max_position_mwh
                    )
                )

                position_size = max(
                    0,
                    position_size
                )

            # ==================================================
            # PnL
            # ==================================================

            gross_pnl_eur = (
                gross_pnl_per_mwh
                * position_size
            )

            transaction_cost_eur = (
                cost_per_mwh
                * position_size
            )

            net_pnl_eur = (
                net_pnl_per_mwh
                * position_size
            )

            current_capital = (
                capital_before
                + net_pnl_eur
            )

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
            # Trade action
            # ==================================================

            if final_multiplier == 0:

                sizing_action = (
                    "Skip Trade"
                )

            elif final_multiplier < 0.75:

                sizing_action = (
                    "Reduced Risk"
                )

            elif final_multiplier < 1:

                sizing_action = (
                    "Moderate Risk"
                )

            else:

                sizing_action = (
                    "Full Risk"
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

                    "VolatilityRegime":
                        volatility_regime,

                    "StructureRegime":
                        structure_regime,

                    "SpreadState":
                        spread_state,

                    "StructuralMultiplier":
                        structural_multiplier,

                    "AdaptiveMultiplier":
                        adaptive_multiplier,

                    "FinalMultiplier":
                        final_multiplier,

                    "HistoricalRegimeLabel":
                        historical_regime_label,

                    "PreviousRegimeTrades":
                        historical_trade_count,

                    "PreviousAveragePnLPerMWh":
                        historical_average_pnl,

                    "PreviousWinRate":
                        historical_win_rate,

                    "BaseRiskPercent":
                        risk_per_trade_percent,

                    "EffectiveRiskPercent":
                        effective_risk_percent,

                    "BaseRiskBudgetEUR":
                        base_risk_budget,

                    "AdjustedRiskBudgetEUR":
                        adjusted_risk_budget,

                    "EstimatedRiskPerMWh":
                        estimated_risk_per_mwh,

                    "PositionSizeMWh":
                        position_size,

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

                    "SizingAction":
                        sizing_action
                }
            )

            # ==================================================
            # Update regime history AFTER this outcome
            #
            # This preserves chronological causality.
            # ==================================================

            if regime_key not in (
                regime_history
            ):

                regime_history[
                    regime_key
                ] = []

            regime_history[
                regime_key
            ].append(
                net_pnl_per_mwh
            )

            equity_times.append(
                trade[
                    "ExitTime"
                ]
            )

            equity_values.append(
                current_capital
            )

        simulation_data = (
            pd.DataFrame(
                simulated_trades
            )
        )

        if simulation_data.empty:

            return self._empty_result(
                (
                    "No regime-aware portfolio "
                    "trades could be simulated."
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
        # Regime-aware statistics
        # ==================================================

        ending_capital = float(
            equity_curve[
                "PortfolioValue"
            ].iloc[
                -1
            ]
        )

        total_pnl = (
            ending_capital
            - float(
                starting_capital
            )
        )

        total_return_percent = (
            total_pnl
            / float(
                starting_capital
            )
        ) * 100

        maximum_drawdown_percent = float(
            equity_curve[
                "DrawdownPercent"
            ].min()
        )

        active_trades = (
            simulation_data[
                simulation_data[
                    "PositionSizeMWh"
                ] > 0
            ]
        )

        skipped_trades = (
            simulation_data[
                simulation_data[
                    "PositionSizeMWh"
                ] == 0
            ]
        )

        if not active_trades.empty:

            win_rate = (
                (
                    active_trades[
                        "NetPnLEUR"
                    ] > 0
                ).mean()
                * 100
            )

            average_position = (
                active_trades[
                    "PositionSizeMWh"
                ].mean()
            )

            average_effective_risk = (
                active_trades[
                    "EffectiveRiskPercent"
                ].mean()
            )

        else:

            win_rate = 0
            average_position = 0
            average_effective_risk = 0

        average_multiplier = (
            simulation_data[
                "FinalMultiplier"
            ].mean()
        )

        # ==================================================
        # Static benchmark
        # ==================================================

        base_statistics = (
            base_results[
                "statistics"
            ]
        )

        static_return = float(
            base_statistics[
                "total_return_percent"
            ]
        )

        static_drawdown = float(
            base_statistics[
                "maximum_drawdown_percent"
            ]
        )

        static_ending_capital = float(
            base_statistics[
                "ending_capital"
            ]
        )

        return_difference = (
            total_return_percent
            - static_return
        )

        drawdown_difference = (
            maximum_drawdown_percent
            - static_drawdown
        )

        # Positive means drawdown became less negative.
        drawdown_improvement = (
            maximum_drawdown_percent
            - static_drawdown
        )

        # ==================================================
        # Attribution by regime
        # ==================================================

        regime_attribution = (
            self._calculate_regime_attribution(
                simulation_data
            )
        )

        return {
            "success":
                True,

            "message":
                (
                    "Regime-aware portfolio "
                    "simulation completed."
                ),

            "simulation_data":
                simulation_data,

            "equity_curve":
                equity_curve,

            "base_results":
                base_results,

            "regime_attribution":
                regime_attribution,

            "statistics":
                {
                    "starting_capital":
                        float(
                            starting_capital
                        ),

                    "ending_capital":
                        ending_capital,

                    "total_pnl":
                        total_pnl,

                    "total_return_percent":
                        total_return_percent,

                    "maximum_drawdown_percent":
                        maximum_drawdown_percent,

                    "active_trades":
                        len(
                            active_trades
                        ),

                    "skipped_trades":
                        len(
                            skipped_trades
                        ),

                    "win_rate":
                        win_rate,

                    "average_position":
                        average_position,

                    "average_effective_risk":
                        average_effective_risk,

                    "average_multiplier":
                        average_multiplier,

                    "static_return":
                        static_return,

                    "static_drawdown":
                        static_drawdown,

                    "static_ending_capital":
                        static_ending_capital,

                    "return_difference":
                        return_difference,

                    "drawdown_difference":
                        drawdown_difference,

                    "drawdown_improvement":
                        drawdown_improvement
                }
        }

    # ==================================================
    # Structural multiplier
    # ==================================================

    def _get_structural_multiplier(
        self,
        volatility_regime,
        structure_regime
    ):

        volatility_multipliers = {
            "Low":
                0.85,

            "Normal":
                1.00,

            "High":
                0.80,

            "Extreme":
                0.50,

            "Unknown":
                0.75
        }

        structure_multipliers = {
            "Mean-Reverting":
                1.00,

            "Mixed":
                0.75,

            "Trending":
                0.50,

            "Unknown":
                0.75
        }

        volatility_multiplier = (
            volatility_multipliers.get(
                volatility_regime,
                0.75
            )
        )

        structure_multiplier = (
            structure_multipliers.get(
                structure_regime,
                0.75
            )
        )

        return min(
            volatility_multiplier,
            structure_multiplier
        )

    # ==================================================
    # Adaptive multiplier
    #
    # Uses only previous outcomes in the same regime.
    # ==================================================

    def _get_adaptive_multiplier(
        self,
        previous_outcomes
    ):

        trade_count = len(
            previous_outcomes
        )

        if trade_count == 0:

            return (
                1.0,
                "No History",
                0,
                0,
                0
            )

        outcomes = np.array(
            previous_outcomes,
            dtype=float
        )

        average_pnl = float(
            outcomes.mean()
        )

        win_rate = float(
            (
                outcomes > 0
            ).mean()
            * 100
        )

        # --------------------------------------------------
        # Require several completed historical signals
        # before making a performance-based adjustment.
        # --------------------------------------------------

        if trade_count < 3:

            return (
                1.0,
                "Insufficient History",
                trade_count,
                average_pnl,
                win_rate
            )

        # --------------------------------------------------
        # Strong historical regime
        # --------------------------------------------------

        if (
            average_pnl > 0
            and
            win_rate >= 60
        ):

            return (
                1.0,
                "Strong",
                trade_count,
                average_pnl,
                win_rate
            )

        # --------------------------------------------------
        # Positive but less convincing
        # --------------------------------------------------

        if (
            average_pnl > 0
            and
            win_rate >= 50
        ):

            return (
                0.75,
                "Positive",
                trade_count,
                average_pnl,
                win_rate
            )

        # --------------------------------------------------
        # Mixed regime
        # --------------------------------------------------

        if (
            average_pnl > 0
        ):

            return (
                0.60,
                "Mixed",
                trade_count,
                average_pnl,
                win_rate
            )

        # --------------------------------------------------
        # Weak historical regime
        # --------------------------------------------------

        if win_rate >= 40:

            return (
                0.50,
                "Weak",
                trade_count,
                average_pnl,
                win_rate
            )

        # --------------------------------------------------
        # Poor regime
        #
        # The strategy signal is observed, but no capital
        # position is opened.
        # --------------------------------------------------

        return (
            0.0,
            "Poor - Skip",
            trade_count,
            average_pnl,
            win_rate
        )

    # ==================================================
    # Regime attribution
    # ==================================================

    def _calculate_regime_attribution(
        self,
        simulation_data
    ):

        if simulation_data.empty:

            return pd.DataFrame()

        rows = []

        grouped = (
            simulation_data.groupby(
                [
                    "VolatilityRegime",
                    "StructureRegime"
                ],
                dropna=False
            )
        )

        for (
            volatility_regime,
            structure_regime
        ), group in grouped:

            active = (
                group[
                    group[
                        "PositionSizeMWh"
                    ] > 0
                ]
            )

            trades = len(
                group
            )

            active_trades = len(
                active
            )

            skipped = (
                trades
                - active_trades
            )

            net_pnl = (
                group[
                    "NetPnLEUR"
                ].sum()
            )

            if active_trades > 0:

                win_rate = (
                    (
                        active[
                            "NetPnLEUR"
                        ] > 0
                    ).mean()
                    * 100
                )

                average_multiplier = (
                    active[
                        "FinalMultiplier"
                    ].mean()
                )

            else:

                win_rate = 0

                average_multiplier = (
                    group[
                        "FinalMultiplier"
                    ].mean()
                )

            rows.append(
                {
                    "Volatility":
                        volatility_regime,

                    "Structure":
                        structure_regime,

                    "Signals":
                        trades,

                    "ActiveTrades":
                        active_trades,

                    "SkippedTrades":
                        skipped,

                    "NetPnL":
                        net_pnl,

                    "WinRate":
                        win_rate,

                    "AverageMultiplier":
                        average_multiplier
                }
            )

        return (
            pd.DataFrame(
                rows
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

            "base_results":
                {},

            "regime_attribution":
                pd.DataFrame(),

            "statistics":
                {}
        }