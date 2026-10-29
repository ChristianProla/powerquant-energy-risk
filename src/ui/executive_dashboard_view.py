import tkinter as tk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.strategy.portfolio_simulation import (
    CapitalPortfolioSimulator
)

from src.strategy.performance_attribution import (
    PerformanceAttributionAnalyzer
)

from src.strategy.live_monitoring import (
    LiveStrategyMonitor
)

from src.strategy.regime_detection import (
    MarketRegimeDetector
)

from src.strategy.regime_aware_portfolio import (
    RegimeAwarePortfolioSimulator
)


class ExecutiveDashboardView:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.portfolio_simulator = (
            CapitalPortfolioSimulator(
                spread_analyzer,
                backtester
            )
        )

        self.performance_analyzer = (
            PerformanceAttributionAnalyzer()
        )

        self.live_monitor = (
            LiveStrategyMonitor(
                spread_analyzer
            )
        )

        self.regime_detector = (
            MarketRegimeDetector()
        )

        self.regime_aware_simulator = (
            RegimeAwarePortfolioSimulator(
                spread_analyzer,
                backtester
            )
        )

    # ==================================================
    # Main dashboard
    # ==================================================

    def show(
        self,
        container,
        spread_data,
        time_range,
        window,
        entry_threshold,
        exit_threshold,
        transaction_cost,
        capital,
        risk_per_trade,
        max_position,
        max_exposure
    ):

        self._clear_frame(
            container
        )

        # ==================================================
        # Header
        # ==================================================

        tk.Label(
            container,
            text="PowerQuant Executive Dashboard",
            font=(
                "Arial",
                14,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        tk.Label(
            container,
            text=(
                "Integrated market, strategy, portfolio "
                "and risk overview."
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        loading_label = tk.Label(
            container,
            text="Calculating dashboard...",
            font=(
                "Arial",
                10,
                "bold"
            )
        )

        loading_label.pack(
            anchor="w",
            pady=15
        )

        container.update()

        # ==================================================
        # Base portfolio
        # ==================================================

        portfolio_results = (
            self.portfolio_simulator
            .run_simulation(
                spread_data,
                window=window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost_per_leg=(
                    transaction_cost
                ),
                starting_capital=capital,
                risk_per_trade_percent=(
                    risk_per_trade
                ),
                max_position_mwh=(
                    max_position
                ),
                max_gross_exposure_percent=(
                    max_exposure
                )
            )
        )

        if not portfolio_results[
            "success"
        ]:

            self._clear_frame(
                container
            )

            self._show_error(
                container,
                portfolio_results[
                    "message"
                ]
            )

            return

        # ==================================================
        # Current strategy state
        # ==================================================

        live_results = (
            self.live_monitor
            .analyse_current_state(
                spread_data,
                window=window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost_per_leg=(
                    transaction_cost
                ),
                capital=capital,
                risk_per_trade_percent=(
                    risk_per_trade
                ),
                max_position_mwh=(
                    max_position
                ),
                max_gross_exposure_percent=(
                    max_exposure
                )
            )
        )

        # ==================================================
        # Performance attribution
        # ==================================================

        if (
            portfolio_results[
                "simulation_data"
            ].empty
        ):

            performance_results = None

        else:

            performance_results = (
                self.performance_analyzer
                .analyze(
                    portfolio_results[
                        "simulation_data"
                    ],
                    portfolio_results[
                        "equity_curve"
                    ],
                    capital
                )
            )

        # ==================================================
        # Regime detection
        # ==================================================

        regime_results = (
            self.regime_detector
            .analyze(
                spread_data,
                portfolio_results[
                    "simulation_data"
                ],
                signal_window=window
            )
        )

        # ==================================================
        # Regime-aware portfolio
        # ==================================================

        adaptive_results = (
            self.regime_aware_simulator
            .run_simulation(
                spread_data,
                window=window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost_per_leg=(
                    transaction_cost
                ),
                starting_capital=capital,
                risk_per_trade_percent=(
                    risk_per_trade
                ),
                max_position_mwh=(
                    max_position
                ),
                max_gross_exposure_percent=(
                    max_exposure
                )
            )
        )

        self._clear_frame(
            container
        )

        # ==================================================
        # Recreate header
        # ==================================================

        tk.Label(
            container,
            text="PowerQuant Executive Dashboard",
            font=(
                "Arial",
                14,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        tk.Label(
            container,
            text=(
                "Integrated market, strategy, portfolio "
                "and risk overview."
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        # ==================================================
        # Current market
        # ==================================================

        if live_results[
            "success"
        ]:

            self._create_section_title(
                container,
                "CURRENT MARKET"
            )

            market_row = tk.Frame(
                container
            )

            market_row.pack(
                fill="x",
                pady=(0, 8)
            )

            self._show_stats(
                market_row,
                [
                    (
                        "DK1",
                        (
                            f"{live_results['dk1_price']:.2f}"
                        )
                    ),
                    (
                        "DK2",
                        (
                            f"{live_results['dk2_price']:.2f}"
                        )
                    ),
                    (
                        "Spread",
                        (
                            f"{live_results['spread']:+.2f}"
                        )
                    ),
                    (
                        "Z-Score",
                        (
                            f"{live_results['z_score']:+.2f}"
                        )
                    ),
                    (
                        "Signal",
                        live_results[
                            "signal"
                        ]
                    ),
                    (
                        "Market State",
                        live_results[
                            "market_state"
                        ]
                    )
                ],
                bold=True
            )

        # ==================================================
        # Regime
        # ==================================================

        if regime_results[
            "success"
        ]:

            current_regime = (
                regime_results[
                    "current_regime"
                ]
            )

            self._create_section_title(
                container,
                "MARKET REGIME"
            )

            regime_row = tk.Frame(
                container
            )

            regime_row.pack(
                fill="x",
                pady=(0, 8)
            )

            self._show_stats(
                regime_row,
                [
                    (
                        "Volatility",
                        current_regime[
                            "volatility"
                        ]
                    ),
                    (
                        "Structure",
                        current_regime[
                            "structure"
                        ]
                    ),
                    (
                        "Spread State",
                        current_regime[
                            "spread_state"
                        ]
                    ),
                    (
                        "Relative Pricing",
                        current_regime[
                            "relative_price_state"
                        ]
                    ),
                    (
                        "Best Historical Vol.",
                        regime_results[
                            "best_volatility_regime"
                        ]
                    ),
                    (
                        "Best Structure",
                        regime_results[
                            "best_structure_regime"
                        ]
                    )
                ]
            )

        # ==================================================
        # Strategy / position
        # ==================================================

        if live_results[
            "success"
        ]:

            self._create_section_title(
                container,
                "CURRENT STRATEGY"
            )

            strategy_row = tk.Frame(
                container
            )

            strategy_row.pack(
                fill="x",
                pady=(0, 8)
            )

            self._show_stats(
                strategy_row,
                [
                    (
                        "Direction",
                        live_results[
                            "direction"
                        ]
                    ),
                    (
                        "Suggested Size",
                        (
                            f"{live_results['active_position_size']:.2f} "
                            f"MWh"
                        )
                    ),
                    (
                        "Risk Budget",
                        (
                            f"€{live_results['risk_budget']:,.0f}"
                        )
                    ),
                    (
                        "Exposure",
                        (
                            f"€{live_results['gross_exposure']:,.0f}"
                        )
                    ),
                    (
                        "Leverage",
                        (
                            f"{live_results['leverage']:.2f}x"
                        )
                    ),
                    (
                        "Constraint",
                        live_results[
                            "binding_constraint"
                        ]
                    )
                ]
            )

        # ==================================================
        # Portfolio
        # ==================================================

        portfolio_statistics = (
            portfolio_results[
                "statistics"
            ]
        )

        self._create_section_title(
            container,
            "PORTFOLIO PERFORMANCE"
        )

        portfolio_row = tk.Frame(
            container
        )

        portfolio_row.pack(
            fill="x",
            pady=(0, 8)
        )

        self._show_stats(
            portfolio_row,
            [
                (
                    "Capital",
                    (
                        f"€{portfolio_statistics['ending_capital']:,.0f}"
                    )
                ),
                (
                    "Return",
                    (
                        f"{portfolio_statistics['total_return_percent']:+.2f}%"
                    )
                ),
                (
                    "PnL",
                    (
                        f"€{portfolio_statistics['total_pnl']:+,.0f}"
                    )
                ),
                (
                    "Max Drawdown",
                    (
                        f"{portfolio_statistics['maximum_drawdown_percent']:.2f}%"
                    )
                ),
                (
                    "Trades",
                    portfolio_statistics[
                        "number_of_trades"
                    ]
                ),
                (
                    "Win Rate",
                    (
                        f"{portfolio_statistics['win_rate']:.1f}%"
                    )
                )
            ],
            bold=True
        )

        # ==================================================
        # Strategy health
        # ==================================================

        self._create_section_title(
            container,
            "STRATEGY HEALTH"
        )

        health_row = tk.Frame(
            container
        )

        health_row.pack(
            fill="x",
            pady=(0, 8)
        )

        if (
            performance_results
            and
            performance_results[
                "success"
            ]
        ):

            profit_factor = (
                performance_results[
                    "profit_factor"
                ]
            )

            if profit_factor == float(
                "inf"
            ):

                profit_factor_text = "∞"

            else:

                profit_factor_text = (
                    f"{profit_factor:.2f}"
                )

            trade_sharpe = (
                performance_results[
                    "trade_sharpe_proxy"
                ]
            )

            cost_drag = (
                performance_results[
                    "cost_drag_percent"
                ]
            )

            profitable_months = (
                performance_results[
                    "profitable_month_percentage"
                ]
            )

        else:

            profit_factor_text = "N/A"
            trade_sharpe = 0
            cost_drag = 0
            profitable_months = 0

        if adaptive_results[
            "success"
        ]:

            adaptive_statistics = (
                adaptive_results[
                    "statistics"
                ]
            )

            adaptive_return = (
                adaptive_statistics[
                    "total_return_percent"
                ]
            )

            adaptive_drawdown = (
                adaptive_statistics[
                    "maximum_drawdown_percent"
                ]
            )

            average_multiplier = (
                adaptive_statistics[
                    "average_multiplier"
                ]
            )

        else:

            adaptive_return = 0
            adaptive_drawdown = 0
            average_multiplier = 0

        self._show_stats(
            health_row,
            [
                (
                    "Profit Factor",
                    profit_factor_text
                ),
                (
                    "Trade Sharpe",
                    (
                        f"{trade_sharpe:.2f}"
                    )
                ),
                (
                    "Cost Drag",
                    (
                        f"{cost_drag:.1f}%"
                    )
                ),
                (
                    "Profitable Months",
                    (
                        f"{profitable_months:.1f}%"
                    )
                ),
                (
                    "Adaptive Return",
                    (
                        f"{adaptive_return:+.2f}%"
                    )
                ),
                (
                    "Adaptive Risk",
                    (
                        f"{average_multiplier:.2f}x"
                    )
                )
            ]
        )

        # ==================================================
        # Overall dashboard assessment
        # ==================================================

        status = (
            self._calculate_strategy_status(
                live_results,
                performance_results,
                adaptive_results,
                regime_results
            )
        )

        self._create_section_title(
            container,
            "POWERQUANT ASSESSMENT"
        )

        assessment_frame = tk.Frame(
            container
        )

        assessment_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        tk.Label(
            assessment_frame,
            text=status[
                "status"
            ],
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            side="left",
            padx=(0, 20)
        )

        tk.Label(
            assessment_frame,
            text=status[
                "explanation"
            ],
            font=(
                "Arial",
                9
            ),
            wraplength=760,
            justify="left"
        ).pack(
            side="left"
        )

        # ==================================================
        # Chart
        # ==================================================

        figure = (
            self._create_dashboard_chart(
                portfolio_results,
                adaptive_results,
                live_results,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Strategy health classification
    # ==================================================

    def _calculate_strategy_status(
        self,
        live_results,
        performance_results,
        adaptive_results,
        regime_results
    ):

        score = 0

        reasons = []

        # ==================================================
        # Historical performance
        # ==================================================

        if (
            performance_results
            and
            performance_results[
                "success"
            ]
        ):

            if (
                performance_results[
                    "total_return_percent"
                ] > 0
            ):

                score += 1

                reasons.append(
                    "historical portfolio return is positive"
                )

            if (
                performance_results[
                    "profit_factor"
                ] > 1
            ):

                score += 1

                reasons.append(
                    "profit factor is above 1"
                )

            if (
                performance_results[
                    "trade_sharpe_proxy"
                ] > 0
            ):

                score += 1

        # ==================================================
        # Adaptive portfolio
        # ==================================================

        if adaptive_results[
            "success"
        ]:

            adaptive_statistics = (
                adaptive_results[
                    "statistics"
                ]
            )

            if (
                adaptive_statistics[
                    "total_return_percent"
                ] > 0
            ):

                score += 1

            if (
                adaptive_statistics[
                    "maximum_drawdown_percent"
                ]
                >
                adaptive_statistics[
                    "static_drawdown"
                ]
            ):

                score += 1

                reasons.append(
                    "regime-aware sizing reduces drawdown"
                )

        # ==================================================
        # Current regime
        # ==================================================

        if regime_results[
            "success"
        ]:

            current = (
                regime_results[
                    "current_regime"
                ]
            )

            if (
                current[
                    "structure"
                ]
                == "Mean-Reverting"
            ):

                score += 1

                reasons.append(
                    "current market structure is mean-reverting"
                )

            if (
                current[
                    "volatility"
                ]
                == "Extreme"
            ):

                score -= 1

                reasons.append(
                    "current volatility is extreme"
                )

        # ==================================================
        # Current signal
        # ==================================================

        if live_results[
            "success"
        ]:

            if (
                live_results[
                    "signal"
                ]
                == "ENTRY SIGNAL"
            ):

                reasons.append(
                    "a current strategy entry signal is active"
                )

        # ==================================================
        # Classification
        # ==================================================

        if score >= 5:

            status = (
                "STRONG"
            )

        elif score >= 3:

            status = (
                "HEALTHY"
            )

        elif score >= 1:

            status = (
                "MIXED"
            )

        else:

            status = (
                "CAUTIOUS"
            )

        if reasons:

            explanation = (
                "; ".join(
                    reasons
                )
                + "."
            )

        else:

            explanation = (
                "Insufficient evidence for a strong "
                "strategy-health assessment."
            )

        return {
            "status":
                status,

            "explanation":
                explanation
        }

    # ==================================================
    # Dashboard chart
    # ==================================================

    def _create_dashboard_chart(
        self,
        portfolio_results,
        adaptive_results,
        live_results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                5.5
            ),
            dpi=100
        )

        # ==================================================
        # Portfolio curves
        # ==================================================

        equity_axis = (
            figure.add_subplot(
                211
            )
        )

        static_equity = (
            portfolio_results[
                "equity_curve"
            ]
        )

        equity_axis.plot(
            static_equity[
                "TimeDK"
            ],
            static_equity[
                "PortfolioValue"
            ],
            label="Static Risk Portfolio",
            linestyle="--"
        )

        if adaptive_results[
            "success"
        ]:

            adaptive_equity = (
                adaptive_results[
                    "equity_curve"
                ]
            )

            equity_axis.plot(
                adaptive_equity[
                    "TimeDK"
                ],
                adaptive_equity[
                    "PortfolioValue"
                ],
                label="Regime-Aware Portfolio",
                linewidth=2
            )

        equity_axis.set_title(
            (
                "Portfolio Strategy Overview "
                f"- {time_range}"
            )
        )

        equity_axis.set_ylabel(
            "Portfolio Value (EUR)"
        )

        equity_axis.grid(
            True,
            alpha=0.3
        )

        equity_axis.legend(
            fontsize=8
        )

        self._format_dates(
            equity_axis
        )

        # ==================================================
        # Current spread / Z
        # ==================================================

        market_axis = (
            figure.add_subplot(
                212
            )
        )

        if live_results[
            "success"
        ]:

            chart_data = (
                live_results[
                    "chart_data"
                ]
            )

            market_axis.plot(
                chart_data[
                    "TimeDK"
                ],
                chart_data[
                    "ZScore"
                ],
                label="Spread Z-Score"
            )

            market_axis.axhline(
                y=(
                    live_results[
                        "entry_threshold"
                    ]
                ),
                linestyle="--",
                label="Entry Threshold"
            )

            market_axis.axhline(
                y=-(
                    live_results[
                        "entry_threshold"
                    ]
                ),
                linestyle="--"
            )

            market_axis.axhline(
                y=0,
                linewidth=1,
                alpha=0.5
            )

            market_axis.scatter(
                [
                    chart_data[
                        "TimeDK"
                    ].iloc[
                        -1
                    ]
                ],
                [
                    live_results[
                        "z_score"
                    ]
                ],
                s=50,
                label="Current Z"
            )

        market_axis.set_title(
            "Current Spread Signal Context"
        )

        market_axis.set_ylabel(
            "Z-Score"
        )

        market_axis.set_xlabel(
            "Time"
        )

        market_axis.grid(
            True,
            alpha=0.3
        )

        market_axis.legend(
            fontsize=8
        )

        self._format_dates(
            market_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Helpers
    # ==================================================

    def _create_section_title(
        self,
        container,
        text
    ):

        tk.Label(
            container,
            text=text,
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(3, 2)
        )

    def _format_dates(
        self,
        axis
    ):

        locator = (
            mdates.AutoDateLocator()
        )

        formatter = (
            mdates.ConciseDateFormatter(
                locator
            )
        )

        axis.xaxis.set_major_locator(
            locator
        )

        axis.xaxis.set_major_formatter(
            formatter
        )

    def _show_stats(
        self,
        container,
        values,
        bold=False
    ):

        for (
            name,
            value
        ) in values:

            font = (
                "Arial",
                9,
                "bold"
            ) if bold else (
                "Arial",
                9
            )

            tk.Label(
                container,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=font,
                padx=7
            ).pack(
                side="left"
            )

    def _embed_chart(
        self,
        container,
        figure
    ):

        canvas = (
            FigureCanvasTkAgg(
                figure,
                master=container
            )
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def _show_error(
        self,
        container,
        message
    ):

        tk.Label(
            container,
            text=message,
            font=(
                "Arial",
                11,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=30
        )

    def _clear_frame(
        self,
        frame
    ):

        for widget in (
            frame.winfo_children()
        ):

            widget.destroy()