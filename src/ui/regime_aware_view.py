import tkinter as tk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.strategy.regime_aware_portfolio import (
    RegimeAwarePortfolioSimulator
)


class RegimeAwareSizingView:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.simulator = (
            RegimeAwarePortfolioSimulator(
                spread_analyzer,
                backtester
            )
        )

    # ==================================================
    # Main view
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

        tk.Label(
            container,
            text=(
                "Regime-Aware Position Sizing"
            ),
            font=(
                "Arial",
                11,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        tk.Label(
            container,
            text=(
                "Position risk is reduced when the current "
                "market regime is structurally less suitable "
                "or previous completed signals in the same "
                "regime performed poorly."
            ),
            font=(
                "Arial",
                8
            ),
            wraplength=900,
            justify="left"
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        # ==================================================
        # Run
        # ==================================================

        tk.Label(
            container,
            text=(
                "Running chronological regime-aware "
                "portfolio simulation..."
            ),
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(5, 10)
        )

        container.update()

        results = (
            self.simulator
            .run_simulation(
                spread_data,
                window=window,
                entry_threshold=(
                    entry_threshold
                ),
                exit_threshold=(
                    exit_threshold
                ),
                transaction_cost_per_leg=(
                    transaction_cost
                ),
                starting_capital=(
                    capital
                ),
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

        if not results[
            "success"
        ]:

            self._show_error(
                container,
                results[
                    "message"
                ]
            )

            return

        self._show_results(
            container,
            results,
            time_range
        )

    # ==================================================
    # Results
    # ==================================================

    def _show_results(
        self,
        container,
        results,
        time_range
    ):

        statistics = (
            results[
                "statistics"
            ]
        )

        simulation_data = (
            results[
                "simulation_data"
            ]
        )

        # ==================================================
        # Title
        # ==================================================

        tk.Label(
            container,
            text="Regime-Aware Portfolio Comparison",
            font=(
                "Arial",
                11,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        # ==================================================
        # Regime-aware results
        # ==================================================

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        self._show_stats(
            row_1,
            [
                (
                    "Ending Capital",
                    (
                        f"€{statistics['ending_capital']:,.0f}"
                    )
                ),
                (
                    "Return",
                    (
                        f"{statistics['total_return_percent']:+.2f}%"
                    )
                ),
                (
                    "Max Drawdown",
                    (
                        f"{statistics['maximum_drawdown_percent']:.2f}%"
                    )
                ),
                (
                    "Active Trades",
                    statistics[
                        "active_trades"
                    ]
                ),
                (
                    "Skipped Trades",
                    statistics[
                        "skipped_trades"
                    ]
                ),
                (
                    "Win Rate",
                    (
                        f"{statistics['win_rate']:.1f}%"
                    )
                )
            ],
            bold=True
        )

        # ==================================================
        # Risk behaviour
        # ==================================================

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 6)
        )

        self._show_stats(
            row_2,
            [
                (
                    "Base Risk",
                    (
                        f"{simulation_data['BaseRiskPercent'].iloc[0]:.2f}%"
                    )
                ),
                (
                    "Avg Effective Risk",
                    (
                        f"{statistics['average_effective_risk']:.2f}%"
                    )
                ),
                (
                    "Avg Multiplier",
                    (
                        f"{statistics['average_multiplier']:.2f}x"
                    )
                ),
                (
                    "Avg Position",
                    (
                        f"{statistics['average_position']:.2f} MWh"
                    )
                )
            ]
        )

        # ==================================================
        # Static versus adaptive
        # ==================================================

        tk.Label(
            container,
            text="Static Risk vs Regime-Aware Risk",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        comparison_frame = tk.Frame(
            container
        )

        comparison_frame.pack(
            fill="x",
            pady=(0, 7)
        )

        headers = [
            "Metric",
            "Static",
            "Regime-Aware",
            "Difference"
        ]

        widths = [
            22,
            16,
            16,
            16
        ]

        for (
            column,
            header
        ) in enumerate(
            headers
        ):

            tk.Label(
                comparison_frame,
                text=header,
                font=(
                    "Arial",
                    8,
                    "bold"
                ),
                width=widths[
                    column
                ],
                anchor="w"
            ).grid(
                row=0,
                column=column
            )

        comparison_rows = [
            (
                "Portfolio Return",
                (
                    f"{statistics['static_return']:+.2f}%"
                ),
                (
                    f"{statistics['total_return_percent']:+.2f}%"
                ),
                (
                    f"{statistics['return_difference']:+.2f} pp"
                )
            ),
            (
                "Max Drawdown",
                (
                    f"{statistics['static_drawdown']:.2f}%"
                ),
                (
                    f"{statistics['maximum_drawdown_percent']:.2f}%"
                ),
                (
                    f"{statistics['drawdown_improvement']:+.2f} pp"
                )
            ),
            (
                "Ending Capital",
                (
                    f"€{statistics['static_ending_capital']:,.0f}"
                ),
                (
                    f"€{statistics['ending_capital']:,.0f}"
                ),
                (
                    f"€{statistics['ending_capital'] - statistics['static_ending_capital']:+,.0f}"
                )
            )
        ]

        for (
            row_number,
            values
        ) in enumerate(
            comparison_rows,
            start=1
        ):

            for (
                column,
                value
            ) in enumerate(
                values
            ):

                tk.Label(
                    comparison_frame,
                    text=value,
                    font=(
                        "Arial",
                        8
                    ),
                    width=widths[
                        column
                    ],
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column
                )

        # ==================================================
        # Recent decisions
        # ==================================================

        tk.Label(
            container,
            text="Recent Regime-Aware Sizing Decisions",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        self._create_trade_table(
            container,
            simulation_data
        )

        # ==================================================
        # Regime attribution
        # ==================================================

        attribution = (
            results[
                "regime_attribution"
            ]
        )

        if not attribution.empty:

            tk.Label(
                container,
                text="Regime Sizing Attribution",
                font=(
                    "Arial",
                    9,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=(0, 3)
            )

            self._create_regime_table(
                container,
                attribution
            )

        # ==================================================
        # Research note
        # ==================================================

        tk.Label(
            container,
            text=(
                "Adaptive regime performance uses only outcomes "
                "from earlier completed signals in the same regime. "
                "Future trades never determine earlier risk sizing."
            ),
            font=(
                "Arial",
                8
            ),
            wraplength=900,
            justify="left"
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        # ==================================================
        # Chart
        # ==================================================

        figure = (
            self._create_chart(
                results,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Recent decisions
    # ==================================================

    def _create_trade_table(
        self,
        container,
        simulation_data
    ):

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 7)
        )

        headers = [
            "Volatility",
            "Structure",
            "History",
            "Multiplier",
            "Risk %",
            "Size",
            "PnL",
            "Action"
        ]

        widths = [
            11,
            15,
            16,
            11,
            9,
            10,
            11,
            14
        ]

        for (
            column,
            header
        ) in enumerate(
            headers
        ):

            tk.Label(
                frame,
                text=header,
                font=(
                    "Arial",
                    8,
                    "bold"
                ),
                width=widths[
                    column
                ],
                anchor="w"
            ).grid(
                row=0,
                column=column
            )

        recent = (
            simulation_data.tail(
                5
            )
        )

        for (
            row_number,
            (_, trade)
        ) in enumerate(
            recent.iterrows(),
            start=1
        ):

            values = [
                trade[
                    "VolatilityRegime"
                ],
                trade[
                    "StructureRegime"
                ],
                trade[
                    "HistoricalRegimeLabel"
                ],
                (
                    f"{trade['FinalMultiplier']:.2f}x"
                ),
                (
                    f"{trade['EffectiveRiskPercent']:.2f}%"
                ),
                (
                    f"{trade['PositionSizeMWh']:.2f}"
                ),
                (
                    f"€{trade['NetPnLEUR']:+,.0f}"
                ),
                trade[
                    "SizingAction"
                ]
            ]

            for (
                column,
                value
            ) in enumerate(
                values
            ):

                tk.Label(
                    frame,
                    text=value,
                    font=(
                        "Arial",
                        8
                    ),
                    width=widths[
                        column
                    ],
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column
                )

    # ==================================================
    # Regime attribution table
    # ==================================================

    def _create_regime_table(
        self,
        container,
        attribution
    ):

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 7)
        )

        headers = [
            "Volatility",
            "Structure",
            "Signals",
            "Active",
            "Skipped",
            "Net PnL",
            "Win %",
            "Avg Mult."
        ]

        widths = [
            11,
            15,
            8,
            8,
            8,
            12,
            9,
            10
        ]

        for (
            column,
            header
        ) in enumerate(
            headers
        ):

            tk.Label(
                frame,
                text=header,
                font=(
                    "Arial",
                    8,
                    "bold"
                ),
                width=widths[
                    column
                ],
                anchor="w"
            ).grid(
                row=0,
                column=column
            )

        for (
            row_number,
            (_, row)
        ) in enumerate(
            attribution.head(
                6
            ).iterrows(),
            start=1
        ):

            values = [
                row[
                    "Volatility"
                ],
                row[
                    "Structure"
                ],
                (
                    f"{int(row['Signals'])}"
                ),
                (
                    f"{int(row['ActiveTrades'])}"
                ),
                (
                    f"{int(row['SkippedTrades'])}"
                ),
                (
                    f"€{row['NetPnL']:+,.0f}"
                ),
                (
                    f"{row['WinRate']:.1f}%"
                ),
                (
                    f"{row['AverageMultiplier']:.2f}x"
                )
            ]

            for (
                column,
                value
            ) in enumerate(
                values
            ):

                tk.Label(
                    frame,
                    text=value,
                    font=(
                        "Arial",
                        8
                    ),
                    width=widths[
                        column
                    ],
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column
                )

    # ==================================================
    # Comparison chart
    # ==================================================

    def _create_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                4.8
            ),
            dpi=100
        )

        adaptive_equity = (
            results[
                "equity_curve"
            ]
        )

        base_equity = (
            results[
                "base_results"
            ][
                "equity_curve"
            ]
        )

        simulation_data = (
            results[
                "simulation_data"
            ]
        )

        # ==================================================
        # Equity comparison
        # ==================================================

        equity_axis = (
            figure.add_subplot(
                211
            )
        )

        equity_axis.plot(
            base_equity[
                "TimeDK"
            ],
            base_equity[
                "PortfolioValue"
            ],
            label="Static Risk",
            linestyle="--"
        )

        equity_axis.plot(
            adaptive_equity[
                "TimeDK"
            ],
            adaptive_equity[
                "PortfolioValue"
            ],
            label="Regime-Aware Risk",
            linewidth=2
        )

        equity_axis.set_title(
            (
                "Static vs Regime-Aware Portfolio "
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
        # Risk multiplier
        # ==================================================

        multiplier_axis = (
            figure.add_subplot(
                212
            )
        )

        multiplier_axis.step(
            simulation_data[
                "EntryTime"
            ],
            simulation_data[
                "FinalMultiplier"
            ],
            where="post",
            label="Risk Multiplier"
        )

        multiplier_axis.axhline(
            y=1,
            linestyle="--",
            linewidth=1,
            label="Full Risk"
        )

        multiplier_axis.axhline(
            y=0.5,
            linestyle=":",
            linewidth=1,
            label="50% Risk"
        )

        multiplier_axis.set_ylim(
            -0.05,
            1.10
        )

        multiplier_axis.set_ylabel(
            "Multiplier"
        )

        multiplier_axis.set_xlabel(
            "Time"
        )

        multiplier_axis.grid(
            True,
            alpha=0.3
        )

        multiplier_axis.legend(
            fontsize=8
        )

        self._format_dates(
            multiplier_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Helpers
    # ==================================================

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