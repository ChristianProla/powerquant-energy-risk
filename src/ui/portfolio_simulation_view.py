import tkinter as tk
from tkinter import ttk

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

from src.ui.live_monitoring_view import (
    LiveStrategyMonitoringView
)


class PortfolioSimulationView:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.simulator = (
            CapitalPortfolioSimulator(
                spread_analyzer,
                backtester
            )
        )

        self.performance_analyzer = (
            PerformanceAttributionAnalyzer()
        )

        self.live_monitoring_view = (
            LiveStrategyMonitoringView(
                spread_analyzer
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
        transaction_cost
    ):

        self._clear_frame(
            container
        )

        tk.Label(
            container,
            text=(
                "Capital-Based Portfolio "
                "& Strategy Performance"
            ),
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        tk.Label(
            container,
            text=(
                "Simulate portfolio risk, analyse historical "
                "performance and monitor the latest strategy signal."
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
        # Strategy parameters
        # ==================================================

        strategy_frame = tk.Frame(
            container
        )

        strategy_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        self._show_stats(
            strategy_frame,
            [
                (
                    "Window",
                    f"{window}h"
                ),
                (
                    "Entry Z",
                    f"±{entry_threshold:.1f}"
                ),
                (
                    "Exit Z",
                    f"±{exit_threshold:.1f}"
                ),
                (
                    "Cost / Leg",
                    (
                        f"{transaction_cost:.2f} "
                        f"EUR/MWh"
                    )
                )
            ]
        )

        # ==================================================
        # Portfolio controls
        # ==================================================

        controls = tk.Frame(
            container
        )

        controls.pack(
            fill="x",
            pady=(0, 8)
        )

        tk.Label(
            controls,
            text="Starting Capital:"
        ).pack(
            side="left"
        )

        capital_variable = (
            tk.StringVar(
                value="100000"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                capital_variable
            ),
            values=[
                "25000",
                "50000",
                "100000",
                "250000",
                "500000"
            ],
            state="readonly",
            width=10
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Risk / Trade:"
        ).pack(
            side="left"
        )

        risk_variable = (
            tk.StringVar(
                value="1.0"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                risk_variable
            ),
            values=[
                "0.5",
                "1.0",
                "1.5",
                "2.0"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Max Position:"
        ).pack(
            side="left"
        )

        position_variable = (
            tk.StringVar(
                value="10"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                position_variable
            ),
            values=[
                "5",
                "10",
                "20",
                "50"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Max Exposure:"
        ).pack(
            side="left"
        )

        exposure_variable = (
            tk.StringVar(
                value="50"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                exposure_variable
            ),
            values=[
                "25",
                "50",
                "100",
                "200"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 5)
        )

        tk.Label(
            controls,
            text="%"
        ).pack(
            side="left"
        )

        # ==================================================
        # Navigation
        # ==================================================

        button_frame = tk.Frame(
            container
        )

        button_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # Simulation helper
        # ==================================================

        def get_simulation_results():

            return (
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
                    starting_capital=float(
                        capital_variable.get()
                    ),
                    risk_per_trade_percent=float(
                        risk_variable.get()
                    ),
                    max_position_mwh=float(
                        position_variable.get()
                    ),
                    max_gross_exposure_percent=float(
                        exposure_variable.get()
                    )
                )
            )

        # ==================================================
        # Portfolio summary
        # ==================================================

        def show_portfolio_summary():

            self._clear_frame(
                output_frame
            )

            results = (
                get_simulation_results()
            )

            if not results[
                "success"
            ]:

                self._show_error(
                    output_frame,
                    results[
                        "message"
                    ]
                )

                return

            self._show_portfolio_results(
                output_frame,
                results,
                time_range
            )

        # ==================================================
        # Performance dashboard
        # ==================================================

        def show_performance_dashboard():

            self._clear_frame(
                output_frame
            )

            simulation_results = (
                get_simulation_results()
            )

            if not simulation_results[
                "success"
            ]:

                self._show_error(
                    output_frame,
                    simulation_results[
                        "message"
                    ]
                )

                return

            if simulation_results[
                "simulation_data"
            ].empty:

                self._show_error(
                    output_frame,
                    (
                        "No completed trades are available "
                        "for performance attribution."
                    )
                )

                return

            starting_capital = (
                simulation_results[
                    "statistics"
                ][
                    "starting_capital"
                ]
            )

            attribution = (
                self.performance_analyzer
                .analyze(
                    simulation_results[
                        "simulation_data"
                    ],
                    simulation_results[
                        "equity_curve"
                    ],
                    starting_capital
                )
            )

            if not attribution[
                "success"
            ]:

                self._show_error(
                    output_frame,
                    attribution[
                        "message"
                    ]
                )

                return

            self._show_performance_results(
                output_frame,
                simulation_results,
                attribution,
                time_range
            )

        # ==================================================
        # Live/current monitoring
        # ==================================================

        def show_live_monitor():

            self.live_monitoring_view.show(
                output_frame,
                spread_data,
                time_range,
                window=window,
                entry_threshold=(
                    entry_threshold
                ),
                exit_threshold=(
                    exit_threshold
                ),
                transaction_cost=(
                    transaction_cost
                ),
                capital=float(
                    capital_variable.get()
                ),
                risk_per_trade=float(
                    risk_variable.get()
                ),
                max_position=float(
                    position_variable.get()
                ),
                max_exposure=float(
                    exposure_variable.get()
                )
            )

        # ==================================================
        # Buttons
        # ==================================================

        tk.Button(
            button_frame,
            text="Portfolio Summary",
            font=(
                "Arial",
                9,
                "bold"
            ),
            command=show_portfolio_summary
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Performance Dashboard",
            font=(
                "Arial",
                9,
                "bold"
            ),
            command=show_performance_dashboard
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Current Signal Monitor",
            font=(
                "Arial",
                9,
                "bold"
            ),
            command=show_live_monitor
        ).pack(
            side="left"
        )

        show_portfolio_summary()

    # ==================================================
    # Portfolio summary
    # ==================================================

    def _show_portfolio_results(
        self,
        container,
        results,
        time_range
    ):

        statistics = results[
            "statistics"
        ]

        simulation_data = results[
            "simulation_data"
        ]

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
                    "Starting Capital",
                    (
                        f"€{statistics['starting_capital']:,.0f}"
                    )
                ),
                (
                    "Ending Capital",
                    (
                        f"€{statistics['ending_capital']:,.0f}"
                    )
                ),
                (
                    "Portfolio PnL",
                    (
                        f"€{statistics['total_pnl']:+,.0f}"
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
                    "Trades",
                    statistics[
                        "number_of_trades"
                    ]
                )
            ],
            bold=True
        )

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        self._show_stats(
            row_2,
            [
                (
                    "Win Rate",
                    (
                        f"{statistics['win_rate']:.1f}%"
                    )
                ),
                (
                    "Avg Position",
                    (
                        f"{statistics['average_position_size']:.2f} MWh"
                    )
                ),
                (
                    "Peak Exposure",
                    (
                        f"€{statistics['peak_gross_exposure']:,.0f}"
                    )
                ),
                (
                    "Max Leverage",
                    (
                        f"{statistics['maximum_leverage']:.2f}x"
                    )
                ),
                (
                    "Avg Risk Used",
                    (
                        f"{statistics['average_risk_utilization']:.2f}%"
                    )
                ),
                (
                    "Trading Costs",
                    (
                        f"€{statistics['total_costs']:,.0f}"
                    )
                )
            ]
        )

        tk.Label(
            container,
            text=(
                "Risk sizing uses two rolling spread standard "
                "deviations as an estimated adverse move."
            ),
            font=(
                "Arial",
                8
            )
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        if not simulation_data.empty:

            tk.Label(
                container,
                text="Recent Capital-Sized Trades",
                font=(
                    "Arial",
                    9,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=(0, 2)
            )

            self._create_trade_table(
                container,
                simulation_data
            )

        if not results[
            "equity_curve"
        ].empty:

            figure = (
                self._create_portfolio_chart(
                    results,
                    time_range
                )
            )

            self._embed_chart(
                container,
                figure
            )

    # ==================================================
    # Performance dashboard
    # ==================================================

    def _show_performance_results(
        self,
        container,
        simulation_results,
        attribution,
        time_range
    ):

        tk.Label(
            container,
            text="Strategy Performance Dashboard",
            font=(
                "Arial",
                11,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        profit_factor = (
            attribution[
                "profit_factor"
            ]
        )

        if profit_factor == float(
            "inf"
        ):

            profit_factor_text = (
                "∞"
            )

        else:

            profit_factor_text = (
                f"{profit_factor:.2f}"
            )

        self._show_stats(
            row_1,
            [
                (
                    "Total Return",
                    (
                        f"{attribution['total_return_percent']:+.2f}%"
                    )
                ),
                (
                    "Net PnL",
                    (
                        f"€{attribution['total_net_pnl']:+,.0f}"
                    )
                ),
                (
                    "Max Drawdown",
                    (
                        f"{attribution['maximum_drawdown_percent']:.2f}%"
                    )
                ),
                (
                    "Win Rate",
                    (
                        f"{attribution['win_rate']:.1f}%"
                    )
                ),
                (
                    "Profit Factor",
                    profit_factor_text
                ),
                (
                    "Trade Sharpe",
                    (
                        f"{attribution['trade_sharpe_proxy']:.2f}"
                    )
                )
            ],
            bold=True
        )

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
                    "Cost Drag",
                    (
                        f"{attribution['cost_drag_percent']:.1f}%"
                    )
                ),
                (
                    "Return / Drawdown",
                    (
                        f"{attribution['return_to_drawdown']:.2f}"
                    )
                ),
                (
                    "Avg Position",
                    (
                        f"{attribution['average_position']:.2f} MWh"
                    )
                ),
                (
                    "Peak Exposure",
                    (
                        f"€{attribution['peak_exposure']:,.0f}"
                    )
                ),
                (
                    "Max Leverage",
                    (
                        f"{attribution['maximum_leverage']:.2f}x"
                    )
                ),
                (
                    "Profitable Months",
                    (
                        f"{attribution['profitable_month_percentage']:.1f}%"
                    )
                )
            ]
        )

        row_3 = tk.Frame(
            container
        )

        row_3.pack(
            fill="x",
            pady=(0, 7)
        )

        self._show_stats(
            row_3,
            [
                (
                    "Best Month",
                    (
                        f"{attribution['best_month']} "
                        f"({attribution['best_month_return']:+.2f}%)"
                    )
                ),
                (
                    "Worst Month",
                    (
                        f"{attribution['worst_month']} "
                        f"({attribution['worst_month_return']:+.2f}%)"
                    )
                ),
                (
                    "Best Trade",
                    (
                        f"€{attribution['best_trade']:+,.0f}"
                    )
                ),
                (
                    "Worst Trade",
                    (
                        f"€{attribution['worst_trade']:+,.0f}"
                    )
                )
            ]
        )

        tk.Label(
            container,
            text="PnL Attribution by Trade Direction",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 2)
        )

        self._create_direction_table(
            container,
            attribution[
                "direction_attribution"
            ]
        )

        figure = (
            self._create_dashboard_chart(
                simulation_results,
                attribution,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Direction table
    # ==================================================

    def _create_direction_table(
        self,
        container,
        table
    ):

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 6)
        )

        headers = [
            "Direction",
            "Trades",
            "Net PnL",
            "Win %",
            "Avg Trade",
            "Costs"
        ]

        widths = [
            23,
            8,
            13,
            10,
            12,
            12
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
            table.iterrows(),
            start=1
        ):

            values = [
                row[
                    "Direction"
                ],
                (
                    f"{int(row['Trades'])}"
                ),
                (
                    f"€{row['NetPnL']:+,.0f}"
                ),
                (
                    f"{row['WinRate']:.1f}%"
                ),
                (
                    f"€{row['AverageTrade']:+,.0f}"
                ),
                (
                    f"€{row['Costs']:,.0f}"
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
    # Trade table
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
            pady=(0, 6)
        )

        headers = [
            "Direction",
            "Size",
            "Exposure",
            "PnL",
            "Return",
            "Constraint"
        ]

        widths = [
            23,
            10,
            13,
            12,
            10,
            15
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

        recent_trades = (
            simulation_data.tail(
                5
            )
        )

        for (
            row_number,
            (_, trade)
        ) in enumerate(
            recent_trades.iterrows(),
            start=1
        ):

            values = [
                trade[
                    "Direction"
                ],
                (
                    f"{trade['PositionSizeMWh']:.2f} MWh"
                ),
                (
                    f"€{trade['GrossExposureEUR']:,.0f}"
                ),
                (
                    f"€{trade['NetPnLEUR']:+,.0f}"
                ),
                (
                    f"{trade['TradeReturnPercent']:+.2f}%"
                ),
                trade[
                    "BindingConstraint"
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
    # Portfolio chart
    # ==================================================

    def _create_portfolio_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                4.2
            ),
            dpi=100
        )

        equity_curve = (
            results[
                "equity_curve"
            ]
        )

        equity_axis = (
            figure.add_subplot(
                211
            )
        )

        equity_axis.plot(
            equity_curve[
                "TimeDK"
            ],
            equity_curve[
                "PortfolioValue"
            ],
            label="Portfolio Value",
            linewidth=2
        )

        equity_axis.axhline(
            y=(
                results[
                    "statistics"
                ][
                    "starting_capital"
                ]
            ),
            linestyle="--",
            linewidth=1,
            label="Starting Capital"
        )

        equity_axis.set_title(
            (
                "Capital-Based Strategy Portfolio "
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

        drawdown_axis = (
            figure.add_subplot(
                212
            )
        )

        drawdown_axis.plot(
            equity_curve[
                "TimeDK"
            ],
            equity_curve[
                "DrawdownPercent"
            ],
            label="Portfolio Drawdown"
        )

        drawdown_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        drawdown_axis.set_xlabel(
            "Time"
        )

        drawdown_axis.set_ylabel(
            "Drawdown (%)"
        )

        drawdown_axis.grid(
            True,
            alpha=0.3
        )

        drawdown_axis.legend(
            fontsize=8
        )

        self._format_dates(
            equity_axis
        )

        self._format_dates(
            drawdown_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Dashboard chart
    # ==================================================

    def _create_dashboard_chart(
        self,
        simulation_results,
        attribution,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                6
            ),
            dpi=100
        )

        equity_curve = (
            simulation_results[
                "equity_curve"
            ]
        )

        simulation_data = (
            simulation_results[
                "simulation_data"
            ]
        )

        monthly = (
            attribution[
                "monthly_performance"
            ]
        )

        equity_axis = (
            figure.add_subplot(
                221
            )
        )

        equity_axis.plot(
            equity_curve[
                "TimeDK"
            ],
            equity_curve[
                "PortfolioValue"
            ]
        )

        equity_axis.set_title(
            "Portfolio Value"
        )

        equity_axis.grid(
            True,
            alpha=0.3
        )

        self._format_dates(
            equity_axis
        )

        monthly_axis = (
            figure.add_subplot(
                222
            )
        )

        if not monthly.empty:

            monthly_axis.bar(
                monthly[
                    "Month"
                ],
                monthly[
                    "MonthlyReturnPercent"
                ]
            )

            monthly_axis.tick_params(
                axis="x",
                rotation=45,
                labelsize=7
            )

        monthly_axis.axhline(
            y=0,
            linewidth=1
        )

        monthly_axis.set_title(
            "Monthly Returns"
        )

        monthly_axis.set_ylabel(
            "%"
        )

        drawdown_axis = (
            figure.add_subplot(
                223
            )
        )

        drawdown_axis.plot(
            equity_curve[
                "TimeDK"
            ],
            equity_curve[
                "DrawdownPercent"
            ]
        )

        drawdown_axis.axhline(
            y=0,
            linewidth=1
        )

        drawdown_axis.set_title(
            "Portfolio Drawdown"
        )

        drawdown_axis.set_ylabel(
            "%"
        )

        self._format_dates(
            drawdown_axis
        )

        exposure_axis = (
            figure.add_subplot(
                224
            )
        )

        exposure_axis.plot(
            simulation_data[
                "EntryTime"
            ],
            simulation_data[
                "GrossExposureEUR"
            ],
            marker="o",
            markersize=3
        )

        exposure_axis.set_title(
            "Gross Exposure"
        )

        exposure_axis.set_ylabel(
            "EUR"
        )

        self._format_dates(
            exposure_axis
        )

        figure.suptitle(
            (
                "Strategy Performance Dashboard "
                f"- {time_range}"
            ),
            fontsize=12
        )

        figure.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96
            ]
        )

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
                padx=8
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