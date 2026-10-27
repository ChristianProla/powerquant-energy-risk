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
                "& Position Sizing"
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
                "Convert spread-strategy signals into "
                "capital-based positions using a volatility "
                "risk budget, exposure limits and position caps."
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
        # Current strategy
        # ==================================================

        strategy_frame = tk.Frame(
            container
        )

        strategy_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        strategy_values = [
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

        self._show_stats(
            strategy_frame,
            strategy_values
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
            padx=(5, 10)
        )

        tk.Label(
            controls,
            text="%"
        ).pack(
            side="left"
        )

        # ==================================================
        # Output
        # ==================================================

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # Run simulation
        # ==================================================

        def run_simulation():

            self._clear_frame(
                output_frame
            )

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

            self._show_results(
                output_frame,
                results,
                time_range
            )

        tk.Button(
            controls,
            text="Run Portfolio Simulation",
            font=(
                "Arial",
                9,
                "bold"
            ),
            command=run_simulation
        ).pack(
            side="left",
            padx=(15, 0)
        )

        run_simulation()

    # ==================================================
    # Results
    # ==================================================

    def _show_results(
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

        # ==================================================
        # Portfolio performance
        # ==================================================

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        values_1 = [
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
        ]

        self._show_stats(
            row_1,
            values_1,
            bold=True
        )

        # ==================================================
        # Risk / exposure
        # ==================================================

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        values_2 = [
            (
                "Win Rate",
                (
                    f"{statistics['win_rate']:.1f}%"
                )
            ),
            (
                "Avg Position",
                (
                    f"{statistics['average_position_size']:.2f} "
                    f"MWh"
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

        self._show_stats(
            row_2,
            values_2
        )

        # ==================================================
        # Explanation
        # ==================================================

        tk.Label(
            container,
            text=(
                "Risk sizing uses two rolling spread standard "
                "deviations as an estimated adverse move. "
                "This is a sizing model, not a guaranteed stop-loss."
            ),
            font=(
                "Arial",
                8
            )
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        # ==================================================
        # Recent trades
        # ==================================================

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

        # ==================================================
        # Equity chart
        # ==================================================

        if not results[
            "equity_curve"
        ].empty:

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

        else:

            tk.Label(
                container,
                text=results[
                    "message"
                ],
                font=(
                    "Arial",
                    10
                )
            ).pack(
                anchor="w",
                pady=20
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

    def _create_chart(
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

        equity_axis = (
            figure.add_subplot(
                211
            )
        )

        equity_curve = (
            results[
                "equity_curve"
            ]
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

        # ==================================================
        # Drawdown
        # ==================================================

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