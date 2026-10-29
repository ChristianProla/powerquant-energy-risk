import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

import matplotlib.dates as mdates

from src.strategy.live_monitoring import (
    LiveStrategyMonitor
)


class LiveStrategyMonitoringView:

    def __init__(
        self,
        spread_analyzer
    ):

        self.monitor = (
            LiveStrategyMonitor(
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
        transaction_cost,
        capital,
        risk_per_trade,
        max_position,
        max_exposure
    ):

        self._clear_frame(
            container
        )

        results = (
            self.monitor
            .analyse_current_state(
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

        # ==================================================
        # Title
        # ==================================================

        tk.Label(
            container,
            text="Current Strategy Monitor",
            font=(
                "Arial",
                11,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        tk.Label(
            container,
            text=(
                "Latest available DK1/DK2 spread state "
                "using the selected strategy and risk settings."
            ),
            font=(
                "Arial",
                8
            )
        ).pack(
            anchor="w",
            pady=(0, 7)
        )

        # ==================================================
        # Signal
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
                    "Current Signal",
                    results[
                        "signal"
                    ]
                ),
                (
                    "Direction",
                    results[
                        "direction"
                    ]
                ),
                (
                    "Market State",
                    results[
                        "market_state"
                    ]
                ),
                (
                    "Signal Strength",
                    (
                        f"{results['signal_strength_percent']:.0f}%"
                    )
                ),
                (
                    "Last Observation",
                    self._format_time(
                        results[
                            "current_time"
                        ]
                    )
                )
            ],
            bold=True
        )

        # ==================================================
        # Current market state
        # ==================================================

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 5)
        )

        self._show_stats(
            row_2,
            [
                (
                    "DK1",
                    (
                        f"{results['dk1_price']:.2f} "
                        f"EUR/MWh"
                    )
                ),
                (
                    "DK2",
                    (
                        f"{results['dk2_price']:.2f} "
                        f"EUR/MWh"
                    )
                ),
                (
                    "Spread",
                    (
                        f"{results['spread']:+.2f}"
                    )
                ),
                (
                    "Rolling Mean",
                    (
                        f"{results['rolling_mean']:+.2f}"
                    )
                ),
                (
                    "Z-Score",
                    (
                        f"{results['z_score']:+.2f}"
                    )
                ),
                (
                    "Entry Distance",
                    (
                        f"{results['distance_to_entry_z']:.2f} Z"
                    )
                )
            ]
        )

        # ==================================================
        # Position / risk
        # ==================================================

        row_3 = tk.Frame(
            container
        )

        row_3.pack(
            fill="x",
            pady=(0, 5)
        )

        self._show_stats(
            row_3,
            [
                (
                    "Risk Budget",
                    (
                        f"€{results['risk_budget']:,.0f}"
                    )
                ),
                (
                    "Suggested Size",
                    (
                        f"{results['active_position_size']:.2f} "
                        f"MWh"
                    )
                ),
                (
                    "Gross Exposure",
                    (
                        f"€{results['gross_exposure']:,.0f}"
                    )
                ),
                (
                    "Leverage",
                    (
                        f"{results['leverage']:.2f}x"
                    )
                ),
                (
                    "Risk / MWh",
                    (
                        f"€{results['estimated_risk_per_mwh']:.2f}"
                    )
                ),
                (
                    "Constraint",
                    results[
                        "binding_constraint"
                    ]
                )
            ]
        )

        # ==================================================
        # Mean reversion
        # ==================================================

        row_4 = tk.Frame(
            container
        )

        row_4.pack(
            fill="x",
            pady=(0, 7)
        )

        self._show_stats(
            row_4,
            [
                (
                    "Gap to Mean",
                    (
                        f"{results['spread_deviation']:+.2f} "
                        f"EUR/MWh"
                    )
                ),
                (
                    "Mean-Reversion Move",
                    (
                        f"{results['mean_reversion_move']:.2f} "
                        f"EUR/MWh"
                    )
                ),
                (
                    "Potential Gross Move",
                    (
                        f"€{results['potential_gross_move_eur']:,.0f}"
                    )
                ),
                (
                    "6h Spread Change",
                    (
                        f"{results['spread_momentum']:+.2f}"
                    )
                ),
                (
                    "6h Z Change",
                    (
                        f"{results['z_momentum']:+.2f}"
                    )
                )
            ]
        )

        # ==================================================
        # Interpretation
        # ==================================================

        tk.Label(
            container,
            text=results[
                "interpretation"
            ],
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 2)
        )

        tk.Label(
            container,
            text=(
                "Suggested position size and potential mean-reversion "
                "move are research estimates only. No trade is "
                "executed automatically."
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
    # Chart
    # ==================================================

    def _create_chart(
        self,
        results,
        time_range
    ):

        data = results[
            "chart_data"
        ]

        figure = Figure(
            figsize=(
                10,
                4.5
            ),
            dpi=100
        )

        spread_axis = (
            figure.add_subplot(
                211
            )
        )

        spread_axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "Spread"
            ],
            label="Spread"
        )

        spread_axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "RollingMean"
            ],
            linestyle="--",
            label="Rolling Mean"
        )

        spread_axis.scatter(
            [
                data[
                    "TimeDK"
                ].iloc[
                    -1
                ]
            ],
            [
                data[
                    "Spread"
                ].iloc[
                    -1
                ]
            ],
            marker="o",
            s=50,
            label="Current"
        )

        spread_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        spread_axis.set_title(
            (
                "Current DK1 / DK2 Strategy State "
                f"- {time_range}"
            )
        )

        spread_axis.set_ylabel(
            "Spread (EUR/MWh)"
        )

        spread_axis.grid(
            True,
            alpha=0.3
        )

        spread_axis.legend(
            fontsize=8
        )

        # ==================================================
        # Z-score
        # ==================================================

        z_axis = (
            figure.add_subplot(
                212
            )
        )

        z_axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "ZScore"
            ],
            label="Z-Score"
        )

        z_axis.axhline(
            y=(
                results[
                    "entry_threshold"
                ]
            ),
            linestyle="--",
            label=(
                f"+{results['entry_threshold']:.1f} Entry"
            )
        )

        z_axis.axhline(
            y=-(
                results[
                    "entry_threshold"
                ]
            ),
            linestyle="--",
            label=(
                f"-{results['entry_threshold']:.1f} Entry"
            )
        )

        z_axis.axhline(
            y=(
                results[
                    "exit_threshold"
                ]
            ),
            linestyle=":",
            alpha=0.7
        )

        z_axis.axhline(
            y=-(
                results[
                    "exit_threshold"
                ]
            ),
            linestyle=":",
            alpha=0.7
        )

        z_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        z_axis.scatter(
            [
                data[
                    "TimeDK"
                ].iloc[
                    -1
                ]
            ],
            [
                results[
                    "z_score"
                ]
            ],
            marker="o",
            s=50,
            label="Current Z"
        )

        z_axis.set_xlabel(
            "Time"
        )

        z_axis.set_ylabel(
            "Z-Score"
        )

        z_axis.grid(
            True,
            alpha=0.3
        )

        z_axis.legend(
            fontsize=8
        )

        self._format_dates(
            spread_axis
        )

        self._format_dates(
            z_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Helpers
    # ==================================================

    def _format_time(
        self,
        value
    ):

        try:

            return value.strftime(
                "%Y-%m-%d %H:%M"
            )

        except Exception:

            return str(
                value
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