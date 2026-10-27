import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)


class SpreadView:

    def __init__(
        self,
        spread_analyzer
    ):

        self.spread_analyzer = (
            spread_analyzer
        )

    # ==================================================
    # Main page
    # ==================================================

    def show(
        self,
        container,
        dk1_prices,
        dk2_prices,
        time_range
    ):

        self._clear_frame(
            container
        )

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        tk.Label(
            container,
            text=(
                "DK1 / DK2 Spread "
                "& Trading Signals"
            ),
            font=(
                "Arial",
                15,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(5, 5)
        )

        tk.Label(
            container,
            text=(
                "Analyse relative electricity prices "
                "between DK1 and DK2 using rolling "
                "spread statistics and z-score signals."
            ),
            font=(
                "Arial",
                10
            )
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        # ==================================================
        # Prepare spread
        # ==================================================

        spread_data = (
            self.spread_analyzer
            .prepare_spread_data(
                dk1_prices,
                dk2_prices
            )
        )

        if spread_data.empty:

            tk.Label(
                container,
                text=(
                    "Not enough matching DK1 "
                    "and DK2 price observations."
                ),
                font=(
                    "Arial",
                    11,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=30
            )

            return

        # ==================================================
        # Controls
        # ==================================================

        controls = tk.Frame(
            container
        )

        controls.pack(
            fill="x",
            pady=(0, 10)
        )

        tk.Label(
            controls,
            text="Rolling Window:"
        ).pack(
            side="left"
        )

        window_variable = (
            tk.StringVar(
                value="24"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                window_variable
            ),
            values=[
                "12",
                "24",
                "48",
                "72",
                "168"
            ],
            state="readonly",
            width=7
        ).pack(
            side="left",
            padx=(5, 20)
        )

        tk.Label(
            controls,
            text="Signal Threshold:"
        ).pack(
            side="left"
        )

        threshold_variable = (
            tk.StringVar(
                value="2.0"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                threshold_variable
            ),
            values=[
                "1.0",
                "1.5",
                "2.0",
                "2.5",
                "3.0"
            ],
            state="readonly",
            width=7
        ).pack(
            side="left",
            padx=(5, 20)
        )

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # Run analysis
        # ==================================================

        def run_analysis():

            self._clear_frame(
                output_frame
            )

            window = int(
                window_variable.get()
            )

            threshold = float(
                threshold_variable.get()
            )

            signal_data = (
                self.spread_analyzer
                .calculate_signals(
                    spread_data,
                    window=window,
                    threshold=threshold
                )
            )

            statistics = (
                self.spread_analyzer
                .get_spread_statistics(
                    signal_data,
                    threshold=threshold
                )
            )

            self._show_results(
                output_frame,
                signal_data,
                statistics,
                window,
                threshold,
                time_range
            )

        tk.Button(
            controls,
            text="Run Spread Analysis",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=(
                run_analysis
            )
        ).pack(
            side="left"
        )

        run_analysis()

    # ==================================================
    # Results
    # ==================================================

    def _show_results(
        self,
        container,
        signal_data,
        statistics,
        window,
        threshold,
        time_range
    ):

        # --------------------------------------------------
        # Current market state
        # --------------------------------------------------

        stats_row_1 = tk.Frame(
            container
        )

        stats_row_1.pack(
            fill="x",
            pady=(0, 6)
        )

        row_1_values = [
            (
                "Current Spread",
                (
                    f"{statistics['current_spread']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Current Z-Score",
                (
                    f"{statistics['current_z_score']:+.2f}"
                )
            ),
            (
                "Signal",
                statistics[
                    "current_signal"
                ]
            ),
            (
                "Rolling Mean",
                (
                    f"{statistics['rolling_mean']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Spread Volatility",
                (
                    f"{statistics['rolling_std']:.2f} "
                    f"EUR/MWh"
                )
            )
        ]

        self._show_stats(
            stats_row_1,
            row_1_values,
            bold=True
        )

        # --------------------------------------------------
        # Historical statistics
        # --------------------------------------------------

        stats_row_2 = tk.Frame(
            container
        )

        stats_row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        row_2_values = [
            (
                "Average Spread",
                (
                    f"{statistics['average_spread']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Maximum Spread",
                (
                    f"{statistics['maximum_spread']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Minimum Spread",
                (
                    f"{statistics['minimum_spread']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Upper Signals",
                (
                    f"{statistics['upper_signals']}"
                )
            ),
            (
                "Lower Signals",
                (
                    f"{statistics['lower_signals']}"
                )
            ),
            (
                "Total Signals",
                (
                    f"{statistics['total_signals']}"
                )
            )
        ]

        self._show_stats(
            stats_row_2,
            row_2_values,
            bold=False
        )

        # --------------------------------------------------
        # Research explanation
        # --------------------------------------------------

        tk.Label(
            container,
            text=(
                "Research signal only: a positive extreme "
                "z-score means DK1 is expensive relative to DK2; "
                "a negative extreme means DK1 is cheap relative to DK2."
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        # --------------------------------------------------
        # Chart
        # --------------------------------------------------

        figure = (
            self._create_chart(
                signal_data,
                window,
                threshold,
                time_range
            )
        )

        canvas = FigureCanvasTkAgg(
            figure,
            master=container
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # Chart
    # ==================================================

    def _create_chart(
        self,
        signal_data,
        window,
        threshold,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                5
            ),
            dpi=100
        )

        # ==================================================
        # Spread chart
        # ==================================================

        spread_axis = (
            figure.add_subplot(
                211
            )
        )

        spread_axis.plot(
            signal_data[
                "TimeDK"
            ],
            signal_data[
                "Spread"
            ],
            label="DK1 - DK2 Spread",
            linewidth=1.8
        )

        spread_axis.plot(
            signal_data[
                "TimeDK"
            ],
            signal_data[
                "RollingMean"
            ],
            label=(
                f"{window}-Hour "
                f"Rolling Mean"
            ),
            linestyle="--"
        )

        spread_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        spread_axis.set_title(
            (
                f"DK1 / DK2 Electricity Price Spread "
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
        # Z-score chart
        # ==================================================

        z_axis = (
            figure.add_subplot(
                212
            )
        )

        z_axis.plot(
            signal_data[
                "TimeDK"
            ],
            signal_data[
                "ZScore"
            ],
            label="Spread Z-Score",
            linewidth=1.8
        )

        z_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        z_axis.axhline(
            y=threshold,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"+{threshold:.1f} Signal"
            )
        )

        z_axis.axhline(
            y=-threshold,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"-{threshold:.1f} Signal"
            )
        )

        # --------------------------------------------------
        # Signal markers
        # --------------------------------------------------

        upper_signals = (
            signal_data[
                signal_data[
                    "ZScore"
                ]
                >= threshold
            ]
        )

        lower_signals = (
            signal_data[
                signal_data[
                    "ZScore"
                ]
                <= -threshold
            ]
        )

        if not (
            upper_signals.empty
        ):

            z_axis.scatter(
                upper_signals[
                    "TimeDK"
                ],
                upper_signals[
                    "ZScore"
                ],
                marker="v",
                label=(
                    "DK1 Expensive Signal"
                )
            )

        if not (
            lower_signals.empty
        ):

            z_axis.scatter(
                lower_signals[
                    "TimeDK"
                ],
                lower_signals[
                    "ZScore"
                ],
                marker="^",
                label=(
                    "DK1 Cheap Signal"
                )
            )

        z_axis.set_ylabel(
            "Z-Score"
        )

        z_axis.set_xlabel(
            "Time"
        )

        z_axis.grid(
            True,
            alpha=0.3
        )

        z_axis.legend(
            fontsize=8
        )

        # --------------------------------------------------
        # Date formatting
        # --------------------------------------------------

        locator = (
            mdates.AutoDateLocator()
        )

        formatter = (
            mdates.ConciseDateFormatter(
                locator
            )
        )

        spread_axis.xaxis.set_major_locator(
            locator
        )

        spread_axis.xaxis.set_major_formatter(
            formatter
        )

        z_axis.xaxis.set_major_locator(
            locator
        )

        z_axis.xaxis.set_major_formatter(
            formatter
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Stats helper
    # ==================================================

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

            if bold:

                font = (
                    "Arial",
                    9,
                    "bold"
                )

            else:

                font = (
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
                padx=10
            ).pack(
                side="left"
            )

    # ==================================================
    # Clear frame
    # ==================================================

    def _clear_frame(
        self,
        frame
    ):

        for widget in (
            frame.winfo_children()
        ):

            widget.destroy()