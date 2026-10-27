import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.strategy.walk_forward import (
    WalkForwardValidator
)


class WalkForwardView:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.validator = (
            WalkForwardValidator(
                spread_analyzer,
                backtester
            )
        )

    # ==================================================
    # Main page
    # ==================================================

    def show(
        self,
        container,
        spread_data,
        time_range
    ):

        self._clear_frame(
            container
        )

        tk.Label(
            container,
            text="Walk-Forward Strategy Validation",
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
                "Parameters are repeatedly selected using "
                "past data only and then tested on the next "
                "unseen market period."
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
            text="Test Folds:"
        ).pack(
            side="left"
        )

        folds_variable = (
            tk.StringVar(
                value="4"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                folds_variable
            ),
            values=[
                "3",
                "4",
                "5"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 20)
        )

        tk.Label(
            controls,
            text="Cost / Leg:"
        ).pack(
            side="left"
        )

        cost_variable = (
            tk.StringVar(
                value="0.25"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                cost_variable
            ),
            values=[
                "0.00",
                "0.25",
                "0.50",
                "1.00"
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
        # Run
        # ==================================================

        def run_validation():

            self._clear_frame(
                output_frame
            )

            tk.Label(
                output_frame,
                text=(
                    "Running repeated parameter selection "
                    "and unseen strategy tests..."
                ),
                font=(
                    "Arial",
                    10,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=20
            )

            container.update()

            results = (
                self.validator
                .run_validation(
                    spread_data,
                    n_folds=int(
                        folds_variable.get()
                    ),
                    transaction_cost_per_leg=float(
                        cost_variable.get()
                    )
                )
            )

            self._clear_frame(
                output_frame
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
            text="Run Walk-Forward Test",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_validation
        ).pack(
            side="left"
        )

        run_validation()

    # ==================================================
    # Results
    # ==================================================

    def _show_results(
        self,
        container,
        results,
        time_range
    ):

        # ==================================================
        # Overall statistics
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
                "Validation",
                results[
                    "validation_label"
                ]
            ),
            (
                "OOS Folds",
                results[
                    "n_folds"
                ]
            ),
            (
                "Total OOS PnL",
                (
                    f"{results['total_oos_pnl']:+.2f}"
                )
            ),
            (
                "Avg OOS Sharpe",
                (
                    f"{results['average_oos_sharpe']:.2f}"
                )
            ),
            (
                "Profitable Folds",
                (
                    f"{results['profitable_fold_percentage']:.1f}%"
                )
            ),
            (
                "Total Trades",
                results[
                    "total_trades"
                ]
            )
        ]

        self._show_stats(
            row_1,
            values_1,
            bold=True
        )

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        values_2 = [
            (
                "Avg Win Rate",
                (
                    f"{results['average_win_rate']:.1f}%"
                )
            ),
            (
                "Positive Sharpe",
                (
                    f"{results['positive_sharpe_percentage']:.1f}%"
                )
            ),
            (
                "Best Fold",
                (
                    f"{results['best_fold_pnl']:+.2f}"
                )
            ),
            (
                "Worst Fold",
                (
                    f"{results['worst_fold_pnl']:+.2f}"
                )
            ),
            (
                "Max Drawdown",
                (
                    f"{results['maximum_drawdown']:.2f}"
                )
            ),
            (
                "Cost / Leg",
                (
                    f"{results['transaction_cost_per_leg']:.2f}"
                )
            )
        ]

        self._show_stats(
            row_2,
            values_2
        )

        # ==================================================
        # Parameter stability
        # ==================================================

        stability = results[
            "parameter_stability"
        ]

        tk.Label(
            container,
            text="Parameter Stability",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        stability_frame = tk.Frame(
            container
        )

        stability_frame.pack(
            fill="x",
            pady=(0, 7)
        )

        stability_values = [
            (
                "Common Window",
                (
                    f"{stability['most_common_window']}h "
                    f"({stability['window_share']:.0f}%)"
                )
            ),
            (
                "Common Entry",
                (
                    f"{stability['most_common_entry']:.1f} "
                    f"({stability['entry_share']:.0f}%)"
                )
            ),
            (
                "Common Exit",
                (
                    f"{stability['most_common_exit']:.1f} "
                    f"({stability['exit_share']:.0f}%)"
                )
            ),
            (
                "Common Configuration",
                (
                    f"{stability['most_common_configuration']} "
                    f"({stability['configuration_share']:.0f}%)"
                )
            )
        ]

        self._show_stats(
            stability_frame,
            stability_values
        )

        # ==================================================
        # Fold table
        # ==================================================

        tk.Label(
            container,
            text="Walk-Forward Folds",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        self._create_fold_table(
            container,
            results[
                "fold_results"
            ]
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
    # Fold table
    # ==================================================

    def _create_fold_table(
        self,
        container,
        fold_results
    ):

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 6)
        )

        headers = [
            "Fold",
            "Window",
            "Entry",
            "Exit",
            "Test PnL",
            "Sharpe",
            "Win %",
            "Trades"
        ]

        widths = [
            7,
            9,
            8,
            8,
            12,
            10,
            10,
            8
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
            fold_results.iterrows(),
            start=1
        ):

            values = [
                (
                    f"{int(row['Fold'])}"
                ),
                (
                    f"{int(row['Window'])}h"
                ),
                (
                    f"{row['EntryZ']:.1f}"
                ),
                (
                    f"{row['ExitZ']:.1f}"
                ),
                (
                    f"{row['TestPnL']:+.2f}"
                ),
                (
                    f"{row['TestSharpe']:.2f}"
                ),
                (
                    f"{row['TestWinRate']:.1f}%"
                ),
                (
                    f"{int(row['TestTrades'])}"
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
    # Walk-forward chart
    # ==================================================

    def _create_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                3.8
            ),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        data = results[
            "combined_test_data"
        ]

        axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "WalkForwardCumulativePnL"
            ],
            label="Walk-Forward OOS PnL",
            linewidth=2
        )

        # --------------------------------------------------
        # Mark fold boundaries
        # --------------------------------------------------

        fold_results = results[
            "fold_results"
        ]

        first_boundary = True

        for _, fold in (
            fold_results.iterrows()
        ):

            if first_boundary:

                label = (
                    "Test Fold Start"
                )

                first_boundary = False

            else:

                label = None

            axis.axvline(
                x=fold[
                    "TestStart"
                ],
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label=label
            )

        axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        axis.set_title(
            (
                "Walk-Forward Out-of-Sample "
                f"Strategy PnL - {time_range}"
            )
        )

        axis.set_xlabel(
            "Time"
        )

        axis.set_ylabel(
            "Cumulative Net PnL"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend(
            fontsize=8
        )

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

        figure.tight_layout()

        return figure

    # ==================================================
    # Helpers
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