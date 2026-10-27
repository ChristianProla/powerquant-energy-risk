import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.strategy.out_of_sample import (
    OutOfSampleValidator
)


class OutOfSampleView:

    def __init__(
        self,
        spread_analyzer,
        backtester
    ):

        self.validator = (
            OutOfSampleValidator(
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
        time_range
    ):

        self._clear_frame(
            container
        )

        tk.Label(
            container,
            text=(
                "Out-of-Sample "
                "Strategy Validation"
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
                "Strategy parameters are selected using "
                "only the training period, frozen, and then "
                "tested on completely unseen later data."
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
            text="Train / Test:"
        ).pack(
            side="left"
        )

        split_variable = (
            tk.StringVar(
                value="70 / 30"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                split_variable
            ),
            values=[
                "60 / 40",
                "70 / 30",
                "80 / 20"
            ],
            state="readonly",
            width=10
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
        # Run validation
        # ==================================================

        def run_validation():

            self._clear_frame(
                output_frame
            )

            split_lookup = {
                "60 / 40":
                    0.60,

                "70 / 30":
                    0.70,

                "80 / 20":
                    0.80
            }

            train_ratio = (
                split_lookup[
                    split_variable.get()
                ]
            )

            transaction_cost = float(
                cost_variable.get()
            )

            loading_label = tk.Label(
                output_frame,
                text=(
                    "Selecting parameters on training data "
                    "and validating on unseen data..."
                ),
                font=(
                    "Arial",
                    10,
                    "bold"
                )
            )

            loading_label.pack(
                anchor="w",
                pady=20
            )

            container.update()

            results = (
                self.validator
                .run_validation(
                    spread_data,
                    train_ratio=train_ratio,
                    transaction_cost_per_leg=(
                        transaction_cost
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
            text="Run Out-of-Sample Test",
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

        # --------------------------------------------------
        # Main interpretation
        # --------------------------------------------------

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 6)
        )

        values_1 = [
            (
                "Generalization",
                results[
                    "generalization_label"
                ]
            ),
            (
                "Train / Test",
                (
                    f"{results['train_ratio'] * 100:.0f}% / "
                    f"{results['test_ratio'] * 100:.0f}%"
                )
            ),
            (
                "Selected Window",
                (
                    f"{results['selected_window']}h"
                )
            ),
            (
                "Entry Z",
                (
                    f"±{results['selected_entry']:.1f}"
                )
            ),
            (
                "Exit Z",
                (
                    f"±{results['selected_exit']:.1f}"
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
            row_1,
            values_1,
            bold=True
        )

        # ==================================================
        # Train vs test table
        # ==================================================

        tk.Label(
            container,
            text="In-Sample vs Out-of-Sample",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        self._create_comparison_table(
            container,
            results
        )

        # ==================================================
        # Generalization metrics
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
                "Sharpe Retention",
                (
                    f"{results['sharpe_retention']:.1f}%"
                )
            ),
            (
                "Avg Trade Retention",
                (
                    f"{results['average_trade_retention']:.1f}%"
                )
            ),
            (
                "Train Obs.",
                (
                    f"{results['train_observations']:,}"
                )
            ),
            (
                "Test Obs.",
                (
                    f"{results['test_observations']:,}"
                )
            )
        ]

        self._show_stats(
            row_2,
            values_2
        )

        # ==================================================
        # Top train configurations
        # ==================================================

        tk.Label(
            container,
            text=(
                "Top Training Configurations "
                "(test data not used)"
            ),
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 3)
        )

        self._create_parameter_table(
            container,
            results[
                "top_training_configurations"
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
    # Train / test comparison table
    # ==================================================

    def _create_comparison_table(
        self,
        container,
        results
    ):

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 7)
        )

        train = results[
            "train_statistics"
        ]

        test = results[
            "test_statistics"
        ]

        headers = [
            "Metric",
            "In-Sample",
            "Out-of-Sample"
        ]

        widths = [
            22,
            18,
            18
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
                    9,
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

        rows = [
            (
                "Net PnL",
                (
                    f"{train['total_net_pnl']:+.2f}"
                ),
                (
                    f"{test['total_net_pnl']:+.2f}"
                )
            ),
            (
                "Sharpe Proxy",
                (
                    f"{train['sharpe_proxy']:.2f}"
                ),
                (
                    f"{test['sharpe_proxy']:.2f}"
                )
            ),
            (
                "Win Rate",
                (
                    f"{train['win_rate']:.1f}%"
                ),
                (
                    f"{test['win_rate']:.1f}%"
                )
            ),
            (
                "Maximum Drawdown",
                (
                    f"{train['maximum_drawdown']:.2f}"
                ),
                (
                    f"{test['maximum_drawdown']:.2f}"
                )
            ),
            (
                "Average Trade",
                (
                    f"{train['average_trade']:+.2f}"
                ),
                (
                    f"{test['average_trade']:+.2f}"
                )
            ),
            (
                "Trades",
                (
                    f"{train['number_of_trades']}"
                ),
                (
                    f"{test['number_of_trades']}"
                )
            )
        ]

        for (
            row_number,
            row_values
        ) in enumerate(
            rows,
            start=1
        ):

            for (
                column,
                value
            ) in enumerate(
                row_values
            ):

                tk.Label(
                    frame,
                    text=value,
                    font=(
                        "Arial",
                        9
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
    # Parameter table
    # ==================================================

    def _create_parameter_table(
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

        columns = [
            "Window",
            "Entry",
            "Exit",
            "PnL",
            "Sharpe",
            "Trades",
            "Win %"
        ]

        widths = [
            9,
            9,
            9,
            12,
            11,
            9,
            10
        ]

        for (
            column,
            name
        ) in enumerate(
            columns
        ):

            tk.Label(
                frame,
                text=name,
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
            table.head(
                5
            ).iterrows(),
            start=1
        ):

            values = [
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
                    f"{row['NetPnL']:+.2f}"
                ),
                (
                    f"{row['SharpeProxy']:.2f}"
                ),
                (
                    f"{int(row['Trades'])}"
                ),
                (
                    f"{row['WinRate']:.1f}%"
                )
            ]

            for (
                column,
                value
            ) in enumerate(
                values
            ):

                font = (
                    "Arial",
                    8,
                    "bold"
                ) if (
                    row_number == 1
                ) else (
                    "Arial",
                    8
                )

                tk.Label(
                    frame,
                    text=value,
                    font=font,
                    width=widths[
                        column
                    ],
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column
                )

    # ==================================================
    # Equity chart
    # ==================================================

    def _create_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                3.7
            ),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        train_data = results[
            "train_backtest_data"
        ]

        test_data = results[
            "test_backtest_data"
        ]

        axis.plot(
            train_data[
                "TimeDK"
            ],
            train_data[
                "CumulativeNetPnL"
            ],
            label="In-Sample PnL",
            linewidth=2
        )

        axis.plot(
            test_data[
                "TimeDK"
            ],
            test_data[
                "CumulativeNetPnL"
            ],
            label="Out-of-Sample PnL",
            linewidth=2
        )

        if not test_data.empty:

            axis.axvline(
                x=test_data[
                    "TimeDK"
                ].iloc[
                    0
                ],
                linestyle="--",
                linewidth=1.5,
                label="Out-of-Sample Start"
            )

        axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        axis.set_title(
            (
                "In-Sample vs Out-of-Sample "
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
                padx=9
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