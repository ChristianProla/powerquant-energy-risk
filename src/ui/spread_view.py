import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.strategy.backtesting import (
    SpreadBacktester
)

from src.strategy.robustness import (
    StrategyRobustnessAnalyzer
)

from src.ui.out_of_sample_view import (
    OutOfSampleView
)


class SpreadView:

    def __init__(
        self,
        spread_analyzer
    ):

        self.spread_analyzer = (
            spread_analyzer
        )

        self.backtester = (
            SpreadBacktester()
        )

        self.robustness_analyzer = (
            StrategyRobustnessAnalyzer(
                self.spread_analyzer,
                self.backtester
            )
        )

        self.out_of_sample_view = (
            OutOfSampleView(
                self.spread_analyzer,
                self.backtester
            )
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

        tk.Label(
            container,
            text=(
                "DK1 / DK2 Spread "
                "& Strategy Research"
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
                "Analyse signals, backtest mean reversion, "
                "test parameter robustness and validate "
                "the strategy on unseen market data."
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
        # Spread data
        # ==================================================

        spread_data = (
            self.spread_analyzer
            .prepare_spread_data(
                dk1_prices,
                dk2_prices
            )
        )

        if spread_data.empty:

            self._show_error(
                container,
                (
                    "Not enough matching DK1 "
                    "and DK2 price observations."
                )
            )

            return

        # ==================================================
        # Strategy controls
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
            text="Window:"
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
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Entry Z:"
        ).pack(
            side="left"
        )

        entry_threshold_variable = (
            tk.StringVar(
                value="2.0"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                entry_threshold_variable
            ),
            values=[
                "1.0",
                "1.5",
                "2.0",
                "2.5",
                "3.0"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Exit Z:"
        ).pack(
            side="left"
        )

        exit_threshold_variable = (
            tk.StringVar(
                value="0.0"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                exit_threshold_variable
            ),
            values=[
                "0.0",
                "0.5",
                "1.0"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
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
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        # ==================================================
        # Navigation buttons
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
        # Spread signal analysis
        # ==================================================

        def run_signal_analysis():

            self._clear_frame(
                output_frame
            )

            window = int(
                window_variable.get()
            )

            threshold = float(
                entry_threshold_variable.get()
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

            self._show_signal_results(
                output_frame,
                signal_data,
                statistics,
                window,
                threshold,
                time_range
            )

        # ==================================================
        # Backtest
        # ==================================================

        def run_backtest():

            self._clear_frame(
                output_frame
            )

            window = int(
                window_variable.get()
            )

            entry_threshold = float(
                entry_threshold_variable.get()
            )

            exit_threshold = float(
                exit_threshold_variable.get()
            )

            transaction_cost = float(
                cost_variable.get()
            )

            signal_data = (
                self.spread_analyzer
                .calculate_signals(
                    spread_data,
                    window=window,
                    threshold=entry_threshold
                )
            )

            results = (
                self.backtester
                .run_backtest(
                    signal_data,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    transaction_cost_per_leg=(
                        transaction_cost
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

            self._show_backtest_results(
                output_frame,
                results,
                window,
                time_range
            )

        # ==================================================
        # Robustness
        # ==================================================

        def run_robustness_test():

            self._clear_frame(
                output_frame
            )

            tk.Label(
                output_frame,
                text=(
                    "Testing 300 strategy "
                    "parameter combinations..."
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
                self.robustness_analyzer
                .run_robustness_test(
                    spread_data
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

            self._show_robustness_results(
                output_frame,
                results,
                time_range
            )

        # ==================================================
        # Out-of-sample
        # ==================================================

        def run_out_of_sample():

            self.out_of_sample_view.show(
                output_frame,
                spread_data,
                time_range
            )

        # ==================================================
        # Buttons
        # ==================================================

        tk.Button(
            button_frame,
            text="Spread Signals",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_signal_analysis
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Strategy Backtest",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_backtest
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Robustness Test",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_robustness_test
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Out-of-Sample Test",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_out_of_sample
        ).pack(
            side="left"
        )

        run_signal_analysis()

    # ==================================================
    # Signal results
    # ==================================================

    def _show_signal_results(
        self,
        container,
        signal_data,
        statistics,
        window,
        threshold,
        time_range
    ):

        tk.Label(
            container,
            text="Spread Signal Analysis",
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        values_1 = [
            (
                "Current Spread",
                (
                    f"{statistics['current_spread']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Current Z",
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
                    f"{statistics['rolling_mean']:+.2f}"
                )
            ),
            (
                "Rolling Std.",
                (
                    f"{statistics['rolling_std']:.2f}"
                )
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
                "Average Spread",
                (
                    f"{statistics['average_spread']:+.2f}"
                )
            ),
            (
                "Maximum Spread",
                (
                    f"{statistics['maximum_spread']:+.2f}"
                )
            ),
            (
                "Minimum Spread",
                (
                    f"{statistics['minimum_spread']:+.2f}"
                )
            ),
            (
                "Upper Signals",
                statistics[
                    "upper_signals"
                ]
            ),
            (
                "Lower Signals",
                statistics[
                    "lower_signals"
                ]
            ),
            (
                "Total Signals",
                statistics[
                    "total_signals"
                ]
            )
        ]

        self._show_stats(
            row_2,
            values_2
        )

        figure = (
            self._create_signal_chart(
                signal_data,
                window,
                threshold,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Backtest results
    # ==================================================

    def _show_backtest_results(
        self,
        container,
        results,
        window,
        time_range
    ):

        statistics = results[
            "statistics"
        ]

        backtest_data = results[
            "backtest_data"
        ]

        trades = results[
            "trades"
        ]

        tk.Label(
            container,
            text=(
                "Mean-Reversion "
                "Strategy Backtest"
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
                "Research backtest only. PnL is measured "
                "in spread EUR/MWh and does not yet model "
                "capital, margin or execution constraints."
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 7)
        )

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        values_1 = [
            (
                "Net PnL",
                (
                    f"{statistics['total_net_pnl']:+.2f}"
                )
            ),
            (
                "Gross PnL",
                (
                    f"{statistics['total_gross_pnl']:+.2f}"
                )
            ),
            (
                "Costs",
                (
                    f"{statistics['total_transaction_cost']:.2f}"
                )
            ),
            (
                "Trades",
                statistics[
                    "number_of_trades"
                ]
            ),
            (
                "Win Rate",
                (
                    f"{statistics['win_rate']:.1f}%"
                )
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
            pady=(0, 6)
        )

        profit_factor = (
            statistics[
                "profit_factor"
            ]
        )

        if (
            profit_factor
            == float(
                "inf"
            )
        ):

            profit_factor_text = (
                "∞"
            )

        else:

            profit_factor_text = (
                f"{profit_factor:.2f}"
            )

        values_2 = [
            (
                "Avg Trade",
                (
                    f"{statistics['average_trade']:+.2f}"
                )
            ),
            (
                "Best Trade",
                (
                    f"{statistics['best_trade']:+.2f}"
                )
            ),
            (
                "Worst Trade",
                (
                    f"{statistics['worst_trade']:+.2f}"
                )
            ),
            (
                "Sharpe Proxy",
                (
                    f"{statistics['sharpe_proxy']:.2f}"
                )
            ),
            (
                "Max Drawdown",
                (
                    f"{statistics['maximum_drawdown']:.2f}"
                )
            ),
            (
                "Profit Factor",
                profit_factor_text
            )
        ]

        self._show_stats(
            row_2,
            values_2
        )

        figure = (
            self._create_backtest_chart(
                backtest_data,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Robustness results
    # ==================================================

    def _show_robustness_results(
        self,
        container,
        results,
        time_range
    ):

        tk.Label(
            container,
            text=(
                "Strategy Robustness "
                "& Parameter Sensitivity"
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

        row_1 = tk.Frame(
            container
        )

        row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        values = [
            (
                "Robustness",
                results[
                    "robustness_label"
                ]
            ),
            (
                "Configurations",
                results[
                    "total_combinations"
                ]
            ),
            (
                "Profitable",
                (
                    f"{results['profitable_percentage']:.1f}%"
                )
            ),
            (
                "Positive Sharpe",
                (
                    f"{results['positive_sharpe_percentage']:.1f}%"
                )
            ),
            (
                "Median PnL",
                (
                    f"{results['median_net_pnl']:+.2f}"
                )
            ),
            (
                "Median Sharpe",
                (
                    f"{results['median_sharpe']:.2f}"
                )
            )
        ]

        self._show_stats(
            row_1,
            values,
            bold=True
        )

        robust = results[
            "robust_configuration"
        ]

        row_2 = tk.Frame(
            container
        )

        row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        values_2 = [
            (
                "Robust Window",
                (
                    f"{int(robust['Window'])}h"
                )
            ),
            (
                "Entry Z",
                (
                    f"{robust['EntryZ']:.1f}"
                )
            ),
            (
                "Avg Sharpe",
                (
                    f"{robust['AverageSharpe']:.2f}"
                )
            ),
            (
                "Avg PnL",
                (
                    f"{robust['AverageNetPnL']:+.2f}"
                )
            ),
            (
                "Profitable Share",
                (
                    f"{robust['ProfitableShare']:.1f}%"
                )
            )
        ]

        self._show_stats(
            row_2,
            values_2
        )

        figure = (
            self._create_robustness_chart(
                results,
                time_range
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Signal chart
    # ==================================================

    def _create_signal_chart(
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
                f"{window}-Hour Rolling Mean"
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
                f"DK1 / DK2 Spread "
                f"- {time_range}"
            )
        )

        spread_axis.set_ylabel(
            "EUR/MWh"
        )

        spread_axis.grid(
            True,
            alpha=0.3
        )

        spread_axis.legend(
            fontsize=8
        )

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
            label="Spread Z-Score"
        )

        z_axis.axhline(
            y=threshold,
            linestyle="--",
            label=(
                f"+{threshold:.1f}"
            )
        )

        z_axis.axhline(
            y=-threshold,
            linestyle="--",
            label=(
                f"-{threshold:.1f}"
            )
        )

        z_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
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

        self._format_dates(
            spread_axis
        )

        self._format_dates(
            z_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Backtest chart
    # ==================================================

    def _create_backtest_chart(
        self,
        data,
        time_range
    ):

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
            label="DK1 - DK2 Spread"
        )

        entries = data.dropna(
            subset=[
                "EntryMarker"
            ]
        )

        exits = data.dropna(
            subset=[
                "ExitMarker"
            ]
        )

        if not entries.empty:

            spread_axis.scatter(
                entries[
                    "TimeDK"
                ],
                entries[
                    "EntryMarker"
                ],
                marker="^",
                label="Entry"
            )

        if not exits.empty:

            spread_axis.scatter(
                exits[
                    "TimeDK"
                ],
                exits[
                    "ExitMarker"
                ],
                marker="x",
                label="Exit"
            )

        spread_axis.set_title(
            (
                f"Mean-Reversion Backtest "
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

        pnl_axis = (
            figure.add_subplot(
                212
            )
        )

        pnl_axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "CumulativeGrossPnL"
            ],
            label="Gross PnL",
            linestyle="--"
        )

        pnl_axis.plot(
            data[
                "TimeDK"
            ],
            data[
                "CumulativeNetPnL"
            ],
            label="Net PnL",
            linewidth=2
        )

        pnl_axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        pnl_axis.set_ylabel(
            "Cumulative PnL"
        )

        pnl_axis.set_xlabel(
            "Time"
        )

        pnl_axis.grid(
            True,
            alpha=0.3
        )

        pnl_axis.legend(
            fontsize=8
        )

        self._format_dates(
            spread_axis
        )

        self._format_dates(
            pnl_axis
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Robustness heatmap
    # ==================================================

    def _create_robustness_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                4
            ),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        heatmap_data = (
            results[
                "heatmap_data"
            ]
        )

        values = (
            heatmap_data.to_numpy(
                dtype=float
            )
        )

        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower"
        )

        axis.set_title(
            (
                "Average Sharpe Proxy Across "
                f"Parameter Regions - {time_range}"
            )
        )

        axis.set_xlabel(
            "Entry Z-Score"
        )

        axis.set_ylabel(
            "Rolling Window"
        )

        axis.set_xticks(
            range(
                len(
                    heatmap_data.columns
                )
            )
        )

        axis.set_xticklabels(
            [
                f"{value:.1f}"
                for value
                in heatmap_data.columns
            ]
        )

        axis.set_yticks(
            range(
                len(
                    heatmap_data.index
                )
            )
        )

        axis.set_yticklabels(
            [
                f"{int(value)}h"
                for value
                in heatmap_data.index
            ]
        )

        for row_index in range(
            values.shape[
                0
            ]
        ):

            for column_index in range(
                values.shape[
                    1
                ]
            ):

                value = values[
                    row_index,
                    column_index
                ]

                if not np.isnan(
                    value
                ):

                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8
                    )

        figure.colorbar(
            image,
            ax=axis,
            label="Average Sharpe Proxy"
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