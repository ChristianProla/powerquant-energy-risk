import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

import matplotlib.dates as mdates

from src.strategy.regime_detection import (
    MarketRegimeDetector
)


class RegimeAnalysisView:

    def __init__(
        self
    ):

        self.detector = (
            MarketRegimeDetector()
        )

    # ==================================================
    # Main view
    # ==================================================

    def show(
        self,
        container,
        spread_data,
        simulation_data,
        time_range,
        signal_window
    ):

        self._clear_frame(
            container
        )

        results = (
            self.detector
            .analyze(
                spread_data,
                simulation_data,
                signal_window=signal_window
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

        current = (
            results[
                "current_regime"
            ]
        )

        # ==================================================
        # Title
        # ==================================================

        tk.Label(
            container,
            text=(
                "Market Regime Detection "
                "& Strategy Attribution"
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
                "Classify current market conditions and "
                "measure how the strategy historically performed "
                "inside different regimes."
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
        # Current regime
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
                    "Volatility",
                    current[
                        "volatility"
                    ]
                ),
                (
                    "Structure",
                    current[
                        "structure"
                    ]
                ),
                (
                    "Spread State",
                    current[
                        "spread_state"
                    ]
                ),
                (
                    "Relative Pricing",
                    current[
                        "relative_price_state"
                    ]
                ),
                (
                    "Current Z",
                    (
                        f"{current['z_score']:+.2f}"
                    )
                ),
                (
                    "Spread Vol.",
                    (
                        f"{current['rolling_volatility']:.2f}"
                    )
                )
            ],
            bold=True
        )

        # ==================================================
        # Historical strategy conclusion
        # ==================================================

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
                    "Best Volatility",
                    results[
                        "best_volatility_regime"
                    ]
                ),
                (
                    "Worst Volatility",
                    results[
                        "worst_volatility_regime"
                    ]
                ),
                (
                    "Best Structure",
                    results[
                        "best_structure_regime"
                    ]
                ),
                (
                    "Trend Strength",
                    (
                        f"{current['trend_strength']:.2f}"
                    )
                )
            ]
        )

        tk.Label(
            container,
            text=results[
                "regime_summary"
            ],
            font=(
                "Arial",
                8
            ),
            wraplength=900,
            justify="left"
        ).pack(
            anchor="w",
            pady=(0, 7)
        )

        # ==================================================
        # Attribution tables
        # ==================================================

        if not results[
            "volatility_attribution"
        ].empty:

            tk.Label(
                container,
                text="Performance by Volatility Regime",
                font=(
                    "Arial",
                    9,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=(0, 2)
            )

            self._create_attribution_table(
                container,
                results[
                    "volatility_attribution"
                ]
            )

        if not results[
            "structure_attribution"
        ].empty:

            tk.Label(
                container,
                text="Performance by Market Structure",
                font=(
                    "Arial",
                    9,
                    "bold"
                )
            ).pack(
                anchor="w",
                pady=(0, 2)
            )

            self._create_attribution_table(
                container,
                results[
                    "structure_attribution"
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
    # Attribution table
    # ==================================================

    def _create_attribution_table(
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
            "Regime",
            "Trades",
            "Net PnL",
            "Avg Trade",
            "Win %",
            "Avg Return"
        ]

        widths = [
            17,
            8,
            13,
            13,
            10,
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
                    "Regime"
                ],
                (
                    f"{int(row['Trades'])}"
                ),
                (
                    f"€{row['NetPnL']:+,.0f}"
                ),
                (
                    f"€{row['AverageTrade']:+,.0f}"
                ),
                (
                    f"{row['WinRate']:.1f}%"
                ),
                (
                    f"{row['AverageReturn']:+.2f}%"
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
    # Chart
    # ==================================================

    def _create_chart(
        self,
        results,
        time_range
    ):

        figure = Figure(
            figsize=(
                10,
                5.5
            ),
            dpi=100
        )

        regime_data = (
            results[
                "regime_data"
            ]
        )

        volatility_table = (
            results[
                "volatility_attribution"
            ]
        )

        structure_table = (
            results[
                "structure_attribution"
            ]
        )

        # ==================================================
        # Rolling volatility
        # ==================================================

        volatility_axis = (
            figure.add_subplot(
                221
            )
        )

        volatility_axis.plot(
            regime_data[
                "TimeDK"
            ],
            regime_data[
                "RollingSpreadVolatility"
            ]
        )

        volatility_axis.set_title(
            "Rolling Spread Volatility"
        )

        volatility_axis.set_ylabel(
            "EUR/MWh"
        )

        volatility_axis.grid(
            True,
            alpha=0.3
        )

        self._format_dates(
            volatility_axis
        )

        # ==================================================
        # Trend strength
        # ==================================================

        trend_axis = (
            figure.add_subplot(
                222
            )
        )

        trend_axis.plot(
            regime_data[
                "TimeDK"
            ],
            regime_data[
                "TrendStrength"
            ]
        )

        trend_axis.axhline(
            y=0.30,
            linestyle="--",
            label="Mean-Reversion Zone"
        )

        trend_axis.axhline(
            y=0.75,
            linestyle="--",
            label="Trending Zone"
        )

        trend_axis.set_title(
            "Spread Trend Strength"
        )

        trend_axis.set_ylabel(
            "Score"
        )

        trend_axis.grid(
            True,
            alpha=0.3
        )

        trend_axis.legend(
            fontsize=7
        )

        self._format_dates(
            trend_axis
        )

        # ==================================================
        # PnL by volatility regime
        # ==================================================

        volatility_pnl_axis = (
            figure.add_subplot(
                223
            )
        )

        if not volatility_table.empty:

            volatility_pnl_axis.bar(
                volatility_table[
                    "Regime"
                ],
                volatility_table[
                    "NetPnL"
                ]
            )

        volatility_pnl_axis.axhline(
            y=0,
            linewidth=1
        )

        volatility_pnl_axis.set_title(
            "Net PnL by Volatility Regime"
        )

        volatility_pnl_axis.set_ylabel(
            "EUR"
        )

        volatility_pnl_axis.tick_params(
            axis="x",
            labelsize=7
        )

        volatility_pnl_axis.grid(
            True,
            axis="y",
            alpha=0.3
        )

        # ==================================================
        # PnL by market structure
        # ==================================================

        structure_axis = (
            figure.add_subplot(
                224
            )
        )

        if not structure_table.empty:

            structure_axis.bar(
                structure_table[
                    "Regime"
                ],
                structure_table[
                    "NetPnL"
                ]
            )

        structure_axis.axhline(
            y=0,
            linewidth=1
        )

        structure_axis.set_title(
            "Net PnL by Market Structure"
        )

        structure_axis.set_ylabel(
            "EUR"
        )

        structure_axis.tick_params(
            axis="x",
            labelsize=7
        )

        structure_axis.grid(
            True,
            axis="y",
            alpha=0.3
        )

        figure.suptitle(
            (
                "Market Regime Analysis "
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