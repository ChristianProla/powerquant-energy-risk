import tkinter as tk

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)


class MarketFactorsView:

    def __init__(
        self,
        feature_engineer
    ):

        self.feature_engineer = (
            feature_engineer
        )

    def show(
        self,
        container,
        feature_data,
        price_area,
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
                "Market Factors & "
                "Feature Engineering"
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
                "Explore how wind generation "
                "and electricity demand relate "
                "to electricity prices."
            ),
            font=(
                "Arial",
                10
            )
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        if feature_data.empty:

            tk.Label(
                container,
                text=(
                    "No matching market-factor "
                    "data was available for "
                    "this period."
                ),
                font=(
                    "Arial",
                    11
                )
            ).pack(
                pady=30
            )

            return

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        statistics = (
            self.feature_engineer
            .calculate_factor_statistics(
                feature_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        stats = [
            (
                "Wind / Price Correlation",
                (
                    f"{statistics['wind_price_correlation']:.3f}"
                )
            ),
            (
                "Demand / Price Correlation",
                (
                    f"{statistics['demand_price_correlation']:.3f}"
                )
            ),
            (
                "Average Wind",
                (
                    f"{statistics['average_wind']:.0f} MW"
                )
            ),
            (
                "Average Demand",
                (
                    f"{statistics['average_demand']:.0f} MW"
                )
            ),
            (
                "Wind / Demand",
                (
                    f"{statistics['average_wind_ratio'] * 100:.1f}%"
                )
            ),
            (
                "Matched Hours",
                (
                    f"{statistics['observations']:,}"
                )
            )
        ]

        for name, value in stats:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9,
                    "bold"
                ),
                padx=10
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Chart
        # --------------------------------------------------

        figure = (
            self._create_chart(
                feature_data,
                price_area,
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

    # --------------------------------------------------
    # Chart
    # --------------------------------------------------

    def _create_chart(
        self,
        feature_data,
        price_area,
        time_range
    ):

        figure = Figure(
            figsize=(10, 5),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        data = (
            feature_data.copy()
        )

        if data.empty:

            axis.text(
                0.5,
                0.5,
                "Not enough data to display market factors.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes
            )

            return figure

        # --------------------------------------------------
        # Price
        # --------------------------------------------------

        price_data = data.dropna(
            subset=[
                "DayAheadPriceEUR"
            ]
        )

        if not price_data.empty:

            axis.plot(
                price_data[
                    "TimeDK"
                ],
                self._normalise(
                    price_data[
                        "DayAheadPriceEUR"
                    ]
                ),
                label="Electricity Price"
            )

        # --------------------------------------------------
        # Wind
        # --------------------------------------------------

        wind_data = data.dropna(
            subset=[
                "WindGenerationMW"
            ]
        )

        if not wind_data.empty:

            axis.plot(
                wind_data[
                    "TimeDK"
                ],
                self._normalise(
                    wind_data[
                        "WindGenerationMW"
                    ]
                ),
                label="Wind Generation"
            )

        # --------------------------------------------------
        # Demand
        # --------------------------------------------------

        demand_data = data.dropna(
            subset=[
                "ElectricityDemandMW"
            ]
        )

        if not demand_data.empty:

            axis.plot(
                demand_data[
                    "TimeDK"
                ],
                self._normalise(
                    demand_data[
                        "ElectricityDemandMW"
                    ]
                ),
                label="Electricity Demand"
            )

        axis.axhline(
            y=100,
            linewidth=1,
            alpha=0.5
        )

        axis.set_title(
            (
                f"{price_area} Market Factors "
                f"- {time_range}"
            )
        )

        axis.set_ylabel(
            "Normalised Index"
        )

        axis.set_xlabel(
            "Time"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend()

        figure.autofmt_xdate(
            rotation=45
        )

        figure.tight_layout()

        return figure

    # --------------------------------------------------
    # Normalise series
    # --------------------------------------------------

    def _normalise(
        self,
        series
    ):

        values = series.astype(
            float
        )

        mean = values.mean()

        if (
            mean == 0
            or
            np.isnan(mean)
        ):

            return values * 0

        return (
            values
            / abs(mean)
        ) * 100

    # --------------------------------------------------
    # Clear frame
    # --------------------------------------------------

    def _clear_frame(
        self,
        frame
    ):

        for widget in (
            frame.winfo_children()
        ):

            widget.destroy()