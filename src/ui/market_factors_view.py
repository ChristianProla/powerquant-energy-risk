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

    # ==================================================
    # Main view
    # ==================================================

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
                "Market Factors, Weather "
                "& Feature Engineering"
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
                "Explore electricity prices together "
                "with wind generation, demand and "
                "weather conditions."
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
                    "data was available."
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

        stats_row_1 = tk.Frame(
            container
        )

        stats_row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        row_1_values = [
            (
                "Generation Wind / Price",
                (
                    f"{statistics['wind_price_correlation']:.3f}"
                )
            ),
            (
                "Demand / Price",
                (
                    f"{statistics['demand_price_correlation']:.3f}"
                )
            ),
            (
                "Weather Wind / Price",
                (
                    f"{statistics['weather_wind_price_correlation']:.3f}"
                )
            ),
            (
                "Temperature / Price",
                (
                    f"{statistics['temperature_price_correlation']:.3f}"
                )
            )
        ]

        for name, value in (
            row_1_values
        ):

            tk.Label(
                stats_row_1,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9,
                    "bold"
                ),
                padx=12
            ).pack(
                side="left"
            )

        stats_row_2 = tk.Frame(
            container
        )

        stats_row_2.pack(
            fill="x",
            pady=(0, 8)
        )

        row_2_values = [
            (
                "Average Generation Wind",
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
                "Average Weather Wind",
                (
                    f"{statistics['average_weather_wind']:.1f} m/s"
                )
            ),
            (
                "Average Temperature",
                (
                    f"{statistics['average_temperature']:.1f} °C"
                )
            ),
            (
                "Weather Hours",
                (
                    f"{statistics['weather_observations']:,}"
                )
            )
        ]

        for name, value in (
            row_2_values
        ):

            tk.Label(
                stats_row_2,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9
                ),
                padx=10
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Explanation
        # --------------------------------------------------

        tk.Label(
            container,
            text=(
                "Weather is represented by a geographic proxy: "
                "Aarhus for DK1 and Copenhagen for DK2."
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

    # ==================================================
    # Chart
    # ==================================================

    def _create_chart(
        self,
        feature_data,
        price_area,
        time_range
    ):

        figure = Figure(
            figsize=(10, 4.5),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        data = (
            feature_data.copy()
        )

        series = [
            (
                "DayAheadPriceEUR",
                "Electricity Price"
            ),
            (
                "WindGenerationMW",
                "Wind Generation"
            ),
            (
                "ElectricityDemandMW",
                "Electricity Demand"
            ),
            (
                "WeatherWindSpeedMS",
                "Weather Wind Speed"
            ),
            (
                "TemperatureC",
                "Temperature"
            )
        ]

        plotted = False

        for (
            column,
            label
        ) in series:

            if column not in (
                data.columns
            ):

                continue

            clean_data = (
                data[
                    [
                        "TimeDK",
                        column
                    ]
                ]
                .dropna()
            )

            if clean_data.empty:

                continue

            standardized = (
                self._standardize(
                    clean_data[
                        column
                    ]
                )
            )

            axis.plot(
                clean_data[
                    "TimeDK"
                ],
                standardized,
                label=label
            )

            plotted = True

        if not plotted:

            axis.text(
                0.5,
                0.5,
                "Not enough data to display market factors.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes
            )

            return figure

        axis.axhline(
            y=0,
            linewidth=1,
            alpha=0.5
        )

        axis.set_title(
            (
                f"{price_area} Market & Weather Factors "
                f"- {time_range}"
            )
        )

        axis.set_ylabel(
            "Standardized Value (z-score)"
        )

        axis.set_xlabel(
            "Time"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend(
            fontsize=8
        )

        figure.autofmt_xdate(
            rotation=45
        )

        figure.tight_layout()

        return figure

    # ==================================================
    # Standardization
    # ==================================================

    def _standardize(
        self,
        series
    ):

        values = (
            series.astype(
                float
            )
        )

        mean = (
            values.mean()
        )

        standard_deviation = (
            values.std()
        )

        if (
            np.isnan(
                standard_deviation
            )
            or
            standard_deviation == 0
        ):

            return (
                values * 0
            )

        return (
            (
                values
                - mean
            )
            / standard_deviation
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