import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)


class MachineLearningView:

    def __init__(
        self,
        machine_learning_forecaster
    ):

        self.machine_learning_forecaster = (
            machine_learning_forecaster
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
                "Machine Learning "
                "Model Comparison"
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
                "Compare forecasting models using "
                "walk-forward time-series validation."
            ),
            font=(
                "Arial",
                10
            )
        ).pack(
            anchor="w",
            pady=(0, 10)
        )

        # --------------------------------------------------
        # Controls
        # --------------------------------------------------

        controls = tk.Frame(
            container
        )

        controls.pack(
            fill="x",
            pady=(0, 10)
        )

        tk.Label(
            controls,
            text="Validation Folds:"
        ).pack(
            side="left"
        )

        folds_variable = (
            tk.StringVar(
                value="5"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=folds_variable,
            values=[
                "3",
                "4",
                "5",
                "6"
            ],
            state="readonly",
            width=8
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

        # --------------------------------------------------
        # Run comparison
        # --------------------------------------------------

        def run_comparison():

            self._clear_frame(
                output_frame
            )

            folds = int(
                folds_variable.get()
            )

            results = (
                self.machine_learning_forecaster
                .compare_models(
                    feature_data,
                    n_splits=folds
                )
            )

            if not results[
                "success"
            ]:

                tk.Label(
                    output_frame,
                    text=results[
                        "message"
                    ],
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

            self._show_results(
                output_frame,
                results,
                price_area,
                time_range
            )

        tk.Button(
            controls,
            text="Compare Models",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_comparison
        ).pack(
            side="left"
        )

        run_comparison()

    # ==================================================
    # Show results
    # ==================================================

    def _show_results(
        self,
        container,
        results,
        price_area,
        time_range
    ):

        # --------------------------------------------------
        # Best model summary
        # --------------------------------------------------

        summary_frame = tk.Frame(
            container
        )

        summary_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        summary_values = [
            (
                "Best Model",
                results[
                    "best_model"
                ]
            ),
            (
                "Current Price",
                (
                    f"{results['current_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Predicted Next Price",
                (
                    f"{results['next_price_prediction']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Predicted Change",
                (
                    f"{results['predicted_change']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Predicted Change %",
                (
                    f"{results['predicted_change_percent']:+.2f}%"
                )
            )
        ]

        for (
            name,
            value
        ) in summary_values:

            tk.Label(
                summary_frame,
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

        # --------------------------------------------------
        # Validation information
        # --------------------------------------------------

        validation_frame = tk.Frame(
            container
        )

        validation_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        validation_values = [
            (
                "Validation",
                "Walk-Forward"
            ),
            (
                "Folds",
                (
                    f"{results['n_splits']}"
                )
            ),
            (
                "Total Observations",
                (
                    f"{results['total_observations']:,}"
                )
            ),
            (
                "Validation Predictions",
                (
                    f"{results['validation_observations']:,}"
                )
            )
        ]

        for (
            name,
            value
        ) in validation_values:

            tk.Label(
                validation_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9
                ),
                padx=12
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Model comparison table
        # --------------------------------------------------

        table_frame = tk.Frame(
            container
        )

        table_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        columns = [
            "Model",
            "MAE",
            "RMSE",
            "R²",
            "vs Baseline"
        ]

        for (
            column_index,
            column_name
        ) in enumerate(
            columns
        ):

            tk.Label(
                table_frame,
                text=column_name,
                font=(
                    "Arial",
                    9,
                    "bold"
                ),
                width=20,
                anchor="w"
            ).grid(
                row=0,
                column=column_index,
                padx=3,
                pady=2
            )

        comparison_table = (
            results[
                "comparison_table"
            ]
        )

        for (
            row_number,
            (_, row)
        ) in enumerate(
            comparison_table.iterrows(),
            start=1
        ):

            values = [
                row["Model"],
                (
                    f"{row['MAE']:.2f}"
                ),
                (
                    f"{row['RMSE']:.2f}"
                ),
                (
                    f"{row['R2']:.3f}"
                ),
                (
                    f"{row['ImprovementVsBaseline']:+.1f}%"
                )
            ]

            for (
                column_index,
                value
            ) in enumerate(
                values
            ):

                font = (
                    "Arial",
                    9
                )

                if (
                    row["Model"]
                    == results[
                        "best_model"
                    ]
                ):

                    font = (
                        "Arial",
                        9,
                        "bold"
                    )

                tk.Label(
                    table_frame,
                    text=value,
                    font=font,
                    width=20,
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column_index,
                    padx=3,
                    pady=1
                )

        # --------------------------------------------------
        # Top features
        # --------------------------------------------------

        if not results[
            "feature_importance"
        ].empty:

            feature_frame = tk.Frame(
                container
            )

            feature_frame.pack(
                fill="x",
                pady=(0, 8)
            )

            tk.Label(
                feature_frame,
                text=(
                    "Top Features:"
                ),
                font=(
                    "Arial",
                    9,
                    "bold"
                )
            ).pack(
                side="left",
                padx=(0, 10)
            )

            top_features = (
                results[
                    "feature_importance"
                ].head(5)
            )

            for _, row in (
                top_features.iterrows()
            ):

                feature_name = (
                    row["Feature"]
                )

                importance = (
                    row["Importance"]
                    * 100
                )

                tk.Label(
                    feature_frame,
                    text=(
                        f"{feature_name}: "
                        f"{importance:.1f}%"
                    ),
                    font=(
                        "Arial",
                        9
                    ),
                    padx=7
                ).pack(
                    side="left"
                )

        # --------------------------------------------------
        # Chart
        # --------------------------------------------------

        figure = (
            self._create_prediction_chart(
                results,
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
    # Actual vs predicted
    # ==================================================

    def _create_prediction_chart(
        self,
        results,
        price_area,
        time_range
    ):

        figure = Figure(
            figsize=(10, 4),
            dpi=100
        )

        axis = figure.add_subplot(
            111
        )

        times = results[
            "prediction_times"
        ]

        actual = results[
            "actual_prices"
        ]

        predicted = results[
            "best_predictions"
        ]

        axis.plot(
            times,
            actual,
            label="Actual Price",
            linewidth=2
        )

        axis.plot(
            times,
            predicted,
            label=(
                f"{results['best_model']} Prediction"
            ),
            linestyle="--",
            linewidth=2
        )

        axis.set_title(
            (
                f"{price_area} "
                f"Walk-Forward Validation "
                f"- {time_range}"
            )
        )

        axis.set_xlabel(
            "Validation Period"
        )

        axis.set_ylabel(
            "Price (EUR/MWh)"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend()

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