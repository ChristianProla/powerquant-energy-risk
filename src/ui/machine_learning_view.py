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
    # Main page
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

        tk.Label(
            container,
            text="Machine Learning Analysis",
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
                "Compare models, measure feature impact "
                "or create a forward-looking electricity "
                "price forecast using future weather."
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
            textvariable=(
                folds_variable
            ),
            values=[
                "3",
                "4",
                "5",
                "6"
            ],
            state="readonly",
            width=6
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Ablation Model:"
        ).pack(
            side="left"
        )

        ablation_model_variable = (
            tk.StringVar(
                value="Random Forest"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                ablation_model_variable
            ),
            values=[
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting"
            ],
            state="readonly",
            width=17
        ).pack(
            side="left",
            padx=(5, 15)
        )

        tk.Label(
            controls,
            text="Forward Horizon:"
        ).pack(
            side="left"
        )

        horizon_variable = (
            tk.StringVar(
                value="24"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=(
                horizon_variable
            ),
            values=[
                "6",
                "12",
                "24"
            ],
            state="readonly",
            width=5
        ).pack(
            side="left",
            padx=(5, 15)
        )

        # ==================================================
        # Button row
        # ==================================================

        button_frame = tk.Frame(
            container
        )

        button_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # Model comparison
        # ==================================================

        def run_model_comparison():

            self._clear_frame(
                output_frame
            )

            results = (
                self.machine_learning_forecaster
                .compare_models(
                    feature_data,
                    n_splits=int(
                        folds_variable.get()
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

            self._show_model_comparison(
                output_frame,
                results,
                price_area,
                time_range
            )

        # ==================================================
        # Feature impact
        # ==================================================

        def run_feature_ablation():

            self._clear_frame(
                output_frame
            )

            results = (
                self.machine_learning_forecaster
                .compare_feature_sets(
                    feature_data,
                    model_name=(
                        ablation_model_variable.get()
                    ),
                    n_splits=int(
                        folds_variable.get()
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

            self._show_ablation_results(
                output_frame,
                results,
                price_area,
                time_range
            )

        # ==================================================
        # Forward forecast
        # ==================================================

        def run_forward_forecast():

            self._clear_frame(
                output_frame
            )

            tk.Label(
                output_frame,
                text=(
                    "Loading future weather and "
                    "training forecast models..."
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
                self.machine_learning_forecaster
                .create_forward_forecast(
                    feature_data,
                    price_area=price_area,
                    horizon=int(
                        horizon_variable.get()
                    ),
                    n_splits=int(
                        folds_variable.get()
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

            self._show_forward_forecast(
                output_frame,
                results,
                price_area
            )

        # ==================================================
        # Buttons
        # ==================================================

        tk.Button(
            button_frame,
            text="Compare Models",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=(
                run_model_comparison
            )
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Feature Impact Test",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=(
                run_feature_ablation
            )
        ).pack(
            side="left",
            padx=(0, 8)
        )

        tk.Button(
            button_frame,
            text="Forward Forecast",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=(
                run_forward_forecast
            )
        ).pack(
            side="left"
        )

        run_model_comparison()

    # ==================================================
    # Model comparison
    # ==================================================

    def _show_model_comparison(
        self,
        container,
        results,
        price_area,
        time_range
    ):

        tk.Label(
            container,
            text="Model Comparison",
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        summary_frame = tk.Frame(
            container
        )

        summary_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        values = [
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
                "Change",
                (
                    f"{results['predicted_change']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Change %",
                (
                    f"{results['predicted_change_percent']:+.2f}%"
                )
            )
        ]

        self._show_stats(
            summary_frame,
            values,
            bold=True
        )

        self._create_model_table(
            container,
            results
        )

        self._show_feature_importance(
            container,
            results[
                "feature_importance"
            ]
        )

        figure = (
            self._create_prediction_chart(
                results[
                    "prediction_times"
                ],
                results[
                    "actual_prices"
                ],
                results[
                    "best_predictions"
                ],
                (
                    f"{results['best_model']} "
                    f"Prediction"
                ),
                (
                    f"{price_area} Walk-Forward "
                    f"Validation - {time_range}"
                )
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Feature impact
    # ==================================================

    def _show_ablation_results(
        self,
        container,
        results,
        price_area,
        time_range
    ):

        tk.Label(
            container,
            text="Feature Impact Analysis",
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        summary_frame = tk.Frame(
            container
        )

        summary_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        values = [
            (
                "Model",
                results[
                    "model_name"
                ]
            ),
            (
                "Best Feature Set",
                results[
                    "best_feature_set"
                ]
            ),
            (
                "Market Impact",
                (
                    f"{results['market_improvement']:+.1f}%"
                )
            ),
            (
                "Weather vs Price",
                (
                    f"{results['weather_improvement_vs_price']:+.1f}%"
                )
            ),
            (
                "Weather Incremental",
                (
                    f"{results['weather_incremental_improvement']:+.1f}%"
                )
            )
        ]

        self._show_stats(
            summary_frame,
            values,
            bold=True
        )

        self._create_ablation_table(
            container,
            results
        )

        self._show_feature_importance(
            container,
            results[
                "feature_importance"
            ]
        )

        figure = (
            self._create_prediction_chart(
                results[
                    "prediction_times"
                ],
                results[
                    "actual_prices"
                ],
                results[
                    "best_predictions"
                ],
                (
                    f"{results['best_feature_set']} "
                    f"Prediction"
                ),
                (
                    f"{price_area} Feature Impact "
                    f"- {time_range}"
                )
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # NEW:
    # Forward forecast display
    # ==================================================

    def _show_forward_forecast(
        self,
        container,
        results,
        price_area
    ):

        tk.Label(
            container,
            text="Forward-Looking Price Forecast",
            font=(
                "Arial",
                12,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        tk.Label(
            container,
            text=(
                "The model uses historical electricity prices "
                "and future weather forecasts to recursively "
                "predict upcoming hourly electricity prices."
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
        # Main forecast stats
        # ==================================================

        stats_row_1 = tk.Frame(
            container
        )

        stats_row_1.pack(
            fill="x",
            pady=(0, 5)
        )

        row_1 = [
            (
                "Model",
                results[
                    "best_model"
                ]
            ),
            (
                "Horizon",
                (
                    f"{results['forecast_hours']} hours"
                )
            ),
            (
                "Current Price",
                (
                    f"{results['current_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "End Forecast",
                (
                    f"{results['ending_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Total Change",
                (
                    f"{results['price_change']:+.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Change %",
                (
                    f"{results['price_change_percent']:+.2f}%"
                )
            )
        ]

        self._show_stats(
            stats_row_1,
            row_1,
            bold=True
        )

        stats_row_2 = tk.Frame(
            container
        )

        stats_row_2.pack(
            fill="x",
            pady=(0, 7)
        )

        row_2 = [
            (
                "Average Forecast",
                (
                    f"{results['average_forecast_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Forecast High",
                (
                    f"{results['maximum_forecast_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Forecast Low",
                (
                    f"{results['minimum_forecast_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Validation MAE",
                (
                    f"{results['validation_mae']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Validation RMSE",
                (
                    f"{results['validation_rmse']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Validation R²",
                (
                    f"{results['validation_r2']:.3f}"
                )
            )
        ]

        self._show_stats(
            stats_row_2,
            row_2,
            bold=False
        )

        tk.Label(
            container,
            text=(
                f"Weather forecast proxy: "
                f"{results['weather_location']} "
                f"for {price_area}"
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        # ==================================================
        # Feature importance
        # ==================================================

        self._show_feature_importance(
            container,
            results[
                "feature_importance"
            ]
        )

        # ==================================================
        # Forward chart
        # ==================================================

        figure = (
            self._create_forward_chart(
                results,
                price_area
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Forward chart
    # ==================================================

    def _create_forward_chart(
        self,
        results,
        price_area
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

        history_times = (
            results[
                "history_times"
            ]
        )

        history_prices = (
            results[
                "history_prices"
            ]
        )

        forecast_times = (
            results[
                "forecast_times"
            ]
        )

        forecast_prices = (
            results[
                "forecast_prices"
            ]
        )

        axis.plot(
            history_times,
            history_prices,
            label="Historical Price",
            linewidth=2
        )

        # Connect historical price to forecast.

        if (
            len(
                history_times
            )
            > 0
            and
            len(
                forecast_times
            )
            > 0
        ):

            connection_times = [
                history_times[
                    -1
                ],
                forecast_times[
                    0
                ]
            ]

            connection_prices = [
                history_prices[
                    -1
                ],
                forecast_prices[
                    0
                ]
            ]

            axis.plot(
                connection_times,
                connection_prices,
                linestyle="--",
                linewidth=2
            )

        axis.plot(
            forecast_times,
            forecast_prices,
            label=(
                f"{results['best_model']} "
                f"Forward Forecast"
            ),
            linestyle="--",
            linewidth=2
        )

        if (
            len(
                forecast_times
            )
            > 0
        ):

            axis.axvline(
                x=forecast_times[
                    0
                ],
                linestyle=":",
                linewidth=1.5,
                label="Forecast Period"
            )

        axis.set_title(
            (
                f"{price_area} "
                f"{results['forecast_hours']}-Hour "
                f"Electricity Price Forecast"
            )
        )

        axis.set_xlabel(
            "Time"
        )

        axis.set_ylabel(
            "Price (EUR/MWh)"
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
    # Model table
    # ==================================================

    def _create_model_table(
        self,
        container,
        results
    ):

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
            column_number,
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
                width=19,
                anchor="w"
            ).grid(
                row=0,
                column=column_number
            )

        table = (
            results[
                "comparison_table"
            ]
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
                    "Model"
                ],
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

            is_best = (
                row[
                    "Model"
                ]
                == results[
                    "best_model"
                ]
            )

            font = (
                (
                    "Arial",
                    9,
                    "bold"
                )
                if is_best
                else (
                    "Arial",
                    9
                )
            )

            for (
                column_number,
                value
            ) in enumerate(
                values
            ):

                tk.Label(
                    table_frame,
                    text=value,
                    font=font,
                    width=19,
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column_number
                )

    # ==================================================
    # Ablation table
    # ==================================================

    def _create_ablation_table(
        self,
        container,
        results
    ):

        table_frame = tk.Frame(
            container
        )

        table_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        columns = [
            "Feature Set",
            "Features",
            "MAE",
            "RMSE",
            "R²",
            "vs Price Only"
        ]

        widths = [
            27,
            10,
            12,
            12,
            10,
            15
        ]

        for (
            column_number,
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
                width=widths[
                    column_number
                ],
                anchor="w"
            ).grid(
                row=0,
                column=column_number
            )

        table = (
            results[
                "comparison_table"
            ]
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
                    "FeatureSet"
                ],
                (
                    f"{int(row['FeatureCount'])}"
                ),
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
                    f"{row['ImprovementVsPriceOnly']:+.1f}%"
                )
            ]

            is_best = (
                row[
                    "FeatureSet"
                ]
                == results[
                    "best_feature_set"
                ]
            )

            font = (
                (
                    "Arial",
                    9,
                    "bold"
                )
                if is_best
                else (
                    "Arial",
                    9
                )
            )

            for (
                column_number,
                value
            ) in enumerate(
                values
            ):

                tk.Label(
                    table_frame,
                    text=value,
                    font=font,
                    width=widths[
                        column_number
                    ],
                    anchor="w"
                ).grid(
                    row=row_number,
                    column=column_number
                )

    # ==================================================
    # Feature importance
    # ==================================================

    def _show_feature_importance(
        self,
        container,
        feature_importance
    ):

        if (
            feature_importance.empty
        ):

            return

        frame = tk.Frame(
            container
        )

        frame.pack(
            fill="x",
            pady=(0, 7)
        )

        tk.Label(
            frame,
            text="Top Features:",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            side="left"
        )

        for _, row in (
            feature_importance.head(
                5
            ).iterrows()
        ):

            tk.Label(
                frame,
                text=(
                    f"{row['Feature']}: "
                    f"{row['Importance'] * 100:.1f}%"
                ),
                font=(
                    "Arial",
                    9
                ),
                padx=6
            ).pack(
                side="left"
            )

    # ==================================================
    # Historical validation chart
    # ==================================================

    def _create_prediction_chart(
        self,
        times,
        actual,
        predicted,
        prediction_label,
        title
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

        axis.plot(
            times,
            actual,
            label="Actual Price",
            linewidth=2
        )

        axis.plot(
            times,
            predicted,
            label=prediction_label,
            linestyle="--",
            linewidth=2
        )

        axis.set_title(
            title
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
    # Embed chart
    # ==================================================

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

    # ==================================================
    # Error
    # ==================================================

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

    # ==================================================
    # Clear
    # ==================================================

    def _clear_frame(
        self,
        frame
    ):

        for widget in (
            frame.winfo_children()
        ):

            widget.destroy()