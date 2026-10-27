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

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        tk.Label(
            container,
            text=(
                "Machine Learning Analysis"
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
                "Compare forecasting models or test "
                "how much market and weather features "
                "improve predictive performance."
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
            width=7
        ).pack(
            side="left",
            padx=(5, 20)
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
            width=18
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
        # Model comparison
        # --------------------------------------------------

        def run_model_comparison():

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

        # --------------------------------------------------
        # Feature ablation
        # --------------------------------------------------

        def run_feature_ablation():

            self._clear_frame(
                output_frame
            )

            folds = int(
                folds_variable.get()
            )

            model_name = (
                ablation_model_variable.get()
            )

            results = (
                self.machine_learning_forecaster
                .compare_feature_sets(
                    feature_data,
                    model_name=model_name,
                    n_splits=folds
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

        tk.Button(
            controls,
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
            controls,
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
            side="left"
        )

        # Default page
        run_model_comparison()

    # ==================================================
    # Model comparison display
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

        # --------------------------------------------------
        # Summary
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
                padx=11
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Validation info
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
                "Observations",
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
        # Comparison table
        # --------------------------------------------------

        self._create_model_table(
            container,
            results
        )

        # --------------------------------------------------
        # Feature importance
        # --------------------------------------------------

        self._show_feature_importance(
            container,
            results[
                "feature_importance"
            ]
        )

        # --------------------------------------------------
        # Chart
        # --------------------------------------------------

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
                    f"{price_area} Model Validation "
                    f"- {time_range}"
                )
            )
        )

        self._embed_chart(
            container,
            figure
        )

    # ==================================================
    # Ablation display
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

        tk.Label(
            container,
            text=(
                "The same model and the same "
                "walk-forward folds are used for every "
                "feature set. This isolates the value "
                "of additional data."
            ),
            font=(
                "Arial",
                9
            )
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        # --------------------------------------------------
        # Main summary
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
                "Market Factor Impact",
                (
                    f"{results['market_improvement']:+.1f}%"
                )
            ),
            (
                "Weather vs Price Only",
                (
                    f"{results['weather_improvement_vs_price']:+.1f}%"
                )
            ),
            (
                "Weather Incremental Impact",
                (
                    f"{results['weather_incremental_improvement']:+.1f}%"
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
                padx=10
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Prediction summary
        # --------------------------------------------------

        prediction_frame = tk.Frame(
            container
        )

        prediction_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        prediction_values = [
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
                "Folds",
                (
                    f"{results['n_splits']}"
                )
            ),
            (
                "Common Observations",
                (
                    f"{results['total_observations']:,}"
                )
            )
        ]

        for (
            name,
            value
        ) in prediction_values:

            tk.Label(
                prediction_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9
                ),
                padx=11
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Feature set comparison table
        # --------------------------------------------------

        self._create_ablation_table(
            container,
            results
        )

        # --------------------------------------------------
        # Feature importance of winning feature set
        # --------------------------------------------------

        self._show_feature_importance(
            container,
            results[
                "feature_importance"
            ]
        )

        # --------------------------------------------------
        # Chart
        # --------------------------------------------------

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
                    f"{price_area} Feature Ablation "
                    f"- {time_range}"
                )
            )
        )

        self._embed_chart(
            container,
            figure
        )

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
                column=column_number,
                padx=2,
                pady=1
            )

        table = results[
            "comparison_table"
        ]

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
                "Arial",
                9,
                "bold"
            ) if is_best else (
                "Arial",
                9
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
                    column=column_number,
                    padx=2,
                    pady=1
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
                column=column_number,
                padx=2,
                pady=1
            )

        table = results[
            "comparison_table"
        ]

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
                "Arial",
                9,
                "bold"
            ) if is_best else (
                "Arial",
                9
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
                    column=column_number,
                    padx=2,
                    pady=1
                )

    # ==================================================
    # Feature importance
    # ==================================================

    def _show_feature_importance(
        self,
        container,
        feature_importance
    ):

        if feature_importance.empty:

            return

        feature_frame = tk.Frame(
            container
        )

        feature_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        tk.Label(
            feature_frame,
            text="Top Features:",
            font=(
                "Arial",
                9,
                "bold"
            )
        ).pack(
            side="left",
            padx=(0, 8)
        )

        top_features = (
            feature_importance.head(
                5
            )
        )

        for _, row in (
            top_features.iterrows()
        ):

            tk.Label(
                feature_frame,
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
    # Prediction chart
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
            figsize=(10, 3.7),
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
    # Embed chart
    # ==================================================

    def _embed_chart(
        self,
        container,
        figure
    ):

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