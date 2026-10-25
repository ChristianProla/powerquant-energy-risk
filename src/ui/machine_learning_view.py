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
                "Price Forecast"
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
                "Train a Random Forest model "
                "to predict the next electricity "
                "price using market and time features."
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
            text="Model:"
        ).pack(
            side="left"
        )

        model_variable = (
            tk.StringVar(
                value="Random Forest"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=model_variable,
            values=[
                "Random Forest"
            ],
            state="readonly",
            width=16
        ).pack(
            side="left",
            padx=(5, 20)
        )

        tk.Label(
            controls,
            text="Trees:"
        ).pack(
            side="left"
        )

        trees_variable = (
            tk.StringVar(
                value="300"
            )
        )

        ttk.Combobox(
            controls,
            textvariable=trees_variable,
            values=[
                "100",
                "200",
                "300",
                "500"
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
        # Run model
        # --------------------------------------------------

        def run_model():

            self._clear_frame(
                output_frame
            )

            number_of_trees = int(
                trees_variable.get()
            )

            results = (
                self.machine_learning_forecaster
                .train_random_forest(
                    feature_data,
                    test_size=0.20,
                    n_estimators=number_of_trees
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
            text="Train Model",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=run_model
        ).pack(
            side="left"
        )

        run_model()

    # ==================================================
    # Display results
    # ==================================================

    def _show_results(
        self,
        container,
        results,
        price_area,
        time_range
    ):

        # --------------------------------------------------
        # Evaluation statistics
        # --------------------------------------------------

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        stats = [
            (
                "MAE",
                (
                    f"{results['mae']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "RMSE",
                (
                    f"{results['rmse']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "R²",
                (
                    f"{results['r2']:.3f}"
                )
            ),
            (
                "Baseline MAE",
                (
                    f"{results['baseline_mae']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "vs Baseline",
                (
                    f"{results['improvement_vs_baseline']:+.1f}%"
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
                padx=13
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Next price prediction
        # --------------------------------------------------

        prediction_frame = tk.Frame(
            container
        )

        prediction_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        prediction_stats = [
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
            ),
            (
                "Train / Test",
                (
                    f"{results['training_observations']} / "
                    f"{results['testing_observations']}"
                )
            )
        ]

        for name, value in prediction_stats:

            tk.Label(
                prediction_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    9,
                    "bold"
                ),
                padx=13
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Feature importance
        # --------------------------------------------------

        importance_frame = tk.Frame(
            container
        )

        importance_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        tk.Label(
            importance_frame,
            text="Top Features:",
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
                importance_frame,
                text=(
                    f"{feature_name}: "
                    f"{importance:.1f}%"
                ),
                font=(
                    "Arial",
                    9
                ),
                padx=8
            ).pack(
                side="left"
            )

        # --------------------------------------------------
        # Actual vs predicted chart
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
    # Prediction chart
    # ==================================================

    def _create_prediction_chart(
        self,
        results,
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

        test_times = results[
            "test_times"
        ]

        actual_prices = results[
            "actual_prices"
        ]

        predicted_prices = results[
            "predicted_prices"
        ]

        axis.plot(
            test_times,
            actual_prices,
            label="Actual Price",
            linewidth=2
        )

        axis.plot(
            test_times,
            predicted_prices,
            label="Random Forest Prediction",
            linestyle="--",
            linewidth=2
        )

        axis.set_title(
            (
                f"{price_area} Machine Learning Forecast "
                f"- {time_range}"
            )
        )

        axis.set_ylabel(
            "Price (EUR/MWh)"
        )

        axis.set_xlabel(
            "Test Period"
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