import tkinter as tk
from tkinter import ttk, messagebox

import requests

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.api.electricity_api import ElectricityAPI
from src.api.market_factors_api import MarketFactorsAPI

from src.analytics.resampling import MarketDataResampler
from src.analytics.returns import ReturnCalculator
from src.analytics.statistics import StatisticsAnalyzer
from src.analytics.volatility import VolatilityAnalyzer
from src.analytics.correlation import CorrelationAnalyzer
from src.analytics.spread import SpreadAnalyzer

from src.portfolio.portfolio import PortfolioAnalyzer
from src.portfolio.optimization import PortfolioOptimizer

from src.risk.value_at_risk import ValueAtRiskAnalyzer
from src.risk.expected_shortfall import ExpectedShortfallAnalyzer
from src.risk.monte_carlo import MonteCarloSimulator
from src.risk.stress_testing import StressTester

from src.forecasting.forecasting import PriceForecaster
from src.forecasting.feature_engineering import FeatureEngineer
from src.forecasting.machine_learning import (
    MachineLearningForecaster
)

from src.ui.market_factors_view import MarketFactorsView
from src.ui.machine_learning_view import MachineLearningView
from src.ui.spread_view import SpreadView

from src.visualization.charts import ChartBuilder


class PowerQuantApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "PowerQuant"
        )

        self.geometry(
            "1200x750"
        )

        self.minsize(
            1000,
            650
        )

        # ==================================================
        # Core services
        # ==================================================

        self.api = (
            ElectricityAPI()
        )

        self.market_factors_api = (
            MarketFactorsAPI()
        )

        self.chart_builder = (
            ChartBuilder()
        )

        self.resampler = (
            MarketDataResampler()
        )

        self.return_calculator = (
            ReturnCalculator()
        )

        self.statistics_analyzer = (
            StatisticsAnalyzer()
        )

        self.volatility_analyzer = (
            VolatilityAnalyzer()
        )

        self.correlation_analyzer = (
            CorrelationAnalyzer()
        )

        self.spread_analyzer = (
            SpreadAnalyzer()
        )

        self.portfolio_analyzer = (
            PortfolioAnalyzer()
        )

        self.portfolio_optimizer = (
            PortfolioOptimizer()
        )

        self.var_analyzer = (
            ValueAtRiskAnalyzer()
        )

        self.es_analyzer = (
            ExpectedShortfallAnalyzer()
        )

        self.monte_carlo_simulator = (
            MonteCarloSimulator()
        )

        self.stress_tester = (
            StressTester()
        )

        self.price_forecaster = (
            PriceForecaster()
        )

        self.feature_engineer = (
            FeatureEngineer()
        )

        self.machine_learning_forecaster = (
            MachineLearningForecaster()
        )

        # ==================================================
        # UI modules
        # ==================================================

        self.market_factors_view = (
            MarketFactorsView(
                self.feature_engineer
            )
        )

        self.machine_learning_view = (
            MachineLearningView(
                self.machine_learning_forecaster
            )
        )

        self.spread_view = (
            SpreadView(
                self.spread_analyzer
            )
        )

        # ==================================================
        # Time ranges
        # ==================================================

        self.selected_range = (
            "1D"
        )

        self.time_ranges = {
            "1H": (
                "now-PT1H",
                "1 Hour"
            ),
            "1D": (
                "now-P1D",
                "1 Day"
            ),
            "1W": (
                "now-P7D",
                "1 Week"
            ),
            "1M": (
                "now-P1M",
                "1 Month"
            ),
            "1Y": (
                "now-P1Y",
                "1 Year"
            )
        }

        self.show_home_page()

    # ==================================================
    # Helpers
    # ==================================================

    def clear_window(
        self
    ):

        for widget in (
            self.winfo_children()
        ):

            widget.destroy()

    def clear_frame(
        self,
        frame
    ):

        for widget in (
            frame.winfo_children()
        ):

            widget.destroy()

    def embed_chart(
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
    # Home
    # ==================================================

    def show_home_page(
        self
    ):

        self.clear_window()

        header = tk.Frame(
            self
        )

        header.pack(
            fill="x",
            padx=40,
            pady=(30, 10)
        )

        tk.Label(
            header,
            text="PowerQuant",
            font=(
                "Arial",
                30,
                "bold"
            )
        ).pack(
            anchor="w"
        )

        tk.Label(
            header,
            text=(
                "Electricity Market Analytics "
                "& Risk Platform"
            ),
            font=(
                "Arial",
                14
            )
        ).pack(
            anchor="w"
        )

        content = tk.Frame(
            self
        )

        content.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=25
        )

        tk.Label(
            content,
            text="Market",
            font=(
                "Arial",
                15,
                "bold"
            )
        ).pack(
            anchor="w"
        )

        self.market_selector = (
            ttk.Combobox(
                content,
                values=[
                    "DK1",
                    "DK2"
                ],
                state="readonly",
                width=15
            )
        )

        self.market_selector.set(
            "DK1"
        )

        self.market_selector.pack(
            anchor="w",
            pady=(8, 25)
        )

        tk.Label(
            content,
            text="Time Range",
            font=(
                "Arial",
                15,
                "bold"
            )
        ).pack(
            anchor="w"
        )

        range_frame = tk.Frame(
            content
        )

        range_frame.pack(
            anchor="w",
            pady=(10, 30)
        )

        for time_range in [
            "1H",
            "1D",
            "1W",
            "1M",
            "1Y"
        ]:

            tk.Button(
                range_frame,
                text=time_range,
                width=7,
                command=lambda value=time_range:
                self.select_range(
                    value
                )
            ).pack(
                side="left",
                padx=(0, 8)
            )

        self.range_label = (
            tk.Label(
                content,
                text="Selected: 1 Day",
                font=(
                    "Arial",
                    11
                )
            )
        )

        self.range_label.pack(
            anchor="w",
            pady=(0, 20)
        )

        tk.Button(
            content,
            text="Open Market Analysis",
            font=(
                "Arial",
                13,
                "bold"
            ),
            padx=15,
            pady=8,
            command=self.load_market_data
        ).pack(
            anchor="w"
        )

        self.status_label = (
            tk.Label(
                content,
                text="Ready",
                font=(
                    "Arial",
                    11
                )
            )
        )

        self.status_label.pack(
            anchor="w",
            pady=(20, 0)
        )

    # ==================================================
    # Time range
    # ==================================================

    def select_range(
        self,
        time_range
    ):

        self.selected_range = (
            time_range
        )

        readable_name = (
            self.time_ranges[
                time_range
            ][1]
        )

        self.range_label.config(
            text=(
                f"Selected: "
                f"{readable_name}"
            )
        )

    # ==================================================
    # Load data
    # ==================================================

    def load_market_data(
        self
    ):

        price_area = (
            self.market_selector.get()
        )

        start_date = (
            self.time_ranges[
                self.selected_range
            ][0]
        )

        readable_range = (
            self.time_ranges[
                self.selected_range
            ][1]
        )

        self.status_label.config(
            text="Loading market data..."
        )

        self.update()

        try:

            # --------------------------------------------------
            # Selected market
            # --------------------------------------------------

            prices = (
                self.api.get_prices(
                    price_area=price_area,
                    start_date=start_date,
                    end_date="now"
                )
            )

            if prices.empty:

                messagebox.showwarning(
                    "No Data",
                    "No market data was returned."
                )

                self.status_label.config(
                    text="No data available"
                )

                return

            chart_data, resolution = (
                self.resampler
                .resample_for_range(
                    prices,
                    self.selected_range
                )
            )

            returns_data = (
                self.return_calculator
                .calculate_returns(
                    chart_data
                )
            )

            volatility_data = (
                self.volatility_analyzer
                .calculate_volatility(
                    returns_data
                )
            )

            # --------------------------------------------------
            # DK1 + DK2
            # --------------------------------------------------

            if (
                price_area
                == "DK1"
            ):

                dk1_prices = (
                    prices
                )

                dk2_prices = (
                    self.api.get_prices(
                        price_area="DK2",
                        start_date=start_date,
                        end_date="now"
                    )
                )

            else:

                dk2_prices = (
                    prices
                )

                dk1_prices = (
                    self.api.get_prices(
                        price_area="DK1",
                        start_date=start_date,
                        end_date="now"
                    )
                )

            dk1_chart_data, _ = (
                self.resampler
                .resample_for_range(
                    dk1_prices,
                    self.selected_range
                )
            )

            dk2_chart_data, _ = (
                self.resampler
                .resample_for_range(
                    dk2_prices,
                    self.selected_range
                )
            )

            dk1_returns = (
                self.return_calculator
                .calculate_returns(
                    dk1_chart_data
                )
            )

            dk2_returns = (
                self.return_calculator
                .calculate_returns(
                    dk2_chart_data
                )
            )

            correlation_data = (
                self.correlation_analyzer
                .prepare_correlation_data(
                    dk1_returns,
                    dk2_returns
                )
            )

            # --------------------------------------------------
            # Market factors + weather
            # --------------------------------------------------

            market_factors = (
                self.market_factors_api
                .get_market_factors(
                    price_area=price_area,
                    start_date=start_date,
                    end_date="now"
                )
            )

            feature_data = (
                self.feature_engineer
                .create_market_features(
                    prices,
                    market_factors
                )
            )

            # --------------------------------------------------
            # Open analysis
            # --------------------------------------------------

            self.show_market_page(
                prices,
                chart_data,
                returns_data,
                volatility_data,
                correlation_data,
                feature_data,
                dk1_prices,
                dk2_prices,
                price_area,
                readable_range,
                resolution
            )

        except (
            requests.exceptions
            .RequestException
        ) as error:

            self.status_label.config(
                text="Connection error"
            )

            messagebox.showerror(
                "Connection Error",
                str(
                    error
                )
            )

        except Exception as error:

            self.status_label.config(
                text="PowerQuant error"
            )

            messagebox.showerror(
                "PowerQuant Error",
                str(
                    error
                )
            )

    # ==================================================
    # Market page
    # ==================================================

    def show_market_page(
        self,
        prices,
        chart_data,
        returns_data,
        volatility_data,
        correlation_data,
        feature_data,
        dk1_prices,
        dk2_prices,
        price_area,
        time_range,
        resolution
    ):

        self.clear_window()

        navigation = tk.Frame(
            self
        )

        navigation.pack(
            fill="x",
            padx=30,
            pady=(20, 5)
        )

        tk.Button(
            navigation,
            text="← Back",
            command=(
                self.show_home_page
            )
        ).pack(
            side="left"
        )

        tk.Label(
            navigation,
            text=(
                f"{price_area} "
                f"Market Analysis"
            ),
            font=(
                "Arial",
                22,
                "bold"
            )
        ).pack(
            side="left",
            padx=20
        )

        content = tk.Frame(
            self
        )

        content.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )

        # --------------------------------------------------
        # Info
        # --------------------------------------------------

        info_frame = tk.Frame(
            content
        )

        info_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        latest_price = (
            prices[
                "DayAheadPriceEUR"
            ].iloc[
                -1
            ]
        )

        information = [
            (
                "Market",
                price_area
            ),
            (
                "Time Range",
                time_range
            ),
            (
                "Latest Price",
                (
                    f"{latest_price:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Data Points",
                (
                    f"{len(prices):,}"
                )
            ),
            (
                "Chart Resolution",
                resolution
            )
        ]

        for (
            name,
            value
        ) in information:

            tk.Label(
                info_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    12,
                    "bold"
                ),
                padx=20
            ).pack(
                side="left"
            )

        # ==================================================
        # Navigation
        # ==================================================

        analysis_navigation = (
            tk.Frame(
                content
            )
        )

        analysis_navigation.pack(
            fill="x",
            pady=(5, 10)
        )

        navigation_row_1 = (
            tk.Frame(
                analysis_navigation
            )
        )

        navigation_row_1.pack(
            anchor="w",
            pady=(0, 5)
        )

        navigation_row_2 = (
            tk.Frame(
                analysis_navigation
            )
        )

        navigation_row_2.pack(
            anchor="w"
        )

        analysis_content = (
            tk.Frame(
                content
            )
        )

        analysis_content.pack(
            fill="both",
            expand=True
        )

        # --------------------------------------------------
        # Row 1
        # --------------------------------------------------

        row_1_buttons = [
            (
                "Price Chart",
                lambda:
                self.show_price_view(
                    analysis_content,
                    chart_data,
                    price_area,
                    time_range,
                    resolution
                )
            ),
            (
                "Returns",
                lambda:
                self.show_returns_view(
                    analysis_content,
                    returns_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Statistics",
                lambda:
                self.show_statistics_view(
                    analysis_content,
                    returns_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Volatility",
                lambda:
                self.show_volatility_view(
                    analysis_content,
                    volatility_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Correlation",
                lambda:
                self.show_correlation_view(
                    analysis_content,
                    correlation_data,
                    time_range
                )
            ),
            (
                "Spread Signals",
                lambda:
                self.spread_view.show(
                    analysis_content,
                    dk1_prices,
                    dk2_prices,
                    time_range
                )
            ),
            (
                "Portfolio",
                lambda:
                self.show_portfolio_view(
                    analysis_content,
                    correlation_data,
                    time_range
                )
            )
        ]

        # --------------------------------------------------
        # Row 2
        # --------------------------------------------------

        row_2_buttons = [
            (
                "Risk",
                lambda:
                self.show_risk_view(
                    analysis_content,
                    returns_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Monte Carlo",
                lambda:
                self.show_monte_carlo_view(
                    analysis_content,
                    returns_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Stress Test",
                lambda:
                self.show_stress_test_view(
                    analysis_content,
                    returns_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Forecast",
                lambda:
                self.show_forecast_view(
                    analysis_content,
                    chart_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Market Factors",
                lambda:
                self.market_factors_view.show(
                    analysis_content,
                    feature_data,
                    price_area,
                    time_range
                )
            ),
            (
                "Machine Learning",
                lambda:
                self.machine_learning_view.show(
                    analysis_content,
                    feature_data,
                    price_area,
                    time_range
                )
            )
        ]

        for (
            button_text,
            command
        ) in row_1_buttons:

            tk.Button(
                navigation_row_1,
                text=button_text,
                command=command
            ).pack(
                side="left",
                padx=(0, 8)
            )

        for (
            button_text,
            command
        ) in row_2_buttons:

            tk.Button(
                navigation_row_2,
                text=button_text,
                command=command
            ).pack(
                side="left",
                padx=(0, 8)
            )

        self.show_price_view(
            analysis_content,
            chart_data,
            price_area,
            time_range,
            resolution
        )

    # ==================================================
    # Price
    # ==================================================

    def show_price_view(
        self,
        container,
        chart_data,
        price_area,
        time_range,
        resolution
    ):

        self.clear_frame(
            container
        )

        figure = (
            self.chart_builder
            .create_price_chart(
                chart_data,
                price_area,
                time_range,
                resolution
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Returns
    # ==================================================

    def show_returns_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        statistics = (
            self.return_calculator
            .get_return_statistics(
                returns_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        values = [
            (
                "Mean Return",
                statistics[
                    "mean"
                ]
            ),
            (
                "Minimum Return",
                statistics[
                    "minimum"
                ]
            ),
            (
                "Maximum Return",
                statistics[
                    "maximum"
                ]
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value:.2f}%"
                ),
                font=(
                    "Arial",
                    11,
                    "bold"
                ),
                padx=25
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_returns_chart(
                returns_data,
                price_area,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Statistics
    # ==================================================

    def show_statistics_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        statistics = (
            self.statistics_analyzer
            .calculate_statistics(
                returns_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        stats = [
            (
                "Mean",
                statistics[
                    "mean"
                ],
                "%"
            ),
            (
                "Std. Deviation",
                statistics[
                    "std"
                ],
                "%"
            ),
            (
                "Minimum",
                statistics[
                    "minimum"
                ],
                "%"
            ),
            (
                "Maximum",
                statistics[
                    "maximum"
                ],
                "%"
            ),
            (
                "Skewness",
                statistics[
                    "skewness"
                ],
                ""
            ),
            (
                "Kurtosis",
                statistics[
                    "kurtosis"
                ],
                ""
            )
        ]

        for (
            name,
            value,
            suffix
        ) in stats:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value:.2f}"
                    f"{suffix}"
                ),
                font=(
                    "Arial",
                    11,
                    "bold"
                ),
                padx=15
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_distribution_chart(
                returns_data,
                price_area,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Volatility
    # ==================================================

    def show_volatility_view(
        self,
        container,
        volatility_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        statistics = (
            self.volatility_analyzer
            .get_volatility_statistics(
                volatility_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        values = [
            (
                "Current 7-Period",
                statistics[
                    "current_7"
                ]
            ),
            (
                "Current 30-Period",
                statistics[
                    "current_30"
                ]
            ),
            (
                "Average 30-Period",
                statistics[
                    "average_30"
                ]
            ),
            (
                "Maximum 30-Period",
                statistics[
                    "maximum_30"
                ]
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value:.2f}%"
                ),
                font=(
                    "Arial",
                    11,
                    "bold"
                ),
                padx=20
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_volatility_chart(
                volatility_data,
                price_area,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Correlation
    # ==================================================

    def show_correlation_view(
        self,
        container,
        correlation_data,
        time_range
    ):

        self.clear_frame(
            container
        )

        statistics = (
            self.correlation_analyzer
            .calculate_correlation(
                correlation_data
            )
        )

        info_frame = tk.Frame(
            container
        )

        info_frame.pack(
            fill="x"
        )

        values = [
            (
                "Pearson Correlation",
                (
                    f"{statistics['correlation']:.3f}"
                )
            ),
            (
                "Relationship",
                statistics[
                    "strength"
                ]
            ),
            (
                "Matched Observations",
                (
                    f"{statistics['observations']:,}"
                )
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                info_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    11,
                    "bold"
                ),
                padx=25
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_correlation_chart(
                correlation_data,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Portfolio
    # ==================================================

    def show_portfolio_view(
        self,
        container,
        correlation_data,
        time_range
    ):

        self.clear_frame(
            container
        )

        tk.Label(
            container,
            text="Portfolio Construction",
            font=(
                "Arial",
                15,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(5, 5)
        )

        controls = tk.Frame(
            container
        )

        controls.pack(
            fill="x",
            pady=(0, 10)
        )

        weight_variable = (
            tk.DoubleVar(
                value=50
            )
        )

        tk.Label(
            controls,
            text="DK1 Weight"
        ).pack(
            side="left"
        )

        tk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            variable=weight_variable,
            length=300
        ).pack(
            side="left",
            padx=10
        )

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        def display_portfolio(
            dk1_weight
        ):

            self.clear_frame(
                output_frame
            )

            portfolio_data = (
                self.portfolio_analyzer
                .build_portfolio(
                    correlation_data,
                    dk1_weight
                )
            )

            statistics = (
                self.portfolio_analyzer
                .calculate_portfolio_statistics(
                    portfolio_data,
                    dk1_weight
                )
            )

            stats_frame = tk.Frame(
                output_frame
            )

            stats_frame.pack(
                fill="x"
            )

            values = [
                (
                    "DK1 Weight",
                    (
                        f"{statistics['dk1_weight'] * 100:.1f}%"
                    )
                ),
                (
                    "DK2 Weight",
                    (
                        f"{statistics['dk2_weight'] * 100:.1f}%"
                    )
                ),
                (
                    "Mean Return",
                    (
                        f"{statistics['mean_return']:.2f}%"
                    )
                ),
                (
                    "Volatility",
                    (
                        f"{statistics['volatility']:.2f}%"
                    )
                ),
                (
                    "Diversification",
                    (
                        f"{statistics['diversification_benefit']:.2f}%"
                    )
                )
            ]

            for (
                name,
                value
            ) in values:

                tk.Label(
                    stats_frame,
                    text=(
                        f"{name}\n"
                        f"{value}"
                    ),
                    font=(
                        "Arial",
                        10,
                        "bold"
                    ),
                    padx=15
                ).pack(
                    side="left"
                )

            figure = (
                self.chart_builder
                .create_portfolio_chart(
                    portfolio_data,
                    time_range,
                    dk1_weight
                )
            )

            self.embed_chart(
                output_frame,
                figure
            )

        def update_portfolio():

            display_portfolio(
                weight_variable.get()
                / 100
            )

        def optimize():

            result = (
                self.portfolio_optimizer
                .find_minimum_volatility_portfolio(
                    correlation_data
                )
            )

            weight_variable.set(
                result[
                    "dk1_weight"
                ]
                * 100
            )

            display_portfolio(
                result[
                    "dk1_weight"
                ]
            )

        tk.Button(
            controls,
            text="Update Portfolio",
            command=update_portfolio
        ).pack(
            side="left",
            padx=10
        )

        tk.Button(
            controls,
            text="Minimum Volatility",
            font=(
                "Arial",
                10,
                "bold"
            ),
            command=optimize
        ).pack(
            side="left",
            padx=10
        )

        display_portfolio(
            0.50
        )

    # ==================================================
    # Risk
    # ==================================================

    def show_risk_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        var_statistics = (
            self.var_analyzer
            .calculate_historical_var(
                returns_data
            )
        )

        es_statistics = (
            self.es_analyzer
            .calculate_historical_expected_shortfall(
                returns_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        values = [
            (
                "95% VaR",
                (
                    f"{var_statistics['var_95']:.2f}%"
                )
            ),
            (
                "99% VaR",
                (
                    f"{var_statistics['var_99']:.2f}%"
                )
            ),
            (
                "95% ES",
                (
                    f"{es_statistics['es_95']:.2f}%"
                )
            ),
            (
                "99% ES",
                (
                    f"{es_statistics['es_99']:.2f}%"
                )
            ),
            (
                "Worst Return",
                (
                    f"{var_statistics['worst_return']:.2f}%"
                )
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    10,
                    "bold"
                ),
                padx=15
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_risk_chart(
                returns_data,
                price_area,
                time_range,
                var_statistics,
                es_statistics
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Monte Carlo
    # ==================================================

    def show_monte_carlo_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        results = (
            self.monte_carlo_simulator
            .run_simulation(
                returns_data
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        values = [
            (
                "Expected Return",
                (
                    f"{results['expected_return']:.2f}%"
                )
            ),
            (
                "Median",
                (
                    f"{results['median_return']:.2f}%"
                )
            ),
            (
                "5th Percentile",
                (
                    f"{results['percentile_5']:.2f}%"
                )
            ),
            (
                "95th Percentile",
                (
                    f"{results['percentile_95']:.2f}%"
                )
            ),
            (
                "Probability of Loss",
                (
                    f"{results['probability_of_loss']:.2f}%"
                )
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    10,
                    "bold"
                ),
                padx=15
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_monte_carlo_chart(
                results,
                price_area,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )

    # ==================================================
    # Stress test
    # ==================================================

    def show_stress_test_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        scenarios = (
            self.stress_tester
            .get_scenarios()
        )

        scenario_variable = (
            tk.StringVar(
                value=scenarios[
                    0
                ]
            )
        )

        controls = tk.Frame(
            container
        )

        controls.pack(
            fill="x"
        )

        ttk.Combobox(
            controls,
            textvariable=(
                scenario_variable
            ),
            values=scenarios,
            state="readonly",
            width=22
        ).pack(
            side="left"
        )

        output_frame = tk.Frame(
            container
        )

        output_frame.pack(
            fill="both",
            expand=True
        )

        def run():

            self.clear_frame(
                output_frame
            )

            results = (
                self.stress_tester
                .run_stress_test(
                    returns_data,
                    scenario_variable.get()
                )
            )

            figure = (
                self.chart_builder
                .create_stress_test_chart(
                    results,
                    price_area,
                    time_range
                )
            )

            self.embed_chart(
                output_frame,
                figure
            )

        tk.Button(
            controls,
            text="Run Stress Test",
            command=run
        ).pack(
            side="left",
            padx=10
        )

        run()

    # ==================================================
    # Baseline Forecast
    # ==================================================

    def show_forecast_view(
        self,
        container,
        price_data,
        price_area,
        time_range
    ):

        self.clear_frame(
            container
        )

        results = (
            self.price_forecaster
            .forecast(
                price_data,
                method="Moving Average",
                horizon=10
            )
        )

        stats_frame = tk.Frame(
            container
        )

        stats_frame.pack(
            fill="x"
        )

        values = [
            (
                "Method",
                results[
                    "method"
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
                "Forecast Price",
                (
                    f"{results['forecast_price']:.2f} "
                    f"EUR/MWh"
                )
            ),
            (
                "Change",
                (
                    f"{results['price_change']:+.2f} "
                    f"EUR/MWh"
                )
            )
        ]

        for (
            name,
            value
        ) in values:

            tk.Label(
                stats_frame,
                text=(
                    f"{name}\n"
                    f"{value}"
                ),
                font=(
                    "Arial",
                    10,
                    "bold"
                ),
                padx=15
            ).pack(
                side="left"
            )

        figure = (
            self.chart_builder
            .create_forecast_chart(
                price_data,
                results,
                price_area,
                time_range
            )
        )

        self.embed_chart(
            container,
            figure
        )