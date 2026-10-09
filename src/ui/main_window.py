import tkinter as tk
from tkinter import ttk, messagebox

import requests

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.api.electricity_api import ElectricityAPI
from src.analytics.resampling import MarketDataResampler
from src.analytics.returns import ReturnCalculator
from src.visualization.charts import ChartBuilder


class PowerQuantApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("PowerQuant")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        self.api = ElectricityAPI()
        self.chart_builder = ChartBuilder()
        self.resampler = MarketDataResampler()
        self.return_calculator = ReturnCalculator()

        self.selected_range = "1D"

        self.time_ranges = {
            "1H": ("now-PT1H", "1 Hour"),
            "1D": ("now-P1D", "1 Day"),
            "1W": ("now-P7D", "1 Week"),
            "1M": ("now-P1M", "1 Month"),
            "1Y": ("now-P1Y", "1 Year")
        }

        self.show_home_page()

    # --------------------------------------------------
    # Clear window
    # --------------------------------------------------

    def clear_window(self):

        for widget in self.winfo_children():
            widget.destroy()

    # --------------------------------------------------
    # Clear frame
    # --------------------------------------------------

    def clear_frame(self, frame):

        for widget in frame.winfo_children():
            widget.destroy()

    # --------------------------------------------------
    # Home page
    # --------------------------------------------------

    def show_home_page(self):

        self.clear_window()

        header = tk.Frame(self)

        header.pack(
            fill="x",
            padx=40,
            pady=(30, 10)
        )

        title = tk.Label(
            header,
            text="PowerQuant",
            font=("Arial", 30, "bold")
        )

        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Electricity Market Analytics & Risk Platform",
            font=("Arial", 14)
        )

        subtitle.pack(anchor="w")

        content = tk.Frame(self)

        content.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=25
        )

        # Market selection

        market_title = tk.Label(
            content,
            text="Market",
            font=("Arial", 15, "bold")
        )

        market_title.pack(anchor="w")

        self.market_selector = ttk.Combobox(
            content,
            values=["DK1", "DK2"],
            state="readonly",
            width=15
        )

        self.market_selector.set("DK1")

        self.market_selector.pack(
            anchor="w",
            pady=(8, 25)
        )

        # Time range

        range_title = tk.Label(
            content,
            text="Time Range",
            font=("Arial", 15, "bold")
        )

        range_title.pack(anchor="w")

        range_frame = tk.Frame(content)

        range_frame.pack(
            anchor="w",
            pady=(10, 30)
        )

        ranges = [
            "1H",
            "1D",
            "1W",
            "1M",
            "1Y"
        ]

        for time_range in ranges:

            button = tk.Button(
                range_frame,
                text=time_range,
                width=7,
                command=lambda value=time_range:
                self.select_range(value)
            )

            button.pack(
                side="left",
                padx=(0, 8)
            )

        self.range_label = tk.Label(
            content,
            text="Selected: 1 Day",
            font=("Arial", 11)
        )

        self.range_label.pack(
            anchor="w",
            pady=(0, 20)
        )

        load_button = tk.Button(
            content,
            text="Open Market Analysis",
            font=("Arial", 13, "bold"),
            padx=15,
            pady=8,
            command=self.load_market_data
        )

        load_button.pack(anchor="w")

        self.status_label = tk.Label(
            content,
            text="Ready",
            font=("Arial", 11)
        )

        self.status_label.pack(
            anchor="w",
            pady=(20, 0)
        )

    # --------------------------------------------------
    # Select time range
    # --------------------------------------------------

    def select_range(self, time_range):

        self.selected_range = time_range

        readable_name = self.time_ranges[
            time_range
        ][1]

        self.range_label.config(
            text=f"Selected: {readable_name}"
        )

    # --------------------------------------------------
    # Load market data
    # --------------------------------------------------

    def load_market_data(self):

        price_area = self.market_selector.get()

        start_date = self.time_ranges[
            self.selected_range
        ][0]

        readable_range = self.time_ranges[
            self.selected_range
        ][1]

        self.status_label.config(
            text="Loading market data..."
        )

        self.update()

        try:

            prices = self.api.get_prices(
                price_area=price_area,
                start_date=start_date,
                end_date="now"
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
                self.resampler.resample_for_range(
                    prices,
                    self.selected_range
                )
            )

            returns_data = (
                self.return_calculator.calculate_returns(
                    chart_data
                )
            )

            self.show_market_page(
                prices,
                chart_data,
                returns_data,
                price_area,
                readable_range,
                resolution
            )

        except requests.exceptions.ConnectionError:

            self.status_label.config(
                text="Connection failed"
            )

            messagebox.showerror(
                "Connection Error",
                "Unable to connect to Energinet.\n\n"
                "Please check your internet connection and try again."
            )

        except requests.exceptions.Timeout:

            self.status_label.config(
                text="Request timed out"
            )

            messagebox.showerror(
                "Timeout",
                "Energinet did not respond in time.\n\n"
                "Please try again."
            )

        except requests.exceptions.HTTPError as error:

            self.status_label.config(
                text="API error"
            )

            messagebox.showerror(
                "API Error",
                f"Energinet returned an error.\n\n{error}"
            )

        except Exception as error:

            self.status_label.config(
                text="Unable to load market data"
            )

            messagebox.showerror(
                "PowerQuant Error",
                f"An unexpected error occurred.\n\n{error}"
            )

    # --------------------------------------------------
    # Market page
    # --------------------------------------------------

    def show_market_page(
        self,
        prices,
        chart_data,
        returns_data,
        price_area,
        time_range,
        resolution
    ):

        self.clear_window()

        # Navigation

        navigation = tk.Frame(self)

        navigation.pack(
            fill="x",
            padx=30,
            pady=(20, 5)
        )

        back_button = tk.Button(
            navigation,
            text="← Back",
            command=self.show_home_page
        )

        back_button.pack(side="left")

        page_title = tk.Label(
            navigation,
            text=f"{price_area} Market Analysis",
            font=("Arial", 22, "bold")
        )

        page_title.pack(
            side="left",
            padx=20
        )

        # Main content

        content = tk.Frame(self)

        content.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )

        # Basic market information

        info_frame = tk.Frame(content)

        info_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        latest_price = prices[
            "DayAheadPriceEUR"
        ].iloc[-1]

        number_of_points = len(prices)

        market_label = tk.Label(
            info_frame,
            text=f"Market\n{price_area}",
            font=("Arial", 12, "bold"),
            padx=25
        )

        market_label.pack(side="left")

        range_label = tk.Label(
            info_frame,
            text=f"Time Range\n{time_range}",
            font=("Arial", 12, "bold"),
            padx=25
        )

        range_label.pack(side="left")

        price_label = tk.Label(
            info_frame,
            text=(
                f"Latest Price\n"
                f"{latest_price:.2f} EUR/MWh"
            ),
            font=("Arial", 12, "bold"),
            padx=25
        )

        price_label.pack(side="left")

        points_label = tk.Label(
            info_frame,
            text=(
                f"Data Points\n"
                f"{number_of_points:,}"
            ),
            font=("Arial", 12, "bold"),
            padx=25
        )

        points_label.pack(side="left")

        resolution_label = tk.Label(
            info_frame,
            text=(
                f"Chart Resolution\n"
                f"{resolution}"
            ),
            font=("Arial", 12, "bold"),
            padx=25
        )

        resolution_label.pack(side="left")

        # -----------------------------------------
        # Analysis navigation
        # -----------------------------------------

        analysis_navigation = tk.Frame(content)

        analysis_navigation.pack(
            fill="x",
            pady=(5, 10)
        )

        analysis_content = tk.Frame(content)

        analysis_content.pack(
            fill="both",
            expand=True
        )

        price_button = tk.Button(
            analysis_navigation,
            text="Price Chart",
            command=lambda:
            self.show_price_view(
                analysis_content,
                chart_data,
                price_area,
                time_range,
                resolution
            )
        )

        price_button.pack(
            side="left",
            padx=(0, 8)
        )

        returns_button = tk.Button(
            analysis_navigation,
            text="Returns",
            command=lambda:
            self.show_returns_view(
                analysis_content,
                returns_data,
                price_area,
                time_range
            )
        )

        returns_button.pack(
            side="left"
        )

        # Show price view by default

        self.show_price_view(
            analysis_content,
            chart_data,
            price_area,
            time_range,
            resolution
        )

    # --------------------------------------------------
    # Price view
    # --------------------------------------------------

    def show_price_view(
        self,
        container,
        chart_data,
        price_area,
        time_range,
        resolution
    ):

        self.clear_frame(container)

        figure = self.chart_builder.create_price_chart(
            chart_data,
            price_area,
            time_range,
            resolution
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
    # Returns view
    # --------------------------------------------------

    def show_returns_view(
        self,
        container,
        returns_data,
        price_area,
        time_range
    ):

        self.clear_frame(container)

        if returns_data.empty:

            message = tk.Label(
                container,
                text="Not enough data to calculate returns.",
                font=("Arial", 12)
            )

            message.pack(pady=30)

            return

        statistics = (
            self.return_calculator.get_return_statistics(
                returns_data
            )
        )

        # Return statistics

        stats_frame = tk.Frame(container)

        stats_frame.pack(
            fill="x",
            pady=(5, 10)
        )

        mean_label = tk.Label(
            stats_frame,
            text=(
                f"Mean Return\n"
                f"{statistics['mean']:.2f}%"
            ),
            font=("Arial", 11, "bold"),
            padx=25
        )

        mean_label.pack(side="left")

        minimum_label = tk.Label(
            stats_frame,
            text=(
                f"Minimum Return\n"
                f"{statistics['minimum']:.2f}%"
            ),
            font=("Arial", 11, "bold"),
            padx=25
        )

        minimum_label.pack(side="left")

        maximum_label = tk.Label(
            stats_frame,
            text=(
                f"Maximum Return\n"
                f"{statistics['maximum']:.2f}%"
            ),
            font=("Arial", 11, "bold"),
            padx=25
        )

        maximum_label.pack(side="left")

        # Returns chart

        figure = self.chart_builder.create_returns_chart(
            returns_data,
            price_area,
            time_range
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