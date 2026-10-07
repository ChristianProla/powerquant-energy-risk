import tkinter as tk
from tkinter import ttk, messagebox
import requests

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.api.electricity_api import ElectricityAPI
from src.visualization.charts import ChartBuilder


class PowerQuantApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("PowerQuant")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        self.api = ElectricityAPI()
        self.chart_builder = ChartBuilder()

        self.selected_range = "1D"

        self.time_ranges = {
            "1H": ("now-PT1H", "1 Hour"),
            "1D": ("now-P1D", "1 Day"),
            "1W": ("now-P7D", "1 Week"),
            "1M": ("now-P1M", "1 Month"),
            "1Y": ("now-P1Y", "1 Year"),
            "5Y": ("now-P5Y", "5 Years"),
            "10Y": ("now-P10Y", "10 Years")
        }

        self.show_home_page()

    # --------------------------------------------------
    # Clear current page
    # --------------------------------------------------

    def clear_window(self):

        for widget in self.winfo_children():
            widget.destroy()

    # --------------------------------------------------
    # Home page
    # --------------------------------------------------

    def show_home_page(self):

        self.clear_window()

        # Header

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

        # Main content

        content = tk.Frame(self)

        content.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=25
        )

        # Market

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
            "1Y",
            "5Y",
            "10Y"
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

        # Selected range

        self.range_label = tk.Label(
            content,
            text="Selected: 1 Day",
            font=("Arial", 11)
        )

        self.range_label.pack(
            anchor="w",
            pady=(0, 20)
        )

        # Load button

        load_button = tk.Button(
            content,
            text="Open Market Analysis",
            font=("Arial", 13, "bold"),
            padx=15,
            pady=8,
            command=self.load_market_data
        )

        load_button.pack(anchor="w")

        # Status

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

            self.show_market_page(
                prices,
                price_area,
                readable_range
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
        price_area,
        time_range
    ):

        self.clear_window()

        # Top navigation

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

        # Main area

        content = tk.Frame(self)

        content.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )

        # Statistics

        info_frame = tk.Frame(content)

        info_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        latest_price = prices[
            "DayAheadPriceEUR"
        ].iloc[-1]

        number_of_points = len(prices)

        first_date = prices[
            "TimeDK"
        ].min()

        last_date = prices[
            "TimeDK"
        ].max()

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

        # Graph

        figure = self.chart_builder.create_price_chart(
            prices,
            price_area,
            time_range
        )

        canvas = FigureCanvasTkAgg(
            figure,
            master=content
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        # Date information

        date_label = tk.Label(
            content,
            text=(
                f"Data from {first_date} "
                f"to {last_date}"
            ),
            font=("Arial", 10)
        )

        date_label.pack(
            anchor="w",
            pady=(10, 0)
        )