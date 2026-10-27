import numpy as np
import pandas as pd


class SpreadAnalyzer:

    # ==================================================
    # Prepare DK1 / DK2 spread data
    # ==================================================

    def prepare_spread_data(
        self,
        dk1_prices,
        dk2_prices
    ):

        if (
            dk1_prices.empty
            or
            dk2_prices.empty
        ):

            return pd.DataFrame()

        dk1 = dk1_prices[
            [
                "TimeDK",
                "DayAheadPriceEUR"
            ]
        ].copy()

        dk2 = dk2_prices[
            [
                "TimeDK",
                "DayAheadPriceEUR"
            ]
        ].copy()

        # --------------------------------------------------
        # Timestamps
        # --------------------------------------------------

        dk1[
            "TimeDK"
        ] = pd.to_datetime(
            dk1[
                "TimeDK"
            ],
            errors="coerce"
        )

        dk2[
            "TimeDK"
        ] = pd.to_datetime(
            dk2[
                "TimeDK"
            ],
            errors="coerce"
        )

        dk1 = dk1.dropna(
            subset=[
                "TimeDK",
                "DayAheadPriceEUR"
            ]
        )

        dk2 = dk2.dropna(
            subset=[
                "TimeDK",
                "DayAheadPriceEUR"
            ]
        )

        # --------------------------------------------------
        # Convert both markets to hourly observations
        # --------------------------------------------------

        dk1[
            "TimeDK"
        ] = (
            dk1[
                "TimeDK"
            ].dt.floor(
                "h"
            )
        )

        dk2[
            "TimeDK"
        ] = (
            dk2[
                "TimeDK"
            ].dt.floor(
                "h"
            )
        )

        dk1 = (
            dk1.groupby(
                "TimeDK",
                as_index=False
            )[
                "DayAheadPriceEUR"
            ]
            .mean()
        )

        dk2 = (
            dk2.groupby(
                "TimeDK",
                as_index=False
            )[
                "DayAheadPriceEUR"
            ]
            .mean()
        )

        dk1 = dk1.rename(
            columns={
                "DayAheadPriceEUR":
                    "DK1Price"
            }
        )

        dk2 = dk2.rename(
            columns={
                "DayAheadPriceEUR":
                    "DK2Price"
            }
        )

        # --------------------------------------------------
        # Merge the two electricity markets
        # --------------------------------------------------

        spread_data = pd.merge(
            dk1,
            dk2,
            on="TimeDK",
            how="inner"
        )

        spread_data = (
            spread_data.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------
        # Spread
        #
        # Positive:
        # DK1 is more expensive than DK2
        #
        # Negative:
        # DK1 is cheaper than DK2
        # --------------------------------------------------

        spread_data[
            "Spread"
        ] = (
            spread_data[
                "DK1Price"
            ]
            - spread_data[
                "DK2Price"
            ]
        )

        return spread_data

    # ==================================================
    # Calculate rolling spread statistics
    # ==================================================

    def calculate_signals(
        self,
        spread_dataframe,
        window=24,
        threshold=2.0
    ):

        if spread_dataframe.empty:

            return pd.DataFrame()

        data = (
            spread_dataframe.copy()
        )

        window = max(
            5,
            int(
                window
            )
        )

        threshold = max(
            0.5,
            float(
                threshold
            )
        )

        # --------------------------------------------------
        # Rolling spread mean
        # --------------------------------------------------

        data[
            "RollingMean"
        ] = (
            data[
                "Spread"
            ].rolling(
                window=window,
                min_periods=window
            ).mean()
        )

        # --------------------------------------------------
        # Rolling spread volatility
        # --------------------------------------------------

        data[
            "RollingStd"
        ] = (
            data[
                "Spread"
            ].rolling(
                window=window,
                min_periods=window
            ).std()
        )

        # --------------------------------------------------
        # Z-score
        #
        # z = (spread - rolling mean) / rolling std
        # --------------------------------------------------

        valid_std = (
            data[
                "RollingStd"
            ].replace(
                0,
                np.nan
            )
        )

        data[
            "ZScore"
        ] = (
            (
                data[
                    "Spread"
                ]
                - data[
                    "RollingMean"
                ]
            )
            / valid_std
        )

        # --------------------------------------------------
        # Research signals
        # --------------------------------------------------

        conditions = [
            (
                data[
                    "ZScore"
                ]
                >= threshold
            ),
            (
                data[
                    "ZScore"
                ]
                <= -threshold
            )
        ]

        signals = [
            (
                "DK1 Expensive / "
                "DK2 Cheap"
            ),
            (
                "DK1 Cheap / "
                "DK2 Expensive"
            )
        ]

        data[
            "Signal"
        ] = np.select(
            conditions,
            signals,
            default="Neutral"
        )

        data[
            "SignalValue"
        ] = np.select(
            conditions,
            [
                -1,
                1
            ],
            default=0
        )

        return data

    # ==================================================
    # Current spread statistics
    # ==================================================

    def get_spread_statistics(
        self,
        signal_dataframe,
        threshold=2.0
    ):

        if signal_dataframe.empty:

            return (
                self._empty_statistics()
            )

        data = (
            signal_dataframe.copy()
        )

        latest = data.iloc[
            -1
        ]

        current_spread = (
            latest[
                "Spread"
            ]
        )

        current_z_score = (
            latest[
                "ZScore"
            ]
        )

        rolling_mean = (
            latest[
                "RollingMean"
            ]
        )

        rolling_std = (
            latest[
                "RollingStd"
            ]
        )

        current_signal = (
            latest[
                "Signal"
            ]
        )

        # --------------------------------------------------
        # Handle early periods with insufficient
        # rolling observations
        # --------------------------------------------------

        if pd.isna(
            current_z_score
        ):

            current_z_score = 0

            current_signal = (
                "Not Enough Data"
            )

        if pd.isna(
            rolling_mean
        ):

            rolling_mean = 0

        if pd.isna(
            rolling_std
        ):

            rolling_std = 0

        # --------------------------------------------------
        # Historical signal counts
        # --------------------------------------------------

        upper_signals = (
            data[
                "ZScore"
            ]
            >= threshold
        ).sum()

        lower_signals = (
            data[
                "ZScore"
            ]
            <= -threshold
        ).sum()

        total_signals = (
            upper_signals
            + lower_signals
        )

        # --------------------------------------------------
        # Average spread
        # --------------------------------------------------

        average_spread = (
            data[
                "Spread"
            ].mean()
        )

        spread_volatility = (
            data[
                "Spread"
            ].std()
        )

        maximum_spread = (
            data[
                "Spread"
            ].max()
        )

        minimum_spread = (
            data[
                "Spread"
            ].min()
        )

        return {
            "current_spread":
                current_spread,

            "current_z_score":
                current_z_score,

            "current_signal":
                current_signal,

            "rolling_mean":
                rolling_mean,

            "rolling_std":
                rolling_std,

            "average_spread":
                average_spread,

            "spread_volatility":
                spread_volatility,

            "maximum_spread":
                maximum_spread,

            "minimum_spread":
                minimum_spread,

            "upper_signals":
                int(
                    upper_signals
                ),

            "lower_signals":
                int(
                    lower_signals
                ),

            "total_signals":
                int(
                    total_signals
                ),

            "observations":
                len(
                    data
                )
        }

    # ==================================================
    # Empty statistics
    # ==================================================

    def _empty_statistics(
        self
    ):

        return {
            "current_spread": 0,
            "current_z_score": 0,
            "current_signal": "No Data",
            "rolling_mean": 0,
            "rolling_std": 0,
            "average_spread": 0,
            "spread_volatility": 0,
            "maximum_spread": 0,
            "minimum_spread": 0,
            "upper_signals": 0,
            "lower_signals": 0,
            "total_signals": 0,
            "observations": 0
        }