import numpy as np
import pandas as pd


class MarketRegimeDetector:

    # ==================================================
    # Main analysis
    # ==================================================

    def analyze(
        self,
        spread_dataframe,
        simulation_data,
        signal_window=48
    ):

        if spread_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        try:

            signal_window = int(
                signal_window
            )

        except (
            TypeError,
            ValueError
        ):

            return self._empty_result(
                "Signal window must be numeric."
            )

        if signal_window < 12:

            return self._empty_result(
                (
                    "Signal window must contain "
                    "at least 12 observations."
                )
            )

        regimes = (
            self._build_regime_dataframe(
                spread_dataframe,
                signal_window
            )
        )

        if regimes.empty:

            return self._empty_result(
                (
                    "Not enough observations to "
                    "calculate market regimes."
                )
            )

        current_regime = (
            self._get_current_regime(
                regimes
            )
        )

        # ==================================================
        # Strategy attribution
        # ==================================================

        if simulation_data.empty:

            return {
                "success":
                    True,

                "message":
                    (
                        "Market regimes calculated, "
                        "but no completed portfolio trades "
                        "were available for attribution."
                    ),

                "regime_data":
                    regimes,

                "trade_regimes":
                    pd.DataFrame(),

                "volatility_attribution":
                    pd.DataFrame(),

                "structure_attribution":
                    pd.DataFrame(),

                "spread_state_attribution":
                    pd.DataFrame(),

                "current_regime":
                    current_regime,

                "best_volatility_regime":
                    "N/A",

                "best_structure_regime":
                    "N/A",

                "worst_volatility_regime":
                    "N/A",

                "regime_summary":
                    "No completed trades available."
            }

        trade_regimes = (
            self._match_trades_to_regimes(
                simulation_data,
                regimes
            )
        )

        volatility_attribution = (
            self._calculate_attribution(
                trade_regimes,
                "VolatilityRegime"
            )
        )

        structure_attribution = (
            self._calculate_attribution(
                trade_regimes,
                "StructureRegime"
            )
        )

        spread_state_attribution = (
            self._calculate_attribution(
                trade_regimes,
                "SpreadState"
            )
        )

        best_volatility_regime = (
            self._best_regime(
                volatility_attribution
            )
        )

        worst_volatility_regime = (
            self._worst_regime(
                volatility_attribution
            )
        )

        best_structure_regime = (
            self._best_regime(
                structure_attribution
            )
        )

        regime_summary = (
            self._create_summary(
                volatility_attribution,
                structure_attribution
            )
        )

        return {
            "success":
                True,

            "message":
                (
                    "Market regime analysis "
                    "completed successfully."
                ),

            "regime_data":
                regimes,

            "trade_regimes":
                trade_regimes,

            "volatility_attribution":
                volatility_attribution,

            "structure_attribution":
                structure_attribution,

            "spread_state_attribution":
                spread_state_attribution,

            "current_regime":
                current_regime,

            "best_volatility_regime":
                best_volatility_regime,

            "best_structure_regime":
                best_structure_regime,

            "worst_volatility_regime":
                worst_volatility_regime,

            "regime_summary":
                regime_summary
        }

    # ==================================================
    # Build regime dataframe
    # ==================================================

    def _build_regime_dataframe(
        self,
        spread_dataframe,
        signal_window
    ):

        data = (
            spread_dataframe.copy()
        )

        required_columns = [
            "TimeDK",
            "DK1Price",
            "DK2Price",
            "Spread"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            return pd.DataFrame()

        data[
            "TimeDK"
        ] = pd.to_datetime(
            data[
                "TimeDK"
            ]
        )

        data = (
            data.sort_values(
                "TimeDK"
            )
            .reset_index(
                drop=True
            )
        )

        # ==================================================
        # Spread movement
        # ==================================================

        data[
            "SpreadChange"
        ] = (
            data[
                "Spread"
            ].diff()
        )

        minimum_periods = max(
            12,
            signal_window // 4
        )

        data[
            "RollingSpreadVolatility"
        ] = (
            data[
                "SpreadChange"
            ]
            .rolling(
                window=signal_window,
                min_periods=minimum_periods
            )
            .std()
        )

        # ==================================================
        # Rolling spread statistics
        # ==================================================

        data[
            "RollingMean"
        ] = (
            data[
                "Spread"
            ]
            .rolling(
                window=signal_window,
                min_periods=minimum_periods
            )
            .mean()
        )

        data[
            "RollingStd"
        ] = (
            data[
                "Spread"
            ]
            .rolling(
                window=signal_window,
                min_periods=minimum_periods
            )
            .std()
        )

        safe_std = (
            data[
                "RollingStd"
            ]
            .replace(
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
                -
                data[
                    "RollingMean"
                ]
            )
            /
            safe_std
        )

        # ==================================================
        # Historical volatility thresholds
        #
        # Expanding historical quantiles are shifted by
        # one observation.
        #
        # Therefore the current observation never helps
        # determine its own volatility-regime threshold.
        # ==================================================

        historical_volatility = (
            data[
                "RollingSpreadVolatility"
            ]
        )

        expanding = (
            historical_volatility.expanding(
                min_periods=minimum_periods
            )
        )

        data[
            "VolatilityQ33"
        ] = (
            expanding.quantile(
                0.33
            )
            .shift(
                1
            )
        )

        data[
            "VolatilityQ67"
        ] = (
            expanding.quantile(
                0.67
            )
            .shift(
                1
            )
        )

        data[
            "VolatilityQ90"
        ] = (
            expanding.quantile(
                0.90
            )
            .shift(
                1
            )
        )

        # ==================================================
        # Trend strength
        #
        # Measures movement in the rolling spread mean
        # relative to current spread volatility.
        # ==================================================

        trend_lookback = max(
            6,
            signal_window // 4
        )

        data[
            "MeanChange"
        ] = (
            data[
                "RollingMean"
            ]
            -
            data[
                "RollingMean"
            ].shift(
                trend_lookback
            )
        )

        data[
            "TrendStrength"
        ] = (
            data[
                "MeanChange"
            ].abs()
            /
            safe_std
        )

        # ==================================================
        # Regime labels
        # ==================================================

        data[
            "VolatilityRegime"
        ] = data.apply(
            self._classify_volatility,
            axis=1
        )

        data[
            "StructureRegime"
        ] = data.apply(
            self._classify_structure,
            axis=1
        )

        data[
            "SpreadState"
        ] = (
            data[
                "ZScore"
            ].apply(
                self._classify_spread_state
            )
        )

        # ==================================================
        # DK1 / DK2 relative-price state
        # ==================================================

        data[
            "RelativePriceState"
        ] = np.where(
            data[
                "Spread"
            ] > 0,
            "DK1 Premium",
            np.where(
                data[
                    "Spread"
                ] < 0,
                "DK2 Premium",
                "Balanced"
            )
        )

        # ==================================================
        # Remove observations without core regime statistics
        # ==================================================

        data = (
            data.dropna(
                subset=[
                    "RollingSpreadVolatility",
                    "RollingMean",
                    "RollingStd",
                    "ZScore"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return data

    # ==================================================
    # Volatility classification
    # ==================================================

    def _classify_volatility(
        self,
        row
    ):

        volatility = (
            row[
                "RollingSpreadVolatility"
            ]
        )

        q33 = (
            row[
                "VolatilityQ33"
            ]
        )

        q67 = (
            row[
                "VolatilityQ67"
            ]
        )

        q90 = (
            row[
                "VolatilityQ90"
            ]
        )

        if (
            pd.isna(
                volatility
            )
            or
            pd.isna(
                q33
            )
            or
            pd.isna(
                q67
            )
            or
            pd.isna(
                q90
            )
        ):

            return "Unknown"

        if volatility <= q33:

            return "Low"

        if volatility <= q67:

            return "Normal"

        if volatility <= q90:

            return "High"

        return "Extreme"

    # ==================================================
    # Structure classification
    # ==================================================

    def _classify_structure(
        self,
        row
    ):

        trend_strength = (
            row[
                "TrendStrength"
            ]
        )

        z_score = abs(
            row[
                "ZScore"
            ]
        )

        if pd.isna(
            trend_strength
        ):

            return "Unknown"

        if trend_strength >= 0.75:

            return "Trending"

        if (
            trend_strength <= 0.30
            and
            z_score <= 2.5
        ):

            return "Mean-Reverting"

        return "Mixed"

    # ==================================================
    # Spread-state classification
    # ==================================================

    def _classify_spread_state(
        self,
        z_score
    ):

        if pd.isna(
            z_score
        ):

            return "Unknown"

        absolute_z = abs(
            z_score
        )

        if absolute_z < 1:

            return "Normal"

        if absolute_z < 2:

            return "Elevated"

        if absolute_z < 3:

            return "Dislocated"

        return "Extreme"

    # ==================================================
    # Current regime
    # ==================================================

    def _get_current_regime(
        self,
        regimes
    ):

        valid = (
            regimes[
                (
                    regimes[
                        "VolatilityRegime"
                    ] != "Unknown"
                )
                &
                (
                    regimes[
                        "StructureRegime"
                    ] != "Unknown"
                )
                &
                (
                    regimes[
                        "SpreadState"
                    ] != "Unknown"
                )
            ]
        )

        if valid.empty:

            current = (
                regimes.iloc[
                    -1
                ]
            )

        else:

            current = (
                valid.iloc[
                    -1
                ]
            )

        return {
            "time":
                current[
                    "TimeDK"
                ],

            "volatility":
                current[
                    "VolatilityRegime"
                ],

            "structure":
                current[
                    "StructureRegime"
                ],

            "spread_state":
                current[
                    "SpreadState"
                ],

            "relative_price_state":
                current[
                    "RelativePriceState"
                ],

            "spread":
                float(
                    current[
                        "Spread"
                    ]
                ),

            "z_score":
                float(
                    current[
                        "ZScore"
                    ]
                ),

            "rolling_volatility":
                float(
                    current[
                        "RollingSpreadVolatility"
                    ]
                ),

            "trend_strength":
                (
                    float(
                        current[
                            "TrendStrength"
                        ]
                    )
                    if not pd.isna(
                        current[
                            "TrendStrength"
                        ]
                    )
                    else 0
                )
        }

    # ==================================================
    # Match trades with exact entry-time regime
    # ==================================================

    def _match_trades_to_regimes(
        self,
        simulation_data,
        regimes
    ):

        trades = (
            simulation_data.copy()
        )

        if (
            "EntryTime"
            not in trades.columns
        ):

            return pd.DataFrame()

        trades[
            "EntryTime"
        ] = pd.to_datetime(
            trades[
                "EntryTime"
            ]
        )

        regime_columns = [
            "TimeDK",
            "VolatilityRegime",
            "StructureRegime",
            "SpreadState",
            "RelativePriceState",
            "RollingSpreadVolatility",
            "TrendStrength",
            "ZScore"
        ]

        regime_lookup = (
            regimes[
                regime_columns
            ]
            .copy()
        )

        regime_lookup[
            "TimeDK"
        ] = pd.to_datetime(
            regime_lookup[
                "TimeDK"
            ]
        )

        # --------------------------------------------------
        # One regime observation per timestamp.
        # --------------------------------------------------

        regime_lookup = (
            regime_lookup
            .sort_values(
                "TimeDK"
            )
            .drop_duplicates(
                subset=[
                    "TimeDK"
                ],
                keep="last"
            )
            .reset_index(
                drop=True
            )
        )

        trades = (
            trades.sort_values(
                "EntryTime"
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------
        # Exact matching is deliberate.
        #
        # A trade should only receive the regime that was
        # calculated for its actual entry timestamp.
        #
        # We do not silently carry a stale earlier regime
        # forward when the entry timestamp is missing.
        # --------------------------------------------------

        matched = (
            trades.merge(
                regime_lookup,
                how="left",
                left_on="EntryTime",
                right_on="TimeDK"
            )
        )

        return (
            matched.reset_index(
                drop=True
            )
        )

    # ==================================================
    # Attribution
    # ==================================================

    def _calculate_attribution(
        self,
        trade_regimes,
        regime_column
    ):

        if trade_regimes.empty:

            return pd.DataFrame()

        if (
            regime_column
            not in trade_regimes.columns
        ):

            return pd.DataFrame()

        valid = (
            trade_regimes[
                trade_regimes[
                    regime_column
                ].notna()
            ]
            .copy()
        )

        valid = (
            valid[
                valid[
                    regime_column
                ] != "Unknown"
            ]
        )

        if valid.empty:

            return pd.DataFrame()

        total_pnl = (
            valid[
                "NetPnLEUR"
            ].sum()
        )

        rows = []

        for (
            regime,
            group
        ) in valid.groupby(
            regime_column
        ):

            trades = len(
                group
            )

            net_pnl = (
                group[
                    "NetPnLEUR"
                ].sum()
            )

            average_trade = (
                group[
                    "NetPnLEUR"
                ].mean()
            )

            wins = (
                group[
                    group[
                        "NetPnLEUR"
                    ] > 0
                ]
            )

            win_rate = (
                len(
                    wins
                )
                / trades
            ) * 100

            average_return = (
                group[
                    "TradeReturnPercent"
                ].mean()
            )

            average_position = (
                group[
                    "PositionSizeMWh"
                ].mean()
            )

            if total_pnl != 0:

                contribution = (
                    net_pnl
                    / total_pnl
                ) * 100

            else:

                contribution = 0

            rows.append(
                {
                    "Regime":
                        regime,

                    "Trades":
                        trades,

                    "NetPnL":
                        net_pnl,

                    "AverageTrade":
                        average_trade,

                    "WinRate":
                        win_rate,

                    "AverageReturn":
                        average_return,

                    "AveragePosition":
                        average_position,

                    "ContributionPercent":
                        contribution
                }
            )

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                by="AverageTrade",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    # ==================================================
    # Best / worst regime
    # ==================================================

    def _best_regime(
        self,
        attribution
    ):

        if attribution.empty:

            return "N/A"

        eligible = (
            attribution[
                attribution[
                    "Trades"
                ] >= 2
            ]
        )

        if eligible.empty:

            eligible = (
                attribution
            )

        best = (
            eligible.sort_values(
                by=[
                    "AverageTrade",
                    "NetPnL"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .iloc[
                0
            ]
        )

        return (
            best[
                "Regime"
            ]
        )

    def _worst_regime(
        self,
        attribution
    ):

        if attribution.empty:

            return "N/A"

        eligible = (
            attribution[
                attribution[
                    "Trades"
                ] >= 2
            ]
        )

        if eligible.empty:

            eligible = (
                attribution
            )

        worst = (
            eligible.sort_values(
                by=[
                    "AverageTrade",
                    "NetPnL"
                ],
                ascending=[
                    True,
                    True
                ]
            )
            .iloc[
                0
            ]
        )

        return (
            worst[
                "Regime"
            ]
        )

    # ==================================================
    # Research summary
    # ==================================================

    def _create_summary(
        self,
        volatility_attribution,
        structure_attribution
    ):

        best_volatility = (
            self._best_regime(
                volatility_attribution
            )
        )

        best_structure = (
            self._best_regime(
                structure_attribution
            )
        )

        worst_volatility = (
            self._worst_regime(
                volatility_attribution
            )
        )

        return (
            f"Historically, the strategy produced its "
            f"strongest average trade performance in "
            f"{best_volatility} volatility and "
            f"{best_structure} market structure regimes. "
            f"The weakest volatility regime was "
            f"{worst_volatility}."
        )

    # ==================================================
    # Empty result
    # ==================================================

    def _empty_result(
        self,
        message
    ):

        return {
            "success":
                False,

            "message":
                message,

            "regime_data":
                pd.DataFrame(),

            "trade_regimes":
                pd.DataFrame(),

            "volatility_attribution":
                pd.DataFrame(),

            "structure_attribution":
                pd.DataFrame(),

            "spread_state_attribution":
                pd.DataFrame(),

            "current_regime":
                {},

            "best_volatility_regime":
                "N/A",

            "best_structure_regime":
                "N/A",

            "worst_volatility_regime":
                "N/A",

            "regime_summary":
                ""
        }