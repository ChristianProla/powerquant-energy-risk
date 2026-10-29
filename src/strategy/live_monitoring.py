import numpy as np
import pandas as pd


class LiveStrategyMonitor:

    def __init__(
        self,
        spread_analyzer
    ):

        self.spread_analyzer = (
            spread_analyzer
        )

    # ==================================================
    # Current strategy state
    # ==================================================

    def analyse_current_state(
        self,
        spread_dataframe,
        window=48,
        entry_threshold=2.0,
        exit_threshold=0.5,
        transaction_cost_per_leg=0.25,
        capital=100000,
        risk_per_trade_percent=1.0,
        max_position_mwh=10.0,
        max_gross_exposure_percent=50.0
    ):

        if spread_dataframe.empty:

            return self._empty_result(
                "No spread data available."
            )

        window = max(
            5,
            int(
                window
            )
        )

        entry_threshold = max(
            0.5,
            float(
                entry_threshold
            )
        )

        exit_threshold = max(
            0.0,
            float(
                exit_threshold
            )
        )

        transaction_cost_per_leg = max(
            0.0,
            float(
                transaction_cost_per_leg
            )
        )

        capital = max(
            100.0,
            float(
                capital
            )
        )

        risk_per_trade_percent = max(
            0.1,
            float(
                risk_per_trade_percent
            )
        )

        max_position_mwh = max(
            0.1,
            float(
                max_position_mwh
            )
        )

        max_gross_exposure_percent = max(
            1.0,
            float(
                max_gross_exposure_percent
            )
        )

        # ==================================================
        # Calculate current spread signal
        # ==================================================

        signal_data = (
            self.spread_analyzer
            .calculate_signals(
                spread_dataframe,
                window=window,
                threshold=entry_threshold
            )
        )

        if signal_data.empty:

            return self._empty_result(
                "No signal data could be generated."
            )

        valid_data = (
            signal_data.dropna(
                subset=[
                    "Spread",
                    "RollingMean",
                    "RollingStd",
                    "ZScore"
                ]
            )
        )

        if valid_data.empty:

            return self._empty_result(
                (
                    "Not enough data for the selected "
                    "rolling window."
                )
            )

        current = (
            valid_data.iloc[
                -1
            ]
        )

        current_time = (
            current[
                "TimeDK"
            ]
        )

        dk1_price = float(
            current[
                "DK1Price"
            ]
        )

        dk2_price = float(
            current[
                "DK2Price"
            ]
        )

        spread = float(
            current[
                "Spread"
            ]
        )

        rolling_mean = float(
            current[
                "RollingMean"
            ]
        )

        rolling_std = float(
            current[
                "RollingStd"
            ]
        )

        z_score = float(
            current[
                "ZScore"
            ]
        )

        # ==================================================
        # Current signal
        # ==================================================

        if z_score >= entry_threshold:

            signal = (
                "ENTRY SIGNAL"
            )

            direction = (
                "Short DK1 / Long DK2"
            )

            position_direction = -1

            interpretation = (
                "DK1 is expensive relative to DK2."
            )

        elif z_score <= -entry_threshold:

            signal = (
                "ENTRY SIGNAL"
            )

            direction = (
                "Long DK1 / Short DK2"
            )

            position_direction = 1

            interpretation = (
                "DK1 is cheap relative to DK2."
            )

        elif abs(
            z_score
        ) >= (
            entry_threshold
            * 0.75
        ):

            signal = (
                "WATCH"
            )

            direction = (
                "No Position Yet"
            )

            position_direction = 0

            interpretation = (
                "Spread is approaching the entry threshold."
            )

        else:

            signal = (
                "NEUTRAL"
            )

            direction = (
                "No Position"
            )

            position_direction = 0

            interpretation = (
                "Spread is inside the normal range."
            )

        # ==================================================
        # Signal strength
        # ==================================================

        signal_strength_percent = min(
            (
                abs(
                    z_score
                )
                / entry_threshold
            ) * 100,
            200
        )

        distance_to_entry_z = max(
            0,
            (
                entry_threshold
                - abs(
                    z_score
                )
            )
        )

        # ==================================================
        # Mean-reversion information
        # ==================================================

        spread_deviation = (
            spread
            - rolling_mean
        )

        mean_reversion_move = abs(
            spread_deviation
        )

        # ==================================================
        # Risk budget
        # ==================================================

        risk_budget = (
            capital
            * (
                risk_per_trade_percent
                / 100
            )
        )

        spread_risk = (
            2
            * abs(
                rolling_std
            )
        )

        round_trip_cost_per_mwh = (
            transaction_cost_per_leg
            * 4
        )

        estimated_risk_per_mwh = max(
            1.0,
            (
                spread_risk
                + round_trip_cost_per_mwh
            )
        )

        # ==================================================
        # Risk-based size
        # ==================================================

        risk_based_size = (
            risk_budget
            / estimated_risk_per_mwh
        )

        # ==================================================
        # Exposure-based size
        # ==================================================

        gross_price_per_mwh = (
            abs(
                dk1_price
            )
            + abs(
                dk2_price
            )
        )

        maximum_gross_exposure = (
            capital
            * (
                max_gross_exposure_percent
                / 100
            )
        )

        if gross_price_per_mwh > 0:

            exposure_based_size = (
                maximum_gross_exposure
                / gross_price_per_mwh
            )

        else:

            exposure_based_size = (
                max_position_mwh
            )

        suggested_position_size = min(
            risk_based_size,
            exposure_based_size,
            max_position_mwh
        )

        suggested_position_size = max(
            0,
            suggested_position_size
        )

        # --------------------------------------------------
        # No position is suggested until there is
        # an actual entry signal.
        # --------------------------------------------------

        if position_direction == 0:

            active_position_size = 0

        else:

            active_position_size = (
                suggested_position_size
            )

        # ==================================================
        # Exposure
        # ==================================================

        gross_exposure = (
            gross_price_per_mwh
            * active_position_size
        )

        if capital > 0:

            leverage = (
                gross_exposure
                / capital
            )

        else:

            leverage = 0

        # ==================================================
        # Estimated mean-reversion opportunity
        #
        # This is NOT a forecast or guaranteed PnL.
        # ==================================================

        potential_gross_move_eur = (
            mean_reversion_move
            * active_position_size
        )

        # ==================================================
        # Binding constraint
        # ==================================================

        size_constraints = {
            "Risk Budget":
                risk_based_size,

            "Max Position":
                max_position_mwh,

            "Exposure Limit":
                exposure_based_size
        }

        binding_constraint = min(
            size_constraints,
            key=size_constraints.get
        )

        # ==================================================
        # Short-term spread momentum
        # ==================================================

        recent_data = (
            valid_data.tail(
                7
            )
        )

        if len(
            recent_data
        ) >= 2:

            spread_momentum = (
                spread
                - float(
                    recent_data[
                        "Spread"
                    ].iloc[
                        0
                    ]
                )
            )

            z_momentum = (
                z_score
                - float(
                    recent_data[
                        "ZScore"
                    ].iloc[
                        0
                    ]
                )
            )

        else:

            spread_momentum = 0
            z_momentum = 0

        # ==================================================
        # Market state
        # ==================================================

        if abs(
            z_score
        ) >= 3:

            market_state = (
                "Extreme Dislocation"
            )

        elif abs(
            z_score
        ) >= entry_threshold:

            market_state = (
                "Statistical Dislocation"
            )

        elif abs(
            z_score
        ) >= 1:

            market_state = (
                "Elevated Spread"
            )

        else:

            market_state = (
                "Normal Spread"
            )

        # ==================================================
        # Recent chart data
        # ==================================================

        chart_data = (
            valid_data.tail(
                168
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        return {
            "success":
                True,

            "message":
                "Current strategy state calculated.",

            "current_time":
                current_time,

            "dk1_price":
                dk1_price,

            "dk2_price":
                dk2_price,

            "spread":
                spread,

            "rolling_mean":
                rolling_mean,

            "rolling_std":
                rolling_std,

            "z_score":
                z_score,

            "spread_deviation":
                spread_deviation,

            "mean_reversion_move":
                mean_reversion_move,

            "signal":
                signal,

            "direction":
                direction,

            "interpretation":
                interpretation,

            "market_state":
                market_state,

            "signal_strength_percent":
                signal_strength_percent,

            "distance_to_entry_z":
                distance_to_entry_z,

            "spread_momentum":
                spread_momentum,

            "z_momentum":
                z_momentum,

            "capital":
                capital,

            "risk_budget":
                risk_budget,

            "estimated_risk_per_mwh":
                estimated_risk_per_mwh,

            "risk_based_size":
                risk_based_size,

            "exposure_based_size":
                exposure_based_size,

            "suggested_position_size":
                suggested_position_size,

            "active_position_size":
                active_position_size,

            "gross_exposure":
                gross_exposure,

            "leverage":
                leverage,

            "potential_gross_move_eur":
                potential_gross_move_eur,

            "binding_constraint":
                binding_constraint,

            "window":
                window,

            "entry_threshold":
                entry_threshold,

            "exit_threshold":
                exit_threshold,

            "transaction_cost_per_leg":
                transaction_cost_per_leg,

            "chart_data":
                chart_data
        }

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

            "chart_data":
                pd.DataFrame()
        }