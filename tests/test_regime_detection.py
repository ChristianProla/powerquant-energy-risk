import numpy as np
import pandas as pd

from src.strategy.regime_detection import (
    MarketRegimeDetector
)


def create_spread_data(
    observations=500
):

    rng = np.random.default_rng(
        42
    )

    times = pd.date_range(
        "2026-01-01",
        periods=observations,
        freq="h"
    )

    spread = np.zeros(
        observations
    )

    for index in range(
        1,
        observations
    ):

        spread[
            index
        ] = (
            0.85
            * spread[
                index - 1
            ]
            +
            rng.normal(
                0,
                5
            )
        )

    return pd.DataFrame(
        {
            "TimeDK":
                times,

            "DK1Price":
                100
                + spread,

            "DK2Price":
                np.full(
                    observations,
                    100.0
                ),

            "Spread":
                spread
        }
    )


def test_regime_data_is_chronological():

    detector = (
        MarketRegimeDetector()
    )

    data = (
        create_spread_data()
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(
            drop=True
        )
    )

    regimes = (
        detector
        ._build_regime_dataframe(
            data,
            signal_window=48
        )
    )

    assert not regimes.empty

    assert regimes[
        "TimeDK"
    ].is_monotonic_increasing


def test_regime_thresholds_use_historical_information():

    detector = (
        MarketRegimeDetector()
    )

    data = create_spread_data(
        observations=500
    )

    original = (
        detector
        ._build_regime_dataframe(
            data,
            signal_window=48
        )
    )

    modified = (
        data.copy()
    )

    # --------------------------------------------------
    # Change only observations far in the future.
    # Historical regime classifications before that
    # point must remain unchanged.
    # --------------------------------------------------

    modification_start = 400

    modified.loc[
        modification_start:,
        "Spread"
    ] = (
        modified.loc[
            modification_start:,
            "Spread"
        ]
        * 20
    )

    modified.loc[
        modification_start:,
        "DK1Price"
    ] = (
        100
        +
        modified.loc[
            modification_start:,
            "Spread"
        ]
    )

    changed = (
        detector
        ._build_regime_dataframe(
            modified,
            signal_window=48
        )
    )

    comparison_time = (
        data[
            "TimeDK"
        ].iloc[
            350
        ]
    )

    original_row = (
        original[
            original[
                "TimeDK"
            ] == comparison_time
        ]
        .iloc[
            0
        ]
    )

    changed_row = (
        changed[
            changed[
                "TimeDK"
            ] == comparison_time
        ]
        .iloc[
            0
        ]
    )

    assert (
        original_row[
            "VolatilityRegime"
        ]
        ==
        changed_row[
            "VolatilityRegime"
        ]
    )

    assert (
        original_row[
            "StructureRegime"
        ]
        ==
        changed_row[
            "StructureRegime"
        ]
    )

    assert (
        original_row[
            "SpreadState"
        ]
        ==
        changed_row[
            "SpreadState"
        ]
    )


def test_trade_regime_matching_requires_exact_timestamp():

    detector = (
        MarketRegimeDetector()
    )

    regimes = pd.DataFrame(
        {
            "TimeDK":
                pd.to_datetime(
                    [
                        "2026-01-01 00:00",
                        "2026-01-01 02:00"
                    ]
                ),

            "VolatilityRegime":
                [
                    "Low",
                    "High"
                ],

            "StructureRegime":
                [
                    "Mean-Reverting",
                    "Trending"
                ],

            "SpreadState":
                [
                    "Normal",
                    "Dislocated"
                ],

            "RelativePriceState":
                [
                    "Balanced",
                    "DK1 Premium"
                ],

            "RollingSpreadVolatility":
                [
                    1.0,
                    5.0
                ],

            "TrendStrength":
                [
                    0.1,
                    1.0
                ],

            "ZScore":
                [
                    0.0,
                    2.5
                ]
        }
    )

    trades = pd.DataFrame(
        {
            "EntryTime":
                pd.to_datetime(
                    [
                        "2026-01-01 01:00"
                    ]
                ),

            "NetPnLEUR":
                [
                    10.0
                ],

            "TradeReturnPercent":
                [
                    0.1
                ],

            "PositionSizeMWh":
                [
                    1.0
                ]
        }
    )

    matched = (
        detector
        ._match_trades_to_regimes(
            trades,
            regimes
        )
    )

    assert pd.isna(
        matched.iloc[
            0
        ][
            "VolatilityRegime"
        ]
    )


def test_trade_regime_matching_uses_exact_entry_regime():

    detector = (
        MarketRegimeDetector()
    )

    entry_time = pd.Timestamp(
        "2026-01-01 02:00"
    )

    regimes = pd.DataFrame(
        {
            "TimeDK":
                [
                    entry_time
                ],

            "VolatilityRegime":
                [
                    "High"
                ],

            "StructureRegime":
                [
                    "Mixed"
                ],

            "SpreadState":
                [
                    "Dislocated"
                ],

            "RelativePriceState":
                [
                    "DK1 Premium"
                ],

            "RollingSpreadVolatility":
                [
                    5.0
                ],

            "TrendStrength":
                [
                    0.5
                ],

            "ZScore":
                [
                    2.2
                ]
        }
    )

    trades = pd.DataFrame(
        {
            "EntryTime":
                [
                    entry_time
                ],

            "NetPnLEUR":
                [
                    10.0
                ],

            "TradeReturnPercent":
                [
                    0.1
                ],

            "PositionSizeMWh":
                [
                    1.0
                ]
        }
    )

    matched = (
        detector
        ._match_trades_to_regimes(
            trades,
            regimes
        )
    )

    assert (
        matched.iloc[
            0
        ][
            "VolatilityRegime"
        ]
        ==
        "High"
    )


def test_regime_analysis_without_trades_still_succeeds():

    detector = (
        MarketRegimeDetector()
    )

    data = create_spread_data()

    result = detector.analyze(
        data,
        pd.DataFrame(),
        signal_window=48
    )

    assert result[
        "success"
    ]

    assert not result[
        "regime_data"
    ].empty


def test_invalid_signal_window_is_rejected():

    detector = (
        MarketRegimeDetector()
    )

    data = create_spread_data()

    result = detector.analyze(
        data,
        pd.DataFrame(),
        signal_window=5
    )

    assert not result[
        "success"
    ]


def test_current_regime_contains_expected_fields():

    detector = (
        MarketRegimeDetector()
    )

    data = create_spread_data()

    result = detector.analyze(
        data,
        pd.DataFrame(),
        signal_window=48
    )

    assert result[
        "success"
    ]

    current = result[
        "current_regime"
    ]

    expected_fields = {
        "time",
        "volatility",
        "structure",
        "spread_state",
        "relative_price_state",
        "spread",
        "z_score",
        "rolling_volatility",
        "trend_strength"
    }

    assert expected_fields.issubset(
        current.keys()
    )