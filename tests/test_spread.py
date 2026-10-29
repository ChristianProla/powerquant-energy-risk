import numpy as np
import pandas as pd

from src.analytics.spread import SpreadAnalyzer


def create_price_data(
    prices,
    start="2026-01-01 00:00:00"
):
    times = pd.date_range(
        start=start,
        periods=len(prices),
        freq="h"
    )

    return pd.DataFrame(
        {
            "TimeDK": times,
            "DayAheadPriceEUR": prices
        }
    )


def test_prepare_spread_data_calculates_correct_spread():

    analyzer = SpreadAnalyzer()

    dk1 = create_price_data(
        [100, 110, 120]
    )

    dk2 = create_price_data(
        [90, 105, 125]
    )

    result = analyzer.prepare_spread_data(
        dk1,
        dk2
    )

    expected = [
        10,
        5,
        -5
    ]

    assert result["Spread"].tolist() == expected


def test_prepare_spread_data_creates_expected_columns():

    analyzer = SpreadAnalyzer()

    dk1 = create_price_data(
        [100, 101, 102]
    )

    dk2 = create_price_data(
        [95, 96, 97]
    )

    result = analyzer.prepare_spread_data(
        dk1,
        dk2
    )

    expected_columns = {
        "TimeDK",
        "DK1Price",
        "DK2Price",
        "Spread"
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_prepare_spread_data_sorts_chronologically():

    analyzer = SpreadAnalyzer()

    dk1 = create_price_data(
        [100, 110, 120]
    ).iloc[::-1]

    dk2 = create_price_data(
        [90, 100, 110]
    ).iloc[::-1]

    result = analyzer.prepare_spread_data(
        dk1,
        dk2
    )

    assert result[
        "TimeDK"
    ].is_monotonic_increasing


def test_calculate_signals_creates_z_score():

    analyzer = SpreadAnalyzer()

    times = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="h"
    )

    spread = pd.DataFrame(
        {
            "TimeDK": times,
            "DK1Price": np.arange(
                100,
                120
            ),
            "DK2Price": np.full(
                20,
                100
            )
        }
    )

    spread[
        "Spread"
    ] = (
        spread["DK1Price"]
        - spread["DK2Price"]
    )

    result = analyzer.calculate_signals(
        spread,
        window=5,
        threshold=2.0
    )

    assert "ZScore" in result.columns
    assert "RollingMean" in result.columns
    assert "RollingStd" in result.columns


def test_positive_extreme_z_score_generates_short_spread_signal():

    analyzer = SpreadAnalyzer()

    spread_values = [
        0,
        0,
        0,
        0,
        0,
        10
    ]

    times = pd.date_range(
        "2026-01-01",
        periods=len(
            spread_values
        ),
        freq="h"
    )

    data = pd.DataFrame(
        {
            "TimeDK": times,
            "DK1Price": (
                np.array(
                    spread_values
                )
                + 100
            ),
            "DK2Price": 100,
            "Spread": spread_values
        }
    )

    result = analyzer.calculate_signals(
        data,
        window=5,
        threshold=1.5
    )

    latest = result.iloc[-1]

    assert latest[
        "ZScore"
    ] > 0

    assert latest[
        "Signal"
    ] == (
        "DK1 Expensive / DK2 Cheap"
    )

    assert latest[
        "SignalValue"
    ] == -1


def test_negative_extreme_z_score_generates_long_spread_signal():

    analyzer = SpreadAnalyzer()

    spread_values = [
        0,
        0,
        0,
        0,
        0,
        -10
    ]

    times = pd.date_range(
        "2026-01-01",
        periods=len(
            spread_values
        ),
        freq="h"
    )

    data = pd.DataFrame(
        {
            "TimeDK": times,
            "DK1Price": (
                np.array(
                    spread_values
                )
                + 100
            ),
            "DK2Price": 100,
            "Spread": spread_values
        }
    )

    result = analyzer.calculate_signals(
        data,
        window=5,
        threshold=1.5
    )

    latest = result.iloc[-1]

    assert latest[
        "ZScore"
    ] < 0

    assert latest[
        "Signal"
    ] == (
        "DK1 Cheap / DK2 Expensive"
    )

    assert latest[
        "SignalValue"
    ] == 1