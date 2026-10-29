import pandas as pd
import pytest

from src.strategy.backtesting import (
    SpreadBacktester
)


def create_signal_data(
    spreads,
    z_scores
):

    times = pd.date_range(
        "2026-01-01",
        periods=len(
            spreads
        ),
        freq="h"
    )

    return pd.DataFrame(
        {
            "TimeDK":
                times,

            "Spread":
                spreads,

            "ZScore":
                z_scores
        }
    )


def test_long_spread_trade_makes_money_when_spread_rises():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            -10,
            -5,
            0
        ],
        z_scores=[
            0,
            -2.5,
            -1.0,
            0
        ]
    )

    result = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0
    )

    assert result["success"]

    trades = result[
        "trades"
    ]

    assert len(trades) >= 1

    first_trade = (
        trades.iloc[0]
    )

    assert first_trade[
        "Direction"
    ] == (
        "Long DK1 / Short DK2"
    )

    assert first_trade[
        "GrossPnL"
    ] > 0


def test_short_spread_trade_makes_money_when_spread_falls():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            10,
            5,
            0
        ],
        z_scores=[
            0,
            2.5,
            1.0,
            0
        ]
    )

    result = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0
    )

    assert result["success"]

    trades = result[
        "trades"
    ]

    assert len(trades) >= 1

    first_trade = (
        trades.iloc[0]
    )

    assert first_trade[
        "Direction"
    ] == (
        "Short DK1 / Long DK2"
    )

    assert first_trade[
        "GrossPnL"
    ] > 0


def test_transaction_cost_reduces_net_pnl():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            -10,
            -5,
            0
        ],
        z_scores=[
            0,
            -2.5,
            -1,
            0
        ]
    )

    no_cost = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0
    )

    with_cost = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0.25
    )

    assert (
        with_cost[
            "statistics"
        ][
            "total_net_pnl"
        ]
        <
        no_cost[
            "statistics"
        ][
            "total_net_pnl"
        ]
    )


def test_round_trip_cost_uses_two_legs_on_entry_and_exit():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            -10,
            -5,
            0
        ],
        z_scores=[
            0,
            -2.5,
            -1,
            0
        ]
    )

    result = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0.25
    )

    trades = result[
        "trades"
    ]

    assert len(trades) >= 1

    trade = trades.iloc[0]

    expected_round_trip_cost = (
        0.25
        * 4
    )

    assert trade[
        "TransactionCost"
    ] == pytest.approx(
        expected_round_trip_cost
    )


def test_no_position_before_entry_signal():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            1,
            2,
            3
        ],
        z_scores=[
            0,
            0.2,
            0.4,
            0.6
        ]
    )

    result = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0
    )

    backtest_data = result[
        "backtest_data"
    ]

    assert (
        backtest_data[
            "Position"
        ]
        == 0
    ).all()


def test_backtest_is_chronological():

    backtester = SpreadBacktester()

    data = create_signal_data(
        spreads=[
            0,
            -10,
            -5,
            0,
            2
        ],
        z_scores=[
            0,
            -2.5,
            -1,
            0,
            0.5
        ]
    )

    result = backtester.run_backtest(
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        transaction_cost_per_leg=0
    )

    backtest_data = result[
        "backtest_data"
    ]

    assert backtest_data[
        "TimeDK"
    ].is_monotonic_increasing