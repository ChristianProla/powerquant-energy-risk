import numpy as np
import pandas as pd
import pytest

from src.strategy.portfolio_simulation import (
    CapitalPortfolioSimulator
)


class FakeSpreadAnalyzer:

    def calculate_signals(
        self,
        spread_dataframe,
        window,
        threshold
    ):

        return (
            spread_dataframe.copy()
        )


class FakeBacktester:

    def __init__(
        self,
        trades
    ):

        self.trades = (
            trades.copy()
        )

    def run_backtest(
        self,
        signal_dataframe,
        entry_threshold,
        exit_threshold,
        transaction_cost_per_leg
    ):

        return {
            "success":
                True,

            "message":
                "Fake backtest completed.",

            "trades":
                self.trades.copy(),

            "backtest_data":
                signal_dataframe.copy(),

            "statistics":
                {}
        }


def create_market_data():

    times = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="h"
    )

    return pd.DataFrame(
        {
            "TimeDK":
                times,

            "Spread":
                [
                    0,
                    5,
                    3,
                    -5,
                    -2,
                    0
                ],

            "ZScore":
                [
                    0,
                    2.5,
                    1,
                    -2.5,
                    -1,
                    0
                ],

            "RollingStd":
                [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10
                ],

            "DK1Price":
                [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100
                ],

            "DK2Price":
                [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100
                ]
        }
    )


def create_trade(
    entry_time,
    exit_time,
    gross_pnl=10.0,
    transaction_cost=1.0
):

    return {
        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "Direction":
            "Long DK1 / Short DK2",

        "EntrySpread":
            -5.0,

        "ExitSpread":
            5.0,

        "EntryZScore":
            -2.5,

        "ExitZScore":
            0.0,

        "GrossPnL":
            gross_pnl,

        "TransactionCost":
            transaction_cost,

        "NetPnL":
            (
                gross_pnl
                - transaction_cost
            )
    }


def create_simulator(
    trades
):

    return CapitalPortfolioSimulator(
        FakeSpreadAnalyzer(),
        FakeBacktester(
            pd.DataFrame(
                trades
            )
        )
    )


def test_position_size_never_exceeds_maximum():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        risk_per_trade_percent=1.0,
        max_position_mwh=10.0,
        max_gross_exposure_percent=50.0
    )

    assert result[
        "success"
    ]

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert trade[
        "PositionSizeMWh"
    ] <= 10.0


def test_risk_based_size_respects_risk_budget():

    market = create_market_data()

    market[
        "RollingStd"
    ] = 100.0

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        risk_per_trade_percent=1.0,
        max_position_mwh=1000.0,
        max_gross_exposure_percent=1000.0
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert (
        trade[
            "EstimatedRiskEUR"
        ]
        <=
        trade[
            "RiskBudgetEUR"
        ]
        + 1e-9
    )

    assert (
        trade[
            "RiskUtilizationPercent"
        ]
        <= 100.0
    )


def test_exposure_limit_can_bind_position_size():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=1000,
        risk_per_trade_percent=100,
        max_position_mwh=100,
        max_gross_exposure_percent=10
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    # €100 allowed gross exposure /
    # €200 gross price per MWh = 0.5 MWh

    assert trade[
        "PositionSizeMWh"
    ] == pytest.approx(
        0.5
    )

    assert trade[
        "BindingConstraint"
    ] == "Exposure Limit"


def test_net_pnl_equals_gross_pnl_minus_costs():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        max_position_mwh=10
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert trade[
        "NetPnLEUR"
    ] == pytest.approx(
        trade[
            "GrossPnLEUR"
        ]
        -
        trade[
            "TransactionCostEUR"
        ]
    )


def test_capital_compounds_between_trades():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        ),
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                3
            ],
            market[
                "TimeDK"
            ].iloc[
                4
            ],
            gross_pnl=5.0,
            transaction_cost=1.0
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        max_position_mwh=1.0,
        max_gross_exposure_percent=100
    )

    simulation = result[
        "simulation_data"
    ]

    assert len(
        simulation
    ) == 2

    assert simulation.iloc[
        1
    ][
        "CapitalBefore"
    ] == pytest.approx(
        simulation.iloc[
            0
        ][
            "CapitalAfter"
        ]
    )


def test_trades_are_processed_chronologically():

    market = create_market_data()

    later_trade = create_trade(
        market[
            "TimeDK"
        ].iloc[
            3
        ],
        market[
            "TimeDK"
        ].iloc[
            4
        ]
    )

    earlier_trade = create_trade(
        market[
            "TimeDK"
        ].iloc[
            1
        ],
        market[
            "TimeDK"
        ].iloc[
            2
        ]
    )

    simulator = (
        create_simulator(
            [
                later_trade,
                earlier_trade
            ]
        )
    )

    result = simulator.run_simulation(
        market,
        window=5
    )

    simulation = result[
        "simulation_data"
    ]

    assert simulation[
        "EntryTime"
    ].is_monotonic_increasing


def test_negative_starting_capital_is_rejected():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=-1000
    )

    assert not result[
        "success"
    ]


def test_invalid_risk_percentage_is_rejected():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        risk_per_trade_percent=150
    )

    assert not result[
        "success"
    ]


def test_risk_utilization_is_true_budget_utilization():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        risk_per_trade_percent=1.0,
        max_position_mwh=10,
        max_gross_exposure_percent=50
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    expected_utilization = (
        trade[
            "EstimatedRiskEUR"
        ]
        /
        trade[
            "RiskBudgetEUR"
        ]
    ) * 100

    assert trade[
        "RiskUtilizationPercent"
    ] == pytest.approx(
        expected_utilization
    )


def test_portfolio_total_pnl_matches_ending_capital():

    market = create_market_data()

    trades = [
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                1
            ],
            market[
                "TimeDK"
            ].iloc[
                2
            ]
        ),
        create_trade(
            market[
                "TimeDK"
            ].iloc[
                3
            ],
            market[
                "TimeDK"
            ].iloc[
                4
            ]
        )
    ]

    simulator = (
        create_simulator(
            trades
        )
    )

    result = simulator.run_simulation(
        market,
        window=5,
        starting_capital=100000,
        max_position_mwh=1
    )

    statistics = result[
        "statistics"
    ]

    assert statistics[
        "total_pnl"
    ] == pytest.approx(
        statistics[
            "ending_capital"
        ]
        -
        statistics[
            "starting_capital"
        ]
    )