import pandas as pd
import pytest

from src.strategy.regime_aware_portfolio import (
    RegimeAwarePortfolioSimulator
)


class FakeBaseSimulator:

    def __init__(
        self,
        trades
    ):

        self.trades = (
            trades.copy()
        )

    def run_simulation(
        self,
        *args,
        **kwargs
    ):

        return {
            "success":
                True,

            "message":
                "Fake static simulation completed.",

            "simulation_data":
                self.trades.copy(),

            "statistics":
                {
                    "starting_capital":
                        100000.0,

                    "ending_capital":
                        100000.0,

                    "total_return_percent":
                        0.0,

                    "maximum_drawdown_percent":
                        0.0
                }
        }


class FakeRegimeDetector:

    def __init__(
        self,
        regime_data
    ):

        self.regime_data = (
            regime_data.copy()
        )

    def analyze(
        self,
        *args,
        **kwargs
    ):

        return {
            "success":
                True,

            "message":
                "Fake regime analysis completed.",

            "regime_data":
                self.regime_data.copy()
        }


def create_trade(
    entry_time,
    exit_time,
    net_pnl_per_mwh,
    original_size=1.0
):

    gross_pnl_per_mwh = (
        net_pnl_per_mwh
        + 1.0
    )

    return {
        "EntryTime":
            pd.Timestamp(
                entry_time
            ),

        "ExitTime":
            pd.Timestamp(
                exit_time
            ),

        "Direction":
            "Long DK1 / Short DK2",

        "PositionSizeMWh":
            original_size,

        "GrossPnLEUR":
            (
                gross_pnl_per_mwh
                * original_size
            ),

        "TransactionCostEUR":
            (
                1.0
                * original_size
            ),

        "NetPnLEUR":
            (
                net_pnl_per_mwh
                * original_size
            ),

        "GrossExposureEUR":
            (
                200.0
                * original_size
            ),

        "EstimatedRiskPerMWh":
            20.0
    }


def create_regime_row(
    time,
    volatility="Normal",
    structure="Mean-Reverting"
):

    return {
        "TimeDK":
            pd.Timestamp(
                time
            ),

        "VolatilityRegime":
            volatility,

        "StructureRegime":
            structure,

        "SpreadState":
            "Dislocated",

        "RelativePriceState":
            "DK1 Premium",

        "RollingSpreadVolatility":
            5.0,

        "TrendStrength":
            0.2,

        "ZScore":
            2.5
    }


def create_simulator(
    trades,
    regimes
):

    simulator = (
        RegimeAwarePortfolioSimulator(
            None,
            None
        )
    )

    simulator.base_simulator = (
        FakeBaseSimulator(
            pd.DataFrame(
                trades
            )
        )
    )

    simulator.regime_detector = (
        FakeRegimeDetector(
            pd.DataFrame(
                regimes
            )
        )
    )

    return simulator


def dummy_spread_data():

    return pd.DataFrame(
        {
            "TimeDK":
                pd.to_datetime(
                    [
                        "2026-01-01 00:00",
                        "2026-01-01 01:00"
                    ]
                ),

            "DK1Price":
                [
                    100,
                    100
                ],

            "DK2Price":
                [
                    100,
                    100
                ],

            "Spread":
                [
                    0,
                    0
                ]
        }
    )


def test_regime_aware_multiplier_never_exceeds_one():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    assert result[
        "success"
    ]

    simulation = result[
        "simulation_data"
    ]

    assert (
        simulation[
            "FinalMultiplier"
        ] <= 1.0
    ).all()


def test_first_signal_has_no_adaptive_history():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    first = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert first[
        "PreviousRegimeSignals"
    ] == 0

    assert first[
        "AdaptiveMultiplier"
    ] == pytest.approx(
        1.0
    )


def test_current_signal_does_not_influence_its_own_multiplier():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            -10
        ),
        create_trade(
            "2026-01-01 03:00",
            "2026-01-01 04:00",
            -10
        ),
        create_trade(
            "2026-01-01 05:00",
            "2026-01-01 06:00",
            -10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        ),
        create_regime_row(
            "2026-01-01 03:00"
        ),
        create_regime_row(
            "2026-01-01 05:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    simulation = result[
        "simulation_data"
    ]

    assert simulation.iloc[
        0
    ][
        "AdaptiveMultiplier"
    ] == pytest.approx(
        1.0
    )

    assert simulation.iloc[
        1
    ][
        "AdaptiveMultiplier"
    ] == pytest.approx(
        1.0
    )

    assert simulation.iloc[
        2
    ][
        "AdaptiveMultiplier"
    ] == pytest.approx(
        1.0
    )

    assert simulation.iloc[
        2
    ][
        "PreviousRegimeSignals"
    ] == 2


def test_fourth_signal_can_be_skipped_after_three_losses():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            -10
        ),
        create_trade(
            "2026-01-01 03:00",
            "2026-01-01 04:00",
            -10
        ),
        create_trade(
            "2026-01-01 05:00",
            "2026-01-01 06:00",
            -10
        ),
        create_trade(
            "2026-01-01 07:00",
            "2026-01-01 08:00",
            20
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        ),
        create_regime_row(
            "2026-01-01 03:00"
        ),
        create_regime_row(
            "2026-01-01 05:00"
        ),
        create_regime_row(
            "2026-01-01 07:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    fourth = result[
        "simulation_data"
    ].iloc[
        3
    ]

    assert fourth[
        "PreviousRegimeSignals"
    ] == 3

    assert fourth[
        "AdaptiveMultiplier"
    ] == pytest.approx(
        0.0
    )

    assert fourth[
        "PositionSizeMWh"
    ] == pytest.approx(
        0.0
    )

    assert not fourth[
        "ExecutedTrade"
    ]


def test_skipped_signal_is_used_for_shadow_learning():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            -10
        ),
        create_trade(
            "2026-01-01 03:00",
            "2026-01-01 04:00",
            -10
        ),
        create_trade(
            "2026-01-01 05:00",
            "2026-01-01 06:00",
            -10
        ),
        create_trade(
            "2026-01-01 07:00",
            "2026-01-01 08:00",
            20
        ),
        create_trade(
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            20
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        ),
        create_regime_row(
            "2026-01-01 03:00"
        ),
        create_regime_row(
            "2026-01-01 05:00"
        ),
        create_regime_row(
            "2026-01-01 07:00"
        ),
        create_regime_row(
            "2026-01-01 09:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    simulation = result[
        "simulation_data"
    ]

    fourth = simulation.iloc[
        3
    ]

    fifth = simulation.iloc[
        4
    ]

    assert not fourth[
        "ExecutedTrade"
    ]

    # The skipped fourth signal is still known after
    # its hypothetical exit and therefore becomes part
    # of the historical signal set before signal five.

    assert fifth[
        "PreviousRegimeSignals"
    ] == 4


def test_skipped_signal_does_not_change_portfolio_capital():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            -10
        ),
        create_trade(
            "2026-01-01 03:00",
            "2026-01-01 04:00",
            -10
        ),
        create_trade(
            "2026-01-01 05:00",
            "2026-01-01 06:00",
            -10
        ),
        create_trade(
            "2026-01-01 07:00",
            "2026-01-01 08:00",
            100
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        ),
        create_regime_row(
            "2026-01-01 03:00"
        ),
        create_regime_row(
            "2026-01-01 05:00"
        ),
        create_regime_row(
            "2026-01-01 07:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    skipped = result[
        "simulation_data"
    ].iloc[
        3
    ]

    assert skipped[
        "PositionSizeMWh"
    ] == 0

    assert skipped[
        "CapitalAfter"
    ] == pytest.approx(
        skipped[
            "CapitalBefore"
        ]
    )


def test_missing_exact_regime_does_not_use_stale_previous_regime():

    trades = [
        create_trade(
            "2026-01-01 02:00",
            "2026-01-01 03:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00",
            volatility="Extreme",
            structure="Trending"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert trade[
        "VolatilityRegime"
    ] == "Unknown"

    assert trade[
        "StructureRegime"
    ] == "Unknown"


def test_structural_multiplier_is_conservative():

    simulator = (
        RegimeAwarePortfolioSimulator(
            None,
            None
        )
    )

    multiplier = (
        simulator
        ._get_structural_multiplier(
            "Extreme",
            "Trending"
        )
    )

    assert multiplier == pytest.approx(
        0.50
    )


def test_adjusted_risk_never_exceeds_base_risk():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data(),
        risk_per_trade_percent=1.0
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert (
        trade[
            "EffectiveRiskPercent"
        ]
        <=
        trade[
            "BaseRiskPercent"
        ]
    )

    assert (
        trade[
            "AdjustedRiskBudgetEUR"
        ]
        <=
        trade[
            "BaseRiskBudgetEUR"
        ]
    )


def test_regime_aware_risk_utilization_never_exceeds_budget():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    trade = result[
        "simulation_data"
    ].iloc[
        0
    ]

    assert (
        trade[
            "RiskUtilizationPercent"
        ]
        <= 100.0
    )


def test_methodology_explicitly_identifies_shadow_learning():

    trades = [
        create_trade(
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            10
        )
    ]

    regimes = [
        create_regime_row(
            "2026-01-01 01:00"
        )
    ]

    simulator = create_simulator(
        trades,
        regimes
    )

    result = simulator.run_simulation(
        dummy_spread_data()
    )

    assert result[
        "methodology"
    ][
        "shadow_learning"
    ] is True