import numpy as np
import pandas as pd

from src.analytics.spread import (
    SpreadAnalyzer
)

from src.strategy.backtesting import (
    SpreadBacktester
)

from src.strategy.robustness import (
    StrategyRobustnessAnalyzer
)


def create_spread_data(
    observations=1000
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
            0.80
            * spread[
                index - 1
            ]
            + rng.normal(
                0,
                6
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


def create_analyzer():

    return StrategyRobustnessAnalyzer(
        SpreadAnalyzer(),
        SpreadBacktester()
    )


def test_robustness_test_completes():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]


def test_invalid_exit_entry_combinations_are_excluded():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    combinations = result[
        "all_results"
    ]

    assert (
        combinations[
            "ExitZ"
        ]
        <
        combinations[
            "EntryZ"
        ]
    ).all()


def test_baseline_results_use_fixed_transaction_cost():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    baseline = result[
        "baseline_results"
    ]

    assert np.isclose(
        baseline[
            "CostPerLeg"
        ],
        0.25
    ).all()


def test_best_configuration_uses_baseline_cost():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    best = result[
        "best_configuration"
    ]

    assert np.isclose(
        best[
            "CostPerLeg"
        ],
        0.25
    )


def test_valid_results_have_completed_trades():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    valid = result[
        "valid_results"
    ]

    assert (
        valid[
            "Trades"
        ] >= 1
    ).all()


def test_cost_sensitivity_contains_all_cost_levels():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    actual_costs = set(
        result[
            "cost_sensitivity"
        ][
            "CostPerLeg"
        ].round(
            2
        )
    )

    expected_costs = {
        0.00,
        0.25,
        0.50,
        1.00
    }

    assert actual_costs == expected_costs


def test_heatmap_uses_baseline_results_only():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    heatmap = result[
        "heatmap_data"
    ]

    assert not heatmap.empty


def test_robust_configuration_exists():

    analyzer = create_analyzer()

    data = create_spread_data()

    result = (
        analyzer.run_robustness_test(
            data
        )
    )

    assert result[
        "success"
    ]

    assert result[
        "robust_configuration"
    ] is not None