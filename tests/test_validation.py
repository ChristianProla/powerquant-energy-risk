import numpy as np
import pandas as pd

from src.analytics.spread import (
    SpreadAnalyzer
)

from src.strategy.backtesting import (
    SpreadBacktester
)

from src.strategy.out_of_sample import (
    OutOfSampleValidator
)

from src.strategy.walk_forward import (
    WalkForwardValidator
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
            0.85
            * spread[
                index - 1
            ]
            + rng.normal(
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


def test_out_of_sample_split_is_chronological():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        OutOfSampleValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data()

    result = validator.run_validation(
        data,
        train_ratio=0.70,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    assert (
        result[
            "train_end"
        ]
        <
        result[
            "test_start"
        ]
    )


def test_out_of_sample_split_sizes_are_correct():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        OutOfSampleValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data()

    result = validator.run_validation(
        data,
        train_ratio=0.70,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    expected_train_size = int(
        len(data)
        * 0.70
    )

    expected_test_size = (
        len(data)
        - expected_train_size
    )

    assert (
        result[
            "train_observations"
        ]
        ==
        expected_train_size
    )

    assert (
        result[
            "test_observations"
        ]
        ==
        expected_test_size
    )


def test_out_of_sample_split_uses_all_observations():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        OutOfSampleValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data()

    result = validator.run_validation(
        data,
        train_ratio=0.70,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    total_split_observations = (
        result[
            "train_observations"
        ]
        +
        result[
            "test_observations"
        ]
    )

    assert (
        total_split_observations
        ==
        len(data)
    )


def test_out_of_sample_test_starts_after_training():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        OutOfSampleValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data()

    result = validator.run_validation(
        data,
        train_ratio=0.70,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    assert (
        result[
            "test_start"
        ]
        >
        result[
            "train_end"
        ]
    )


def test_walk_forward_uses_multiple_test_periods():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        WalkForwardValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    folds = result[
        "fold_results"
    ]

    assert len(
        folds
    ) >= 2


def test_walk_forward_test_periods_are_chronological():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        WalkForwardValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    folds = (
        result[
            "fold_results"
        ]
        .sort_values(
            "Fold"
        )
        .reset_index(
            drop=True
        )
    )

    for index in range(
        1,
        len(
            folds
        )
    ):

        assert (
            folds.loc[
                index,
                "TestStart"
            ]
            >
            folds.loc[
                index - 1,
                "TestStart"
            ]
        )


def test_walk_forward_train_period_expands():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        WalkForwardValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    folds = (
        result[
            "fold_results"
        ]
        .sort_values(
            "Fold"
        )
        .reset_index(
            drop=True
        )
    )

    train_end_times = (
        folds[
            "TrainEnd"
        ].tolist()
    )

    assert train_end_times == sorted(
        train_end_times
    )


def test_each_walk_forward_test_occurs_after_training():

    analyzer = SpreadAnalyzer()

    backtester = SpreadBacktester()

    validator = (
        WalkForwardValidator(
            analyzer,
            backtester
        )
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        transaction_cost_per_leg=0.25
    )

    assert result[
        "success"
    ]

    folds = result[
        "fold_results"
    ]

    for _, fold in (
        folds.iterrows()
    ):

        assert (
            fold[
                "TestStart"
            ]
            >
            fold[
                "TrainEnd"
            ]
        )