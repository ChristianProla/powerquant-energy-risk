import numpy as np
import pandas as pd
import pytest

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


def create_oos_validator():

    return OutOfSampleValidator(
        SpreadAnalyzer(),
        SpreadBacktester()
    )


def create_walk_forward_validator():

    return WalkForwardValidator(
        SpreadAnalyzer(),
        SpreadBacktester()
    )


# ==========================================================
# Out-of-sample tests
# ==========================================================

def test_out_of_sample_split_is_chronological():

    validator = (
        create_oos_validator()
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

    validator = (
        create_oos_validator()
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

    validator = (
        create_oos_validator()
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

    validator = (
        create_oos_validator()
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


def test_out_of_sample_parameter_results_are_valid():

    validator = (
        create_oos_validator()
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

    parameters = result[
        "parameter_results"
    ]

    assert not parameters.empty

    assert (
        parameters[
            "ExitZ"
        ]
        <
        parameters[
            "EntryZ"
        ]
    ).all()


def test_out_of_sample_selected_parameters_are_valid():

    validator = (
        create_oos_validator()
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
            "selected_exit"
        ]
        <
        result[
            "selected_entry"
        ]
    )


def test_out_of_sample_warmup_does_not_enter_test_backtest():

    validator = (
        create_oos_validator()
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

    test_backtest = result[
        "test_backtest_data"
    ]

    assert not test_backtest.empty

    assert (
        test_backtest[
            "TimeDK"
        ].min()
        >=
        result[
            "test_start"
        ]
    )


def test_future_test_data_does_not_change_parameter_selection():

    train_ratio = 0.70

    original_data = (
        create_spread_data(
            observations=1000
        )
    )

    modified_data = (
        original_data.copy()
    )

    split_index = int(
        len(
            modified_data
        )
        * train_ratio
    )

    future_length = (
        len(
            modified_data
        )
        - split_index
    )

    future_spread = (
        np.linspace(
            -200,
            200,
            future_length
        )
    )

    modified_data.loc[
        split_index:,
        "Spread"
    ] = future_spread

    modified_data.loc[
        split_index:,
        "DK1Price"
    ] = (
        100
        + future_spread
    )

    modified_data.loc[
        split_index:,
        "DK2Price"
    ] = 100.0

    original_result = (
        create_oos_validator()
        .run_validation(
            original_data,
            train_ratio=train_ratio,
            transaction_cost_per_leg=0.25
        )
    )

    modified_result = (
        create_oos_validator()
        .run_validation(
            modified_data,
            train_ratio=train_ratio,
            transaction_cost_per_leg=0.25
        )
    )

    assert original_result[
        "success"
    ]

    assert modified_result[
        "success"
    ]

    assert (
        original_result[
            "selected_window"
        ]
        ==
        modified_result[
            "selected_window"
        ]
    )

    assert (
        original_result[
            "selected_entry"
        ]
        ==
        modified_result[
            "selected_entry"
        ]
    )

    assert (
        original_result[
            "selected_exit"
        ]
        ==
        modified_result[
            "selected_exit"
        ]
    )


# ==========================================================
# Walk-forward tests
# ==========================================================

def test_walk_forward_uses_multiple_test_periods():

    validator = (
        create_walk_forward_validator()
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

    assert len(
        result[
            "fold_results"
        ]
    ) >= 2


def test_walk_forward_test_periods_are_chronological():

    validator = (
        create_walk_forward_validator()
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

    validator = (
        create_walk_forward_validator()
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

    train_sizes = (
        folds[
            "TrainObservations"
        ].tolist()
    )

    assert train_sizes == sorted(
        train_sizes
    )

    assert len(
        set(
            train_sizes
        )
    ) == len(
        train_sizes
    )


def test_each_walk_forward_test_occurs_after_training():

    validator = (
        create_walk_forward_validator()
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


def test_walk_forward_selected_parameters_are_valid():

    validator = (
        create_walk_forward_validator()
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

    selections = result[
        "parameter_selections"
    ]

    assert (
        selections[
            "ExitZ"
        ]
        <
        selections[
            "EntryZ"
        ]
    ).all()


def test_walk_forward_initial_training_ratio_is_configurable():

    validator = (
        create_walk_forward_validator()
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        transaction_cost_per_leg=0.25,
        initial_train_ratio=0.60
    )

    assert result[
        "success"
    ]

    assert result[
        "initial_train_ratio"
    ] == pytest.approx(
        0.60
    )

    assert result[
        "initial_train_observations"
    ] == int(
        len(data)
        * 0.60
    )


def test_walk_forward_rejects_invalid_initial_training_ratio():

    validator = (
        create_walk_forward_validator()
    )

    data = create_spread_data(
        observations=1500
    )

    result = validator.run_validation(
        data,
        n_folds=3,
        initial_train_ratio=0.95
    )

    assert not result[
        "success"
    ]


def test_walk_forward_max_drawdown_includes_initial_loss():

    validator = (
        create_walk_forward_validator()
    )

    data = pd.DataFrame(
        {
            "WalkForwardCumulativePnL": [
                -2.0,
                -5.0,
                -3.0
            ]
        }
    )

    drawdown = (
        validator
        ._calculate_max_drawdown(
            data
        )
    )

    assert drawdown == pytest.approx(
        -5.0
    )


def test_walk_forward_combined_test_data_has_no_duplicate_times():

    validator = (
        create_walk_forward_validator()
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

    combined = result[
        "combined_test_data"
    ]

    assert not combined[
        "TimeDK"
    ].duplicated().any()