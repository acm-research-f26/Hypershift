"""Contract tests for preserving bars while deriving feature eligibility."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype
from pandas.testing import assert_frame_equal

from data import resampling


if not hasattr(resampling, "BarEligibilityPolicy") or not hasattr(
    resampling, "apply_bar_eligibility"
):
    pytest.skip(
        "Implement BarEligibilityPolicy and data.resampling.apply_bar_eligibility",
        allow_module_level=True,
    )


BarEligibilityPolicy = resampling.BarEligibilityPolicy
apply_bar_eligibility = resampling.apply_bar_eligibility


@pytest.fixture
def bars():
    return pd.DataFrame(
        {
            "close": [11.0, 12.0, 13.0, np.nan],
            "vwap": [10.5, 11.5, np.nan, np.nan],
            "expected_minutes": [2, 2, 1, 2],
            "observed_minutes": [2, 1, 1, 0],
            "coverage_fraction": [1.0, 0.5, 1.0, 0.0],
            "is_complete": [True, False, True, False],
            "is_partial": [False, False, True, False],
        },
        index=pd.date_range(
            "2026-09-30 13:32Z",
            periods=4,
            freq="2min",
            name="bar_end",
        ),
    )


def test_default_policy_masks_incomplete_and_empty_bars_but_keeps_short_final_bin(
    bars,
):
    original = bars.copy(deep=True)
    result = apply_bar_eligibility(bars, BarEligibilityPolicy())

    assert_frame_equal(bars, original)
    assert result["is_eligible"].tolist() == [True, False, True, False]
    assert is_bool_dtype(result["is_eligible"])
    assert_frame_equal(result.drop(columns="is_eligible"), original)


def test_excluding_partial_bins_masks_a_complete_short_final_bin(bars):
    result = apply_bar_eligibility(
        bars,
        BarEligibilityPolicy(
            minimum_coverage=1.0,
            partial_bar_policy="exclude",
        ),
    )

    assert result["is_eligible"].tolist() == [True, False, False, False]


def test_lower_coverage_threshold_admits_observed_incomplete_bar_but_not_empty_bar(
    bars,
):
    result = apply_bar_eligibility(
        bars,
        {
            "minimum_coverage": 0.5,
            "partial_bar_policy": "keep_and_flag",
        },
    )

    assert result["is_eligible"].tolist() == [True, True, True, False]


def test_eligibility_is_separate_from_field_specific_missingness(bars):
    result = apply_bar_eligibility(bars, BarEligibilityPolicy())

    final_bar = result.iloc[2]
    assert final_bar["is_eligible"]
    assert pd.isna(final_bar["vwap"])
    assert final_bar["close"] == 13.0

    close_mask = result["is_eligible"] & result["close"].notna()
    vwap_mask = result["is_eligible"] & result["vwap"].notna()
    assert close_mask.tolist() == [True, False, True, False]
    assert vwap_mask.tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_coverage": -0.1},
        {"minimum_coverage": 1.1},
        {"minimum_coverage": np.nan},
        {"minimum_coverage": True},
        {"partial_bar_policy": "silently_drop"},
        {"unknown_option": "value"},
    ],
)
def test_invalid_policies_raise_value_error(bars, policy):
    with pytest.raises(ValueError):
        apply_bar_eligibility(bars, policy)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_column",
        "nonpositive_expected",
        "observed_exceeds_expected",
        "incorrect_fraction",
        "incorrect_complete_flag",
        "nonboolean_flag",
    ],
)
def test_inconsistent_coverage_metadata_is_rejected(bars, damage):
    damaged = deepcopy(bars)

    if damage == "missing_column":
        damaged = damaged.drop(columns="observed_minutes")
    elif damage == "nonpositive_expected":
        damaged.loc[damaged.index[0], "expected_minutes"] = 0
    elif damage == "observed_exceeds_expected":
        damaged.loc[damaged.index[0], "observed_minutes"] = 3
    elif damage == "incorrect_fraction":
        damaged.loc[damaged.index[1], "coverage_fraction"] = 0.75
    elif damage == "incorrect_complete_flag":
        damaged.loc[damaged.index[1], "is_complete"] = True
    else:
        damaged["is_partial"] = damaged["is_partial"].astype(int)

    with pytest.raises(ValueError):
        apply_bar_eligibility(damaged, BarEligibilityPolicy())
