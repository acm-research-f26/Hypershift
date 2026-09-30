"""Offline acceptance checks for explicit node/time alignment.

Only the missing learner module is skipped; errors inside it must fail normally.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


try:
    alignment = importlib.import_module("data.alignment")
except ModuleNotFoundError as exc:
    if exc.name != "data.alignment":
        raise
    pytest.skip("Implement data/alignment.py for roadmap item 4", allow_module_level=True)


@pytest.fixture
def grid():
    return pd.date_range("2024-01-02 14:32Z", periods=3, freq="2min", name="bar_end")


def frame(index, close, *, usable=None):
    return pd.DataFrame(
        {"close": close, "vwap": np.asarray(close, dtype=float) - 0.25,
         "is_eligible": [True] * len(index) if usable is None else usable}, index=index)


def test_alignment_uses_labels_and_explicit_node_and_field_orders(grid):
    a = frame(grid, [10, 11, 12])
    b = frame(grid[[0, 2]], [100, 102])
    originals = {"AAA": a.copy(deep=True), "BBB": b.copy(deep=True)}
    values, observed, eligible = alignment.align_nodes_and_times(
        {"AAA": a, "BBB": b}, ("BBB", "AAA"), grid, ("vwap", "close"))
    assert values.shape == (3, 2, 2)
    np.testing.assert_allclose(values[:, 1, :], [[9.75, 10], [10.75, 11], [11.75, 12]])
    np.testing.assert_allclose(values[[0, 2], 0, :], [[99.75, 100], [101.75, 102]])
    assert np.isnan(values[1, 0]).all()
    assert not observed[1, 0].any()
    assert not eligible[1, 0].any()
    assert observed[:, 1].all() and eligible[:, 1].all()
    for node, original in originals.items():
        assert_frame_equal({"AAA": a, "BBB": b}[node], original)


def test_observation_and_policy_masks_are_distinct_and_field_specific(grid):
    a = frame(grid, [10, 11, 12], usable=[False, True, True])
    a.loc[grid[1], "vwap"] = np.nan
    a.loc[grid[2], "vwap"] = np.inf
    values, observed, eligible = alignment.align_nodes_and_times(
        {"AAA": a}, ("AAA",), grid, ("close", "vwap"))
    assert values[0, 0, 0] == 10  # Policy exclusion preserves the raw aggregate.
    np.testing.assert_array_equal(observed[:, 0], [[True, True], [True, False], [True, False]])
    np.testing.assert_array_equal(eligible[:, 0], [[False, False], [True, False], [True, False]])
    assert observed.dtype == eligible.dtype == np.dtype(bool)


def test_equivalent_timezones_align_by_instant_and_outside_rows_are_excluded(grid):
    extra = grid[-1] + pd.Timedelta("2min")
    index = grid.append(pd.DatetimeIndex([extra])).tz_convert("America/New_York")
    values, _, _ = alignment.align_nodes_and_times(
        {"AAA": frame(index, [10, 11, 12, 999])}, ("AAA",), grid, ("close",))
    np.testing.assert_array_equal(values[:, 0, 0], [10, 11, 12])


def test_empty_time_grid_has_typed_three_dimensional_outputs(grid):
    values, observed, eligible = alignment.align_nodes_and_times(
        {"AAA": frame(grid, [10, 11, 12])}, ("AAA",), grid[:0], ("close",))
    assert values.shape == observed.shape == eligible.shape == (0, 1, 1)
    assert values.dtype.kind == "f"
    assert observed.dtype == eligible.dtype == np.dtype(bool)


@pytest.mark.parametrize("damage", ["duplicate", "unsorted", "naive", "missing_time"])
def test_bad_frame_timestamps_cannot_be_aligned_by_row_position(grid, damage):
    indices = {
        "duplicate": grid[[0, 0, 2]], "unsorted": grid[[1, 0, 2]],
        "naive": grid.tz_localize(None),
        "missing_time": pd.DatetimeIndex([grid[0], pd.NaT, grid[2]]),
    }
    with pytest.raises(ValueError):
        alignment.align_nodes_and_times(
            {"AAA": frame(indices[damage], [10, 11, 12])}, ("AAA",), grid, ("close",))


@pytest.mark.parametrize("nodes", [(), ("AAA", "AAA"), ("BBB",), "AAA"])
def test_invalid_node_orders_fail_instead_of_silently_changing_universe(grid, nodes):
    with pytest.raises(ValueError):
        alignment.align_nodes_and_times(
            {"AAA": frame(grid, [10, 11, 12])}, nodes, grid, ("close",))


@pytest.mark.parametrize("damage", ["missing_field", "numeric_eligibility", "missing_eligibility"])
def test_missing_fields_and_ambiguous_eligibility_are_rejected(grid, damage):
    a = frame(grid, [10, 11, 12])
    if damage == "missing_field":
        a = a.drop(columns="close")
    elif damage == "numeric_eligibility":
        a["is_eligible"] = [1, 0, 1]
    else:
        a["is_eligible"] = pd.array([True, None, True], dtype="boolean")
    with pytest.raises(ValueError):
        alignment.align_nodes_and_times({"AAA": a}, ("AAA",), grid, ("close",))
