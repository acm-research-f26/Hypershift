"""Segment 1: causal, masked intraday returns from ObservationPanel."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.types import ObservationPanel


try:
    market = importlib.import_module("features.market")
except ModuleNotFoundError as exc:
    if exc.name not in {"features", "features.market"}:
        raise
    pytest.skip("Add features/market.py with log_returns", allow_module_level=True)


def make_panel(close, *, eligible=None, observed=None, timestamps=None, sessions=None, starts=None):
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    t, n = close.shape
    times = (pd.date_range("2024-01-02T14:45Z", periods=t, freq="15min", name="bar_end")
             if timestamps is None else pd.DatetimeIndex(timestamps, name="bar_end"))
    values = close[:, :, None]
    presence = np.isfinite(close) if observed is None else np.asarray(observed, dtype=bool)
    usable = presence if eligible is None else np.asarray(eligible, dtype=bool)
    return ObservationPanel(
        timestamps=times, node_ids=tuple(f"STOCK{i}" for i in range(n)), field_names=("close",),
        values=values, observation_mask=presence[:, :, None], eligible_mask=usable[:, :, None],
        session_ids=pd.DatetimeIndex(["2024-01-02"] * t if sessions is None else sessions),
        bar_starts=times - pd.Timedelta("15min") if starts is None else pd.DatetimeIndex(starts),
        availability_times=times, expected_minutes=np.full(t, 15),
        observed_minutes=np.full((t, n), 15), is_partial=np.zeros(t, dtype=bool),
        units={"close": "USD"}, provenance={},
    )


def test_known_returns_preserve_timestamp_and_stock_axes_and_input_values():
    panel = make_panel([[100, 200], [110, 180], [121, 198]])
    original = panel.values.copy()
    returns, mask = market.log_returns(panel)
    assert returns.shape == mask.shape == (3, 2)
    assert np.isnan(returns[0]).all() and not mask[0].any()
    np.testing.assert_allclose(returns[1:], [[np.log(1.1), np.log(0.9)], [np.log(1.1), np.log(1.1)]])
    assert mask[1:].all() and mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(panel.values, original)


def test_present_but_ineligible_bar_invalidates_both_neighboring_returns():
    panel = make_panel([[100], [110], [121], [130]], eligible=[[True], [False], [True], [True]])
    returns, mask = market.log_returns(panel)
    np.testing.assert_array_equal(mask[:, 0], [False, False, False, True])
    assert np.isnan(returns[:3]).all()
    assert returns[3, 0] == pytest.approx(np.log(130 / 121))


def test_absent_observation_cannot_be_used_even_if_eligibility_flag_is_true():
    panel = make_panel([[100], [110], [121]], observed=[[True], [False], [True]],
                       eligible=[[True], [True], [True]])
    returns, mask = market.log_returns(panel)
    assert not mask.any() and np.isnan(returns).all()


@pytest.mark.parametrize("bad_price", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_price_never_produces_a_return_or_log_warning(bad_price):
    panel = make_panel([[100], [bad_price], [121]], eligible=[[True], [True], [True]])
    with np.errstate(all="raise"):
        returns, mask = market.log_returns(panel)
    assert not mask.any() and np.isnan(returns).all()


def test_gap_in_time_axis_is_not_treated_as_an_ordinary_one_step_return():
    panel = make_panel([[100], [121]], timestamps=["2024-01-02T14:45Z", "2024-01-02T15:15Z"])
    returns, mask = market.log_returns(panel)
    assert not mask.any() and np.isnan(returns).all()


def test_overnight_boundary_is_excluded_and_next_intraday_pair_remains_usable():
    panel = make_panel(
        [[100], [110], [120], [132]],
        timestamps=["2024-01-02T20:45Z", "2024-01-02T21:00Z", "2024-01-03T14:45Z", "2024-01-03T15:00Z"],
        sessions=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
    )
    returns, mask = market.log_returns(panel)
    np.testing.assert_array_equal(mask[:, 0], [False, True, False, True])
    assert np.isnan(returns[2, 0])
    np.testing.assert_allclose(returns[[1, 3], 0], np.log(1.1))


def test_short_final_bar_can_follow_a_contiguous_eligible_bar():
    panel = make_panel([[100], [110]], timestamps=["2024-01-02T20:30Z", "2024-01-02T21:00Z"],
                       starts=["2024-01-02T19:30Z", "2024-01-02T20:30Z"])
    panel = replace(panel, expected_minutes=np.array([60, 30]), is_partial=np.array([False, True]))
    returns, mask = market.log_returns(panel)
    assert mask[1, 0] and returns[1, 0] == pytest.approx(np.log(1.1))


def test_close_field_is_selected_by_name_instead_of_field_position():
    panel = make_panel([[100], [110]])
    values = np.concatenate([np.full(panel.values.shape, 500.0), panel.values], axis=2)
    flags = np.ones(values.shape, dtype=bool)
    panel = replace(panel, values=values, field_names=("volume", "close"),
                    observation_mask=flags, eligible_mask=flags)
    returns, mask = market.log_returns(panel)
    assert mask[1, 0] and returns[1, 0] == pytest.approx(np.log(1.1))


@pytest.mark.parametrize("t", [0, 1])
def test_no_price_pair_returns_correct_shape_with_no_valid_returns(t):
    panel = make_panel(np.full((t, 2), 100.0))
    returns, mask = market.log_returns(panel)
    assert returns.shape == mask.shape == (t, 2)
    assert not mask.any() and np.isnan(returns).all()


def test_changing_future_price_cannot_change_earlier_returns():
    panel = make_panel([[100], [110], [121], [130]])
    before, before_mask = market.log_returns(panel)
    changed_values = panel.values.copy()
    changed_values[3, 0, 0] = 1_000_000
    after, after_mask = market.log_returns(replace(panel, values=changed_values))
    np.testing.assert_allclose(before[:3], after[:3], equal_nan=True)
    np.testing.assert_array_equal(before_mask, after_mask)
    assert before[3, 0] != after[3, 0]


@pytest.mark.parametrize("arguments", [{"overnight_policy": "include"}, {"gap_policy": "bridge"}])
def test_unimplemented_policy_is_rejected_instead_of_silently_ignored(arguments):
    with pytest.raises(ValueError):
        market.log_returns(make_panel([[100], [110]]), **arguments)


@pytest.mark.parametrize("damage", ["mask_shape", "numeric_mask", "metadata_length", "unsorted", "naive"])
def test_invalid_panel_structure_or_chronology_fails_explicitly(damage):
    panel = make_panel([[100], [110]])
    if damage == "mask_shape":
        panel = replace(panel, eligible_mask=np.ones((2, 1), dtype=bool))
    elif damage == "numeric_mask":
        panel = replace(panel, eligible_mask=panel.eligible_mask.astype(int))
    elif damage == "metadata_length":
        panel = replace(panel, session_ids=panel.session_ids[:1])
    elif damage == "unsorted":
        panel = replace(panel, timestamps=panel.timestamps[::-1])
    else:
        panel = replace(panel, timestamps=panel.timestamps.tz_localize(None))
    with pytest.raises(ValueError):
        market.log_returns(panel)
