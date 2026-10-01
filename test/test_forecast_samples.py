"""Forecast samples preserve causal inputs, independent label masks, and timing."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import data.types as sample_types
from data import windows
from features.market import log_returns
from test.test_market_returns import make_panel


if not hasattr(windows, "make_forecast_samples"):
    pytest.skip("Add make_forecast_samples to data/windows.py", allow_module_level=True)


def test_known_sample_preserves_axes_and_forward_target():
    panel = make_panel([[100, 200], [110, 180], [121, 198], [133.1, 217.8], [146.41, 239.58]])
    samples = list(windows.make_forecast_samples(panel, 2))
    assert len(samples) == 2
    sample = samples[0]
    assert isinstance(sample, sample_types.ForecastSample)
    assert sample.values.shape == sample.mask.shape == (2, 2, 1)
    np.testing.assert_allclose(sample.values[:, :, 0], [[np.log(1.1), np.log(0.9)], [np.log(1.1), np.log(1.1)]])
    assert sample.mask.all()
    np.testing.assert_allclose(sample.target, [np.log(1.1), np.log(1.1)])
    assert sample.eligible_nodes.all() and sample.target_mask.all()
    assert sample.origin_time == panel.timestamps[2]
    assert sample.history_times.equals(panel.timestamps[1:3])
    assert sample.node_ids == panel.node_ids and sample.feature_names == ("log_return",)
    assert sample.target_start == panel.timestamps[2] and sample.target_end == panel.timestamps[3]
    assert sample.target_availability == panel.timestamps[3]
    assert sample.target_units == "log_return" and sample.snapshot_id is None


def test_first_session_return_cannot_supply_complete_history():
    panel = make_panel([100, 110, 121, 133.1])
    samples = list(windows.make_forecast_samples(panel, 1))
    assert [sample.target_start for sample in samples] == list(panel.timestamps[1:3])


def test_stock_eligibility_uses_history_and_preserves_full_node_order():
    panel = make_panel([[100, 200], [110, 180], [121, 198], [133.1, 217.8]],
                       eligible=[[True, True], [True, False], [True, True], [True, True]])
    sample, = list(windows.make_forecast_samples(panel, 2))
    np.testing.assert_array_equal(sample.eligible_nodes, [True, False])
    np.testing.assert_array_equal(sample.target_mask, [True, True])
    assert np.isnan(sample.values[:, 1]).all()
    assert sample.values.shape == (2, 2, 1) and sample.node_ids == panel.node_ids


def test_missing_future_prices_do_not_change_inputs_or_prediction_eligibility():
    panel = make_panel([[100, 200], [110, 180], [121, 198], [133.1, 217.8], [146.41, 239.58]])
    before = list(windows.make_forecast_samples(panel, 2))[0]
    prices = panel.values.copy()
    prices[3:] = np.nan
    after = list(windows.make_forecast_samples(replace(panel, values=prices), 2))[0]
    np.testing.assert_array_equal(before.values, after.values)
    np.testing.assert_array_equal(before.mask, after.mask)
    np.testing.assert_array_equal(before.eligible_nodes, after.eligible_nodes)
    assert before.origin_time == after.origin_time
    assert before.target_mask.all() and not after.target_mask.any()
    assert np.isnan(after.target).all()


def test_future_eligibility_policy_cannot_choose_prediction_nodes():
    panel = make_panel([100, 110, 121, 133.1])
    policy = panel.eligible_mask.copy()
    policy[-1] = False
    sample, = list(windows.make_forecast_samples(replace(panel, eligible_mask=policy), 2))
    assert sample.eligible_nodes.all() and not sample.target_mask.any()


def test_no_stock_with_complete_history_produces_no_samples():
    panel = make_panel([100, 110, 121, 133.1])
    panel = replace(panel, eligible_mask=np.zeros(panel.eligible_mask.shape, dtype=bool))
    assert list(windows.make_forecast_samples(panel, 2)) == []


def test_declared_delay_sets_origin_and_label_availability_separately():
    panel = make_panel([100, 110, 121, 133.1])
    panel = replace(panel, availability_times=panel.timestamps + pd.Timedelta("30s"))
    sample, = list(windows.make_forecast_samples(panel, 2))
    assert sample.origin_time == panel.timestamps[2] + pd.Timedelta("30s")
    assert sample.target_start == panel.timestamps[2]
    assert sample.target_availability == panel.timestamps[3] + pd.Timedelta("30s")
    assert sample.history_times[-1] < sample.origin_time < sample.target_end


def test_origin_includes_close_before_earliest_historical_return():
    panel = make_panel([100, 110, 121, 133.1])
    available = list(panel.timestamps)
    available[0] = panel.timestamps[2] + pd.Timedelta("1min")
    panel = replace(panel, availability_times=pd.DatetimeIndex(available))
    sample, = list(windows.make_forecast_samples(panel, 2))
    assert sample.history_times[0] == panel.timestamps[1]
    assert sample.origin_time == available[0]
    assert sample.origin_time < sample.target_end


@pytest.mark.parametrize("extra_delay", ["0s", "1s"])
def test_inputs_arriving_at_or_after_target_end_cannot_issue_forecast(extra_delay):
    panel = make_panel([100, 110, 121, 133.1])
    available = list(panel.timestamps)
    available[0] = panel.timestamps[3] + pd.Timedelta(extra_delay)
    panel = replace(panel, availability_times=pd.DatetimeIndex(available))
    assert list(windows.make_forecast_samples(panel, 2)) == []


def test_target_availability_is_not_used_to_delay_prediction_origin():
    panel = make_panel([100, 110, 121, 133.1])
    available = list(panel.timestamps)
    available[3] += pd.Timedelta("1h")
    sample, = list(windows.make_forecast_samples(replace(panel, availability_times=pd.DatetimeIndex(available)), 2))
    assert sample.origin_time == panel.timestamps[2]
    assert sample.target_availability == available[3]


def test_session_boundaries_exclude_overnight_history_and_labels():
    times = pd.date_range("2024-01-02T14:45Z", periods=4, freq="15min").append(
        pd.date_range("2024-01-03T14:45Z", periods=4, freq="15min"))
    panel = make_panel([100, 110, 121, 133.1, 150, 165, 181.5, 199.65],
                       timestamps=times, sessions=["2024-01-02"] * 4 + ["2024-01-03"] * 4)
    samples = list(windows.make_forecast_samples(panel, 2))
    assert [sample.target_start for sample in samples] == [times[2], times[6]]
    for sample in samples:
        assert sample.history_times[0].date() == sample.target_end.date()


def test_gap_in_close_support_invalidates_earliest_return_even_outside_feature_slice():
    panel = make_panel([100, 110, 121, 133.1], timestamps=["2024-01-02T14:45Z", "2024-01-02T15:15Z",
                                                       "2024-01-02T15:30Z", "2024-01-02T15:45Z"])
    assert list(windows.iter_forecast_origins(panel, 2)) == [2]
    assert list(windows.make_forecast_samples(panel, 2)) == []


def test_mutating_sample_arrays_does_not_change_panel_or_other_samples():
    panel = make_panel([100, 110, 121, 133.1, 146.41])
    original = panel.values.copy()
    samples = list(windows.make_forecast_samples(panel, 2))
    later_values = samples[1].values.copy()
    later_mask = samples[1].mask.copy()
    samples[0].values[:] = -999
    samples[0].mask[:] = False
    samples[0].target[:] = -999
    samples[0].target_mask[:] = False
    np.testing.assert_array_equal(panel.values, original)
    np.testing.assert_array_equal(samples[1].values, later_values)
    np.testing.assert_array_equal(samples[1].mask, later_mask)
    expected, _ = log_returns(panel)
    np.testing.assert_array_equal(samples[1].target, expected[4])


def test_equivalent_timezones_are_normalized_to_utc():
    panel = make_panel([100, 110, 121, 133.1])
    panel = replace(panel, timestamps=panel.timestamps.tz_convert("America/New_York"),
                    availability_times=panel.availability_times.tz_convert("America/New_York"))
    sample, = list(windows.make_forecast_samples(panel, 2))
    assert str(sample.origin_time.tz) == str(sample.history_times.tz) == "UTC"
    assert str(sample.target_start.tz) == str(sample.target_end.tz) == "UTC"
    assert str(sample.target_availability.tz) == "UTC"


@pytest.mark.parametrize("size,lookback", [(0, 1), (1, 1), (4, 4), (4, 5)])
def test_insufficient_bars_produce_no_samples(size, lookback):
    assert list(windows.make_forecast_samples(make_panel(np.arange(size) + 100), lookback)) == []


@pytest.mark.parametrize("lookback", [0, -1, True, 1.5, None])
def test_invalid_lookback_is_rejected(lookback):
    with pytest.raises(ValueError):
        list(windows.make_forecast_samples(make_panel([100, 110, 121]), lookback))


@pytest.mark.parametrize("damage", ["length", "naive", "missing", "premature"])
def test_invalid_input_availability_is_rejected(damage):
    panel = make_panel([100, 110, 121, 133.1])
    available = panel.availability_times
    if damage == "length":
        available = available[:-1]
    elif damage == "naive":
        available = available.tz_localize(None)
    elif damage == "missing":
        available = pd.DatetimeIndex([pd.NaT, *available[1:]])
    else:
        available = available - pd.Timedelta("1s")
    with pytest.raises(ValueError):
        list(windows.make_forecast_samples(replace(panel, availability_times=available), 2))
