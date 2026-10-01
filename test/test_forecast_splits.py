"""One fixed split respects prediction times and information deadlines."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.windows import make_forecast_samples
from test.test_market_returns import make_panel


try:
    splits_module = importlib.import_module("experiments.splits")
except ModuleNotFoundError as exc:
    if exc.name != "experiments.splits":
        raise
    pytest.skip("Add experiments/splits.py with split_forecast_samples", allow_module_level=True)


def fixture_samples(delay="30s"):
    panel = make_panel(np.column_stack((100 * 1.01 ** np.arange(12), 200 * 0.99 ** np.arange(12))))
    panel = replace(panel, availability_times=panel.timestamps + pd.Timedelta(delay))
    samples = list(make_forecast_samples(panel, 2))
    bounds = dict(train_start=panel.timestamps[0], fit_cutoff=panel.timestamps[4],
                  validation_end=panel.timestamps[7], test_end=panel.timestamps[10])
    return panel, samples, bounds


def run_split(samples, bounds):
    return splits_module.split_forecast_samples(samples, **bounds)


def test_delayed_labels_are_excluded_at_training_and_validation_deadlines():
    panel, samples, bounds = fixture_samples()
    result = run_split(samples, bounds)
    assert list(result) == ["train", "validation", "test"]
    assert [s.target_start for s in result["train"]] == list(panel.timestamps[[2]])
    assert [s.target_start for s in result["validation"]] == list(panel.timestamps[[4, 5]])
    assert [s.target_start for s in result["test"]] == list(panel.timestamps[[7, 8, 9]])
    assert all(s.origin_time < bounds["fit_cutoff"] and s.target_availability <= bounds["fit_cutoff"]
               for s in result["train"])
    assert all(s.target_availability <= bounds["validation_end"] for s in result["validation"])


def test_exact_boundaries_use_half_open_origins_and_inclusive_information_cutoffs():
    panel, samples, bounds = fixture_samples("0s")
    result = run_split(samples, bounds)
    assert [s.origin_time for s in result["train"]] == list(panel.timestamps[[2, 3]])
    assert [s.origin_time for s in result["validation"]] == list(panel.timestamps[[4, 5, 6]])
    assert [s.origin_time for s in result["test"]] == list(panel.timestamps[[7, 8, 9]])
    assert result["train"][-1].target_availability == bounds["fit_cutoff"]
    assert result["validation"][-1].target_availability == bounds["validation_end"]
    assert not any(s.origin_time == bounds["test_end"] for group in result.values() for s in group)


def test_test_labels_can_arrive_after_end_of_prediction_period():
    _, samples, bounds = fixture_samples()
    result = run_split(samples, bounds)
    assert result["test"][-1].target_availability > bounds["test_end"]
    shorter = dict(bounds, test_end=bounds["test_end"] - pd.Timedelta("5min"))
    again = run_split(samples, shorter)
    assert again["test"][-1].target_end > shorter["test_end"]


def test_prediction_origin_not_bar_end_decides_split():
    panel, samples, bounds = fixture_samples()
    # This sample's target begins before the cutoff, but prediction occurs after it.
    bounds["fit_cutoff"] = panel.timestamps[4] + pd.Timedelta("15s")
    result = run_split(samples, bounds)
    sample = samples[2]
    assert sample.target_start < bounds["fit_cutoff"] < sample.origin_time
    assert any(s is sample for s in result["validation"])
    assert not any(s is sample for s in result["train"])


def test_training_requires_at_least_one_stock_with_both_inputs_and_label():
    _, samples, bounds = fixture_samples()
    altered = list(samples)
    altered[0] = replace(samples[0], eligible_nodes=np.array([True, False]),
                         target_mask=np.array([False, True]))
    assert run_split(altered, bounds)["train"] == []
    altered[0] = replace(samples[0], eligible_nodes=np.array([True, False]),
                         target_mask=np.array([True, False]))
    assert len(run_split(altered, bounds)["train"]) == 1


def test_unlabeled_validation_and_test_predictions_remain_for_coverage():
    _, samples, bounds = fixture_samples()
    altered = [replace(s, target=np.full(s.target.shape, np.nan),
                       target_mask=np.zeros(s.target_mask.shape, dtype=bool)) for s in samples]
    result = run_split(altered, bounds)
    assert result["train"] == []
    assert len(result["validation"]) == 2 and len(result["test"]) == 3
    assert all(not s.target_mask.any() for s in result["validation"] + result["test"])


def test_changing_labels_known_only_after_fit_cannot_change_training():
    _, samples, bounds = fixture_samples()
    altered = [replace(s, target=s.target + 999, target_mask=np.zeros_like(s.target_mask))
               if s.target_availability > bounds["fit_cutoff"] else s for s in samples]
    original = run_split(samples, bounds)["train"]
    changed = run_split(altered, bounds)["train"]
    assert [s.origin_time for s in original] == [s.origin_time for s in changed]
    for before, after in zip(original, changed):
        np.testing.assert_array_equal(before.target, after.target)


def test_long_label_delay_excludes_sample_even_when_target_ended_before_fit():
    _, samples, bounds = fixture_samples()
    altered = list(samples)
    altered[0] = replace(samples[0], target_availability=bounds["fit_cutoff"] + pd.Timedelta("1s"))
    assert altered[0].target_end < bounds["fit_cutoff"]
    assert run_split(altered, bounds)["train"] == []


def test_equivalent_timezones_do_not_change_membership():
    _, samples, bounds = fixture_samples()
    alternate = {key: value.tz_convert("America/New_York") for key, value in bounds.items()}
    moved = [replace(s, origin_time=s.origin_time.tz_convert("Asia/Tokyo"),
                     target_end=s.target_end.tz_convert("America/New_York"),
                     target_availability=s.target_availability.tz_convert("Europe/London")) for s in samples]
    original = run_split(samples, bounds)
    changed = run_split(moved, alternate)
    for name in original:
        assert [s.origin_time for s in original[name]] == [s.origin_time for s in changed[name]]


def test_aware_boundary_strings_and_generator_input_are_supported():
    _, samples, bounds = fixture_samples()
    strings = {key: value.isoformat() for key, value in bounds.items()}
    result = run_split(iter(samples), strings)
    assert [len(result[name]) for name in ("train", "validation", "test")] == [1, 2, 3]


def test_empty_input_returns_three_empty_lists():
    _, _, bounds = fixture_samples()
    assert run_split([], bounds) == {"train": [], "validation": [], "test": []}


def test_excluded_origins_are_not_reassigned_to_a_later_split():
    panel, samples, bounds = fixture_samples()
    result = run_split(samples, bounds)
    admitted = [s.target_start for group in result.values() for s in group]
    assert panel.timestamps[3] not in admitted and panel.timestamps[6] not in admitted


def test_split_preserves_sample_identity_and_does_not_mutate_arrays():
    _, samples, bounds = fixture_samples()
    before = [(s.values.copy(), s.mask.copy(), s.target.copy(), s.target_mask.copy()) for s in samples]
    result = run_split(samples, bounds)
    assert result["train"][0] is samples[0]
    for sample, original in zip(samples, before):
        for actual, expected in zip((sample.values, sample.mask, sample.target, sample.target_mask), original):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["train_start", "fit_cutoff", "validation_end", "test_end"])
@pytest.mark.parametrize("damage", ["naive", "missing"])
def test_naive_or_missing_boundaries_are_rejected(name, damage):
    _, samples, bounds = fixture_samples()
    bounds[name] = bounds[name].tz_localize(None) if damage == "naive" else pd.NaT
    with pytest.raises(ValueError):
        run_split(samples, bounds)


@pytest.mark.parametrize("name", ["fit_cutoff", "validation_end", "test_end"])
def test_nonincreasing_boundaries_are_rejected(name):
    _, samples, bounds = fixture_samples()
    bounds[name] = bounds["train_start"]
    with pytest.raises(ValueError):
        run_split(samples, bounds)


@pytest.mark.parametrize("name", ["origin_time", "target_end", "target_availability"])
@pytest.mark.parametrize("damage", ["naive", "missing"])
def test_naive_or_missing_sample_times_are_rejected(name, damage):
    _, samples, bounds = fixture_samples()
    value = getattr(samples[0], name)
    changed = value.tz_localize(None) if damage == "naive" else pd.NaT
    samples[0] = replace(samples[0], **{name: changed})
    with pytest.raises(ValueError):
        run_split(samples, bounds)


@pytest.mark.parametrize("damage", ["target_not_future", "label_available_before_end"])
def test_inconsistent_target_timing_is_rejected(damage):
    _, samples, bounds = fixture_samples()
    if damage == "target_not_future":
        samples[0] = replace(samples[0], target_end=samples[0].origin_time)
    else:
        samples[0] = replace(samples[0], target_availability=samples[0].target_end - pd.Timedelta("1s"))
    with pytest.raises(ValueError):
        run_split(samples, bounds)


@pytest.mark.parametrize("damage", ["reversed", "duplicate"])
def test_unsorted_or_duplicate_prediction_origins_are_rejected(damage):
    _, samples, bounds = fixture_samples()
    samples = samples[::-1] if damage == "reversed" else [samples[0], samples[0]]
    with pytest.raises(ValueError):
        run_split(samples, bounds)
