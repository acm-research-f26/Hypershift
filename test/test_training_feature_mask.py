"""Training-window history is selected once on the original feature time axis."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.windows import make_forecast_samples
from experiments.splits import split_forecast_samples
from features.market import log_returns
from features.transforms import fit_standardizer
from test.test_market_returns import make_panel


try:
    pipeline = importlib.import_module("features.pipeline")
except ModuleNotFoundError as exc:
    if exc.name != "features.pipeline":
        raise
    pytest.skip("Add features/pipeline.py with training_feature_mask", allow_module_level=True)

if not hasattr(pipeline, "training_feature_mask"):
    pytest.skip("Add training_feature_mask to features/pipeline.py", allow_module_level=True)


def fixture():
    step = np.arange(8, dtype=float)
    panel = make_panel(np.column_stack((100 * np.exp(.01 * step ** 2), 200 * np.exp(-.005 * step ** 2))))
    samples = list(make_forecast_samples(panel, 2))
    cutoff = panel.timestamps[4]
    split = split_forecast_samples(
        samples, train_start=panel.timestamps[0], fit_cutoff=cutoff,
        validation_end=panel.timestamps[6], test_end=panel.timestamps[7],
    )
    assert len(split["train"]) == 2
    return panel, split, cutoff


def build(panel, samples, cutoff, **changes):
    arguments = dict(node_ids=panel.node_ids, feature_names=("log_return",), fit_cutoff=cutoff)
    arguments.update(changes)
    return pipeline.training_feature_mask(panel.timestamps, samples, **arguments)


def test_overlapping_training_windows_select_unique_observations_for_fitting():
    panel, split, cutoff = fixture()
    selected = build(panel, split["train"], cutoff)
    expected = np.zeros((8, 2, 1), dtype=bool)
    expected[1:4] = True
    np.testing.assert_array_equal(selected, expected)
    returns, valid = log_returns(panel)
    state = fit_standardizer(
        returns[:, :, None], selected & valid[:, :, None], selected.any(axis=(1, 2)),
        node_ids=panel.node_ids, feature_names=("log_return",),
    )
    np.testing.assert_array_equal(state.count, [[3], [3]])
    np.testing.assert_allclose(state.mean[:, 0], [.03, -.015], atol=1e-14)
    np.testing.assert_allclose(state.scale[:, 0], np.sqrt(2 / 3) * np.array([.02, .01]), atol=1e-14)


def test_node_eligibility_applies_per_window_before_masks_are_combined():
    panel, split, cutoff = fixture()
    samples = [replace(split["train"][0], eligible_nodes=np.array([True, False])),
               replace(split["train"][1], eligible_nodes=np.array([False, True]))]
    result = build(panel, samples, cutoff)
    expected = np.zeros((8, 2, 1), dtype=bool)
    expected[1:3, 0, 0] = True
    expected[2:4, 1, 0] = True
    np.testing.assert_array_equal(result, expected)


def test_feature_mask_still_excludes_an_invalid_observation():
    panel, split, cutoff = fixture()
    mask = split["train"][0].mask.copy()
    mask[0, 0, 0] = False
    samples = [replace(split["train"][0], mask=mask), split["train"][1]]
    result = build(panel, samples, cutoff)
    assert not result[1, 0, 0] and result[1, 1, 0]
    assert result[2:4].all()


def test_multiple_features_keep_their_separate_observation_masks():
    panel, split, cutoff = fixture()
    names = ("log_return", "other")
    samples = [replace(sample, feature_names=names, values=np.repeat(sample.values, 2, axis=2),
                       mask=np.repeat(sample.mask, 2, axis=2))
               for sample in split["train"]]
    samples[0].mask[0, :, 1] = False
    result = build(panel, samples, cutoff, feature_names=names)
    expected = np.zeros((8, 2, 2), dtype=bool)
    expected[1:4, :, 0] = True
    expected[2:4, :, 1] = True
    np.testing.assert_array_equal(result, expected)


def test_historical_context_can_precede_the_training_origin_start():
    panel, split, cutoff = fixture()
    sample = split["train"][0]
    result = build(panel, [sample], cutoff)
    assert sample.history_times[0] < sample.origin_time
    assert result[1:3].all() and not result[3:].any()


def test_empty_training_list_has_no_selected_observations():
    panel, _, cutoff = fixture()
    result = build(panel, [], cutoff)
    assert result.shape == (8, 2, 1) and result.dtype.kind == "b" and not result.any()


def test_generator_input_and_equivalent_timezones_give_the_same_selection():
    panel, split, cutoff = fixture()
    samples = [replace(sample,
                       history_times=sample.history_times.tz_convert("Asia/Tokyo"),
                       origin_time=sample.origin_time.tz_convert("America/New_York"))
               for sample in split["train"]]
    result = build(panel, iter(samples), cutoff.tz_convert("Europe/London").isoformat())
    np.testing.assert_array_equal(result, build(panel, split["train"], cutoff))


def test_selection_does_not_mutate_samples_or_panel_arrays():
    panel, split, cutoff = fixture()
    original = [(sample.values.copy(), sample.mask.copy(), sample.eligible_nodes.copy())
                for sample in split["train"]]
    panel_before = panel.values.copy()
    result = build(panel, split["train"], cutoff)
    result[:] = False
    for sample, arrays in zip(split["train"], original):
        for actual, expected in zip((sample.values, sample.mask, sample.eligible_nodes), arrays):
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(panel.values, panel_before)


def test_validation_samples_cannot_be_used_as_training_history():
    panel, split, cutoff = fixture()
    with pytest.raises(ValueError):
        build(panel, split["validation"], cutoff)


def test_training_label_must_already_be_known_at_fit_cutoff():
    panel, split, cutoff = fixture()
    delayed = replace(split["train"][0], target_availability=cutoff + pd.Timedelta("1s"))
    with pytest.raises(ValueError):
        build(panel, [delayed], cutoff)


@pytest.mark.parametrize("damage", ["naive", "missing", "duplicate", "unsorted"])
def test_original_feature_time_axis_must_be_aware_present_unique_and_sorted(damage):
    panel, split, cutoff = fixture()
    times = panel.timestamps
    if damage == "naive":
        times = times.tz_localize(None)
    elif damage == "missing":
        times = pd.DatetimeIndex([pd.NaT, *times[1:]])
    elif damage == "duplicate":
        times = pd.DatetimeIndex([times[0], times[0], *times[2:]])
    else:
        times = times[::-1]
    with pytest.raises(ValueError):
        build(replace(panel, timestamps=times), split["train"], cutoff)


@pytest.mark.parametrize("cutoff", [pd.NaT, "2024-01-02 15:00"])
def test_cutoff_must_be_present_and_timezone_aware(cutoff):
    panel, split, _ = fixture()
    with pytest.raises(ValueError):
        build(panel, split["train"], cutoff)


@pytest.mark.parametrize("damage", ["empty", "naive", "missing", "duplicate", "unsorted", "not_on_axis", "after_origin"])
def test_history_times_must_describe_sorted_past_rows_on_the_feature_axis(damage):
    panel, split, cutoff = fixture()
    sample = split["train"][0]
    history = sample.history_times
    if damage == "empty":
        history = history[:0]
    elif damage == "naive":
        history = history.tz_localize(None)
    elif damage == "missing":
        history = pd.DatetimeIndex([pd.NaT, history[1]])
    elif damage == "duplicate":
        history = pd.DatetimeIndex([history[0], history[0]])
    elif damage == "unsorted":
        history = history[::-1]
    elif damage == "not_on_axis":
        history = history - pd.Timedelta("1s")
    else:
        history = panel.timestamps[2:4]
    with pytest.raises(ValueError):
        build(panel, [replace(sample, history_times=history)], cutoff)


@pytest.mark.parametrize("name", ["origin_time", "target_end", "target_availability"])
@pytest.mark.parametrize("damage", ["naive", "missing"])
def test_sample_information_times_must_be_present_and_aware(name, damage):
    panel, split, cutoff = fixture()
    sample = split["train"][0]
    value = getattr(sample, name)
    changed = value.tz_localize(None) if damage == "naive" else pd.NaT
    with pytest.raises(ValueError):
        build(panel, [replace(sample, **{name: changed})], cutoff)


@pytest.mark.parametrize("damage", ["mask_shape", "numeric_mask", "eligibility_shape", "numeric_eligibility"])
def test_sample_masks_require_boolean_dtypes_and_correct_axes(damage):
    panel, split, cutoff = fixture()
    sample = split["train"][0]
    if damage == "mask_shape":
        sample = replace(sample, mask=sample.mask[:, :, 0])
    elif damage == "numeric_mask":
        sample = replace(sample, mask=sample.mask.astype(int))
    elif damage == "eligibility_shape":
        sample = replace(sample, eligible_nodes=sample.eligible_nodes[:, None])
    else:
        sample = replace(sample, eligible_nodes=sample.eligible_nodes.astype(int))
    with pytest.raises(ValueError):
        build(panel, [sample], cutoff)


@pytest.mark.parametrize("name, value", [
    ("node_ids", ("STOCK1", "STOCK0")),
    ("feature_names", ("other",)),
])
def test_sample_axes_must_match_the_original_feature_axes(name, value):
    panel, split, cutoff = fixture()
    sample = replace(split["train"][0], **{name: value})
    with pytest.raises(ValueError):
        build(panel, [sample], cutoff)


@pytest.mark.parametrize("damage", ["duplicate", "reversed"])
def test_training_sample_origins_must_be_unique_and_increasing(damage):
    panel, split, cutoff = fixture()
    samples = split["train"]
    samples = [samples[0], samples[0]] if damage == "duplicate" else samples[::-1]
    with pytest.raises(ValueError):
        build(panel, samples, cutoff)
