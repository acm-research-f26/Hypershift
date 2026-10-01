"""A sample transformation preserves forecasting metadata and label semantics."""

from dataclasses import replace

import numpy as np
import pytest

from data.types import ForecastSample
from data.windows import make_forecast_samples
from experiments.splits import split_forecast_samples
from features import pipeline
from features.market import log_returns
from features.transforms import fit_standardizer
from test.test_market_returns import make_panel


if not hasattr(pipeline, "transform_forecast_sample"):
    pytest.skip("Add transform_forecast_sample to features/pipeline.py", allow_module_level=True)


def fixture():
    panel = make_panel([[100, 200], [110, 180], [121, 198], [133.1, 217.8], [146.41, 239.58]])
    sample = list(make_forecast_samples(panel, 2))[0]
    state = fit_standardizer(
        sample.values, sample.mask, np.ones(2, dtype=bool),
        node_ids=sample.node_ids, feature_names=sample.feature_names,
    )
    return sample, state


def without_second_stock(state):
    mean, scale, count, fitted = (array.copy() for array in
                                   (state.mean, state.scale, state.count, state.fitted_mask))
    mean[1] = np.nan
    scale[1] = np.nan
    count[1] = 0
    fitted[1] = False
    return replace(state, mean=mean, scale=scale, count=count, fitted_mask=fitted)


def test_known_transformation_returns_a_new_sample_with_preserved_metadata():
    sample, state = fixture()
    sample = replace(sample, snapshot_id="example-snapshot")
    transformed = pipeline.transform_forecast_sample(sample, state)
    assert isinstance(transformed, ForecastSample) and transformed is not sample
    np.testing.assert_allclose(transformed.values[:, 0, 0], [0, 0], atol=1e-14)
    np.testing.assert_allclose(transformed.values[:, 1, 0], [-1, 1], atol=1e-14)
    assert transformed.mask.all() and transformed.eligible_nodes.all()
    for name in ("origin_time", "node_ids", "feature_names", "target_start", "target_end",
                 "target_availability", "target_units", "snapshot_id"):
        assert getattr(transformed, name) == getattr(sample, name)
    assert transformed.history_times.equals(sample.history_times)
    np.testing.assert_array_equal(transformed.target, sample.target)
    np.testing.assert_array_equal(transformed.target_mask, sample.target_mask)


def test_unfitted_stock_loses_prediction_eligibility_but_retains_label():
    sample, state = fixture()
    transformed = pipeline.transform_forecast_sample(sample, without_second_stock(state))
    np.testing.assert_array_equal(transformed.eligible_nodes, [True, False])
    assert not transformed.mask[:, 1].any() and np.isnan(transformed.values[:, 1]).all()
    np.testing.assert_array_equal(transformed.target, sample.target)
    assert transformed.target_mask.all()


def test_original_stock_ineligibility_cannot_be_promoted_by_scaling():
    sample, state = fixture()
    sample = replace(sample, eligible_nodes=np.array([True, False]))
    transformed = pipeline.transform_forecast_sample(sample, state)
    assert transformed.mask.all()
    np.testing.assert_array_equal(transformed.eligible_nodes, [True, False])


def test_one_missing_transformed_history_entry_invalidates_prediction_stock():
    sample, state = fixture()
    values = sample.values.copy()
    values[0, 0, 0] = np.nan
    sample = replace(sample, values=values)
    transformed = pipeline.transform_forecast_sample(sample, state)
    assert not transformed.mask[0, 0, 0]
    np.testing.assert_array_equal(transformed.eligible_nodes, [False, True])
    assert transformed.target_mask.all()


def test_future_labels_cannot_change_standardized_inputs_or_prediction_eligibility():
    sample, state = fixture()
    missing_labels = replace(sample, target=np.full(2, np.nan), target_mask=np.zeros(2, dtype=bool))
    before = pipeline.transform_forecast_sample(sample, state)
    after = pipeline.transform_forecast_sample(missing_labels, state)
    for name in ("values", "mask", "eligible_nodes"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert np.isnan(after.target).all() and not after.target_mask.any()


def test_sample_with_no_remaining_eligible_stock_is_retained_for_coverage():
    sample, state = fixture()
    sample = replace(sample, eligible_nodes=np.array([False, True]))
    transformed = pipeline.transform_forecast_sample(sample, without_second_stock(state))
    assert isinstance(transformed, ForecastSample)
    assert not transformed.eligible_nodes.any() and transformed.target_mask.all()
    assert transformed.origin_time == sample.origin_time


def test_output_arrays_are_independent_of_raw_sample_and_fitted_state():
    sample, state = fixture()
    names = ("values", "mask", "eligible_nodes", "target", "target_mask")
    originals = [getattr(sample, name).copy() for name in names]
    fitted_before = [array.copy() for array in (state.mean, state.scale, state.count, state.fitted_mask)]
    transformed = pipeline.transform_forecast_sample(sample, state)
    for name in names:
        assert not np.shares_memory(getattr(transformed, name), getattr(sample, name))
    assert not np.shares_memory(transformed.history_times.asi8, sample.history_times.asi8)
    transformed.values[:] = -999
    transformed.mask[:] = False
    transformed.eligible_nodes[:] = False
    transformed.target[:] = -999
    transformed.target_mask[:] = False
    for name, expected in zip(names, originals):
        np.testing.assert_array_equal(getattr(sample, name), expected)
    for actual, expected in zip((state.mean, state.scale, state.count, state.fitted_mask), fitted_before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("damage", ["eligibility_shape", "numeric_eligibility", "target_shape", "target_mask_shape", "numeric_target_mask"])
def test_sample_node_arrays_require_correct_shapes_and_boolean_masks(damage):
    sample, state = fixture()
    if damage == "eligibility_shape":
        sample = replace(sample, eligible_nodes=sample.eligible_nodes[:, None])
    elif damage == "numeric_eligibility":
        sample = replace(sample, eligible_nodes=sample.eligible_nodes.astype(int))
    elif damage == "target_shape":
        sample = replace(sample, target=sample.target[:, None])
    elif damage == "target_mask_shape":
        sample = replace(sample, target_mask=sample.target_mask[:, None])
    else:
        sample = replace(sample, target_mask=sample.target_mask.astype(int))
    with pytest.raises(ValueError):
        pipeline.transform_forecast_sample(sample, state)


@pytest.mark.parametrize("damage", ["empty", "wrong_length"])
def test_sample_must_have_a_nonempty_history_matching_feature_rows(damage):
    sample, state = fixture()
    if damage == "empty":
        sample = replace(sample, history_times=sample.history_times[:0], values=sample.values[:0], mask=sample.mask[:0])
    else:
        sample = replace(sample, history_times=sample.history_times[:1])
    with pytest.raises(ValueError):
        pipeline.transform_forecast_sample(sample, state)


@pytest.mark.parametrize("field, value", [
    ("node_ids", ("STOCK1", "STOCK0")), ("feature_names", ("other",)),
])
def test_saved_state_ordering_is_checked_by_the_array_transform(field, value):
    sample, state = fixture()
    with pytest.raises(ValueError):
        pipeline.transform_forecast_sample(replace(sample, **{field: value}), state)


def test_full_fold_connection_fits_unique_training_history_and_transforms_all_splits():
    step = np.arange(12, dtype=float)
    panel = make_panel(np.column_stack((100 * np.exp(.01 * step ** 2), 200 * np.exp(-.005 * step ** 2))))
    samples = list(make_forecast_samples(panel, 2))
    cutoff = panel.timestamps[4]
    split = split_forecast_samples(
        samples, train_start=panel.timestamps[0], fit_cutoff=cutoff,
        validation_end=panel.timestamps[7], test_end=panel.timestamps[10],
    )
    axes = dict(node_ids=panel.node_ids, feature_names=("log_return",))
    selected = pipeline.training_feature_mask(panel.timestamps, split["train"], fit_cutoff=cutoff, **axes)
    returns, valid = log_returns(panel)
    state = fit_standardizer(returns[:, :, None], valid[:, :, None] & selected,
                            selected.any(axis=(1, 2)), **axes)
    np.testing.assert_array_equal(state.count, [[3], [3]])
    transformed = {name: [pipeline.transform_forecast_sample(sample, state) for sample in group]
                   for name, group in split.items()}
    assert [len(transformed[name]) for name in ("train", "validation", "test")] == [2, 3, 3]
    for name in split:
        for raw, scaled in zip(split[name], transformed[name]):
            assert scaled.origin_time == raw.origin_time
            np.testing.assert_array_equal(scaled.target, raw.target)
            np.testing.assert_array_equal(scaled.target_mask, raw.target_mask)
    first_test = transformed["test"][0]
    assert first_test.values[0, 0, 0] == pytest.approx(4 / np.sqrt(2 / 3))
    # Later observations use the training mean and scale even far from training.
    assert first_test.values[0, 0, 0] > 3
