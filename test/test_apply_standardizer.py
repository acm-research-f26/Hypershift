"""Saved training parameters transform later inputs without changing their fit."""

from dataclasses import replace

import numpy as np
import pytest

from features import transforms


if not hasattr(transforms, "apply_standardizer"):
    pytest.skip("Add apply_standardizer to features/transforms.py", allow_module_level=True)


def fixture_state():
    values = np.array([[[1, 10], [100, -4]], [[3, 14], [104, 0]]], dtype=float)
    state = transforms.fit_standardizer(
        values, np.ones(values.shape, dtype=bool), np.ones(2, dtype=bool),
        node_ids=("A", "B"), feature_names=("return", "volume"),
    )
    return values, state


def apply(values, state, mask=None, **axes):
    values = np.asarray(values, dtype=float)
    mask = np.ones(values.shape, dtype=bool) if mask is None else mask
    arguments = dict(node_ids=state.node_ids, feature_names=state.feature_names)
    arguments.update(axes)
    return transforms.apply_standardizer(values, mask, state, **arguments)


def test_saved_training_parameters_give_known_later_z_scores():
    _, state = fixture_state()
    values = np.array([[[4, 16], [98, 2]]], dtype=float)
    result, valid = apply(values, state)
    np.testing.assert_allclose(result, [[[2, 2], [-2, 2]]])
    assert valid.shape == values.shape and valid.dtype.kind == "b" and valid.all()


def test_nonconstant_training_features_have_zero_mean_and_unit_population_scale():
    values, state = fixture_state()
    result, valid = apply(values, state)
    assert valid.all()
    np.testing.assert_allclose(result.mean(axis=0), np.zeros((2, 2)), atol=1e-14)
    np.testing.assert_allclose(result.std(axis=0, ddof=0), np.ones((2, 2)))


def test_later_data_is_not_recentered_or_rescaled_using_itself():
    _, state = fixture_state()
    values = np.array([[[4, 16], [98, 2]], [[8, 20], [110, 6]]], dtype=float)
    before, _ = apply(values, state)
    changed = values.copy()
    changed[1] = 1_000_000
    after, _ = apply(changed, state)
    np.testing.assert_array_equal(before[0], after[0])
    # A single row receives exactly the same transform as it does in a batch.
    alone, _ = apply(values[:1], state)
    np.testing.assert_array_equal(before[:1], alone)


def test_invalid_inputs_stay_missing_and_masks_never_expand():
    _, state = fixture_state()
    values = np.array([[[4, np.inf], [np.nan, 2]]], dtype=float)
    mask = np.array([[[False, True], [True, True]]])
    with np.errstate(all="raise"):
        result, valid = apply(values, state, mask)
    np.testing.assert_array_equal(valid, [[[False, False], [False, True]]])
    assert np.isnan(result[~valid]).all() and result[0, 1, 1] == 2
    assert not (valid & ~mask).any()


def test_all_masked_input_returns_all_missing_without_refitting():
    _, state = fixture_state()
    values = np.full((3, 2, 2), 123.0)
    result, valid = apply(values, state, np.zeros(values.shape, dtype=bool))
    assert not valid.any() and np.isnan(result).all()


def test_unfitted_training_coordinate_cannot_become_valid_from_later_values():
    training = np.array([[[1], [9], [np.nan]], [[3], [np.nan], [np.nan]]])
    state = transforms.fit_standardizer(
        training, np.isfinite(training), np.ones(2, dtype=bool),
        node_ids=("A", "B", "C"), feature_names=("return",),
    )
    with np.errstate(all="raise"):
        result, valid = apply([[[4], [11], [12]]], state)
    np.testing.assert_array_equal(valid, [[[True], [False], [False]]])
    assert result[0, 0, 0] == 2 and np.isnan(result[0, 1:]).all()


def test_constant_training_feature_uses_saved_unit_denominator():
    training = np.full((2, 1, 1), 7.0)
    state = transforms.fit_standardizer(
        training, np.ones(training.shape, dtype=bool), np.ones(2, dtype=bool),
        node_ids=("A",), feature_names=("return",),
    )
    result, valid = apply([[[7]], [[9]]], state)
    np.testing.assert_array_equal(result[:, 0, 0], [0, 2])
    assert valid.all()


def test_inputs_and_fitted_state_are_not_mutated_or_shared_with_outputs():
    values, state = fixture_state()
    mask = np.ones(values.shape, dtype=bool)
    arrays = (values, mask, state.mean, state.scale, state.count, state.fitted_mask)
    original = [array.copy() for array in arrays]
    result, valid = apply(values, state, mask)
    result[:] = 999
    valid[:] = False
    for actual, expected in zip(arrays, original):
        np.testing.assert_array_equal(actual, expected)


def test_empty_time_axis_retains_stock_and_feature_axes():
    _, state = fixture_state()
    result, valid = apply(np.empty((0, 2, 2)), state)
    assert result.shape == valid.shape == (0, 2, 2)


@pytest.mark.parametrize("axes", [
    {"node_ids": ("B", "A")},
    {"node_ids": ("A", "C")},
    {"feature_names": ("volume", "return")},
    {"feature_names": ("return", "other")},
])
def test_stock_and_feature_order_must_match_the_fitted_state(axes):
    values, state = fixture_state()
    with pytest.raises(ValueError):
        apply(values, state, **axes)


@pytest.mark.parametrize("damage", ["numeric_mask", "mask_shape", "values_ndim", "values_axes"])
def test_input_masks_and_values_require_exact_declared_shapes(damage):
    values, state = fixture_state()
    mask = np.ones(values.shape, dtype=bool)
    if damage == "numeric_mask":
        mask = mask.astype(int)
    elif damage == "mask_shape":
        mask = mask[:, :, 0]
    elif damage == "values_ndim":
        values, mask = values[:, :, 0], mask[:, :, 0]
    else:
        values, mask = values[:, :1], mask[:, :1]
    with pytest.raises(ValueError):
        apply(values, state, mask)


@pytest.mark.parametrize("field", ["mean", "scale", "count", "fitted_mask"])
def test_fitted_state_arrays_must_match_its_declared_axes(field):
    values, state = fixture_state()
    damaged = replace(state, **{field: getattr(state, field)[:, :1]})
    with pytest.raises(ValueError):
        apply(values, damaged)


@pytest.mark.parametrize("damage", ["numeric_fitted_mask", "fractional_count", "negative_count", "contradictory_count"])
def test_fitted_state_counts_and_flags_must_be_consistent(damage):
    values, state = fixture_state()
    if damage == "numeric_fitted_mask":
        state = replace(state, fitted_mask=state.fitted_mask.astype(int))
    elif damage == "fractional_count":
        state = replace(state, count=state.count.astype(float))
    else:
        count = state.count.copy()
        count[0, 0] = -1 if damage == "negative_count" else 1
        state = replace(state, count=count)
    with pytest.raises(ValueError):
        apply(values, state)


@pytest.mark.parametrize("field, value", [
    ("scale", 0), ("scale", -1), ("scale", np.inf), ("scale", np.nan),
    ("mean", np.inf), ("mean", np.nan),
])
def test_fitted_statistics_must_be_finite_with_positive_denominators(field, value):
    values, state = fixture_state()
    array = getattr(state, field).copy()
    array[0, 0] = value
    with pytest.raises(ValueError):
        apply(values, replace(state, **{field: array}))


def test_nonfinite_arithmetic_result_is_masked_without_an_infinite_feature():
    _, state = fixture_state()
    mean = state.mean.copy()
    mean[0, 0] = -1e308
    state = replace(state, mean=mean)
    with np.errstate(all="raise"):
        result, valid = apply([[[1e308, 12], [102, -2]]], state)
    assert not valid[0, 0, 0] and np.isnan(result[0, 0, 0])
    assert valid.sum() == 3 and np.isfinite(result[valid]).all()
