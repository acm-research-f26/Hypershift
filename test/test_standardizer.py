"""Training-only per-node, per-feature population standardization."""

import importlib

import numpy as np
import pytest


try:
    transforms = importlib.import_module("features.transforms")
except ModuleNotFoundError as exc:
    if exc.name != "features.transforms":
        raise
    pytest.skip("Add features/transforms.py with fit_standardizer", allow_module_level=True)


def fit(values, mask=None, fit_window=None, **axes):
    values = np.asarray(values, dtype=float)
    mask = np.ones(values.shape, dtype=bool) if mask is None else mask
    fit_window = np.ones(len(values), dtype=bool) if fit_window is None else fit_window
    arguments = dict(
        node_ids=tuple(f"NODE{index}" for index in range(values.shape[1])),
        feature_names=tuple(f"FEATURE{index}" for index in range(values.shape[2])),
    )
    arguments.update(axes)
    return transforms.fit_standardizer(values, mask, fit_window, **arguments)


def test_known_population_statistics_are_separate_for_every_node_and_feature():
    values = np.array([[[1, 10], [100, -4]], [[3, 14], [104, 0]], [[5, 18], [108, 4]]], dtype=float)
    state = fit(values, node_ids=("B", "A"), feature_names=("return", "volume"))
    np.testing.assert_allclose(state.mean, [[3, 14], [104, 0]])
    np.testing.assert_allclose(state.scale, np.sqrt([[8 / 3, 32 / 3], [32 / 3, 32 / 3]]))
    np.testing.assert_array_equal(state.count, np.full((2, 2), 3))
    assert state.fitted_mask.shape == (2, 2) and state.fitted_mask.all()
    assert state.fitted_mask.dtype.kind == "b"
    assert state.node_ids == ("B", "A") and state.feature_names == ("return", "volume")


def test_only_selected_training_rows_affect_fitted_statistics():
    values = np.array([1, 999, 3, -999], dtype=float)[:, None, None]
    selection = np.array([True, False, True, False])
    state = fit(values, fit_window=selection)
    assert state.mean[0, 0] == pytest.approx(2)
    assert state.scale[0, 0] == pytest.approx(1)
    assert state.count[0, 0] == 2
    changed = values.copy()
    changed[~selection] = 1_000_000
    after = fit(changed, fit_window=selection)
    for name in ("mean", "scale", "count", "fitted_mask"):
        np.testing.assert_array_equal(getattr(state, name), getattr(after, name))


def test_masked_finite_outlier_does_not_affect_mean_or_scale():
    values = np.array([1, 100_000, 3], dtype=float)[:, None, None]
    state = fit(values, mask=np.array([True, False, True])[:, None, None])
    np.testing.assert_allclose(state.mean, [[2]])
    np.testing.assert_allclose(state.scale, [[1]])
    np.testing.assert_array_equal(state.count, [[2]])


def test_nonfinite_values_are_excluded_even_if_input_mask_is_true():
    values = np.array([1, np.nan, np.inf, -np.inf, 3])[:, None, None]
    with np.errstate(all="raise"):
        state = fit(values)
    np.testing.assert_allclose(state.mean, [[2]])
    np.testing.assert_allclose(state.scale, [[1]])
    np.testing.assert_array_equal(state.count, [[2]])


def test_constant_training_feature_gets_unit_denominator():
    state = fit(np.array([7, 7, 7])[:, None, None])
    assert state.mean[0, 0] == 7 and state.scale[0, 0] == 1
    assert state.fitted_mask[0, 0] and state.count[0, 0] == 3


def test_missing_or_single_observation_coordinates_remain_unfitted():
    values = np.array([[[1], [99], [np.nan]], [[3], [np.nan], [np.nan]]])
    with np.errstate(all="raise"):
        state = fit(values)
    np.testing.assert_array_equal(state.count[:, 0], [2, 1, 0])
    np.testing.assert_array_equal(state.fitted_mask[:, 0], [True, False, False])
    assert state.mean[0, 0] == 2 and state.scale[0, 0] == 1
    assert np.isnan(state.mean[1:]).all() and np.isnan(state.scale[1:]).all()


@pytest.mark.parametrize("damage", ["empty_time", "no_training_rows", "no_eligible_values", "one_observation"])
def test_no_fittable_coordinate_fails_explicitly(damage):
    values = np.ones((2, 1, 1))
    mask = np.ones(values.shape, dtype=bool)
    selection = np.ones(2, dtype=bool)
    if damage == "empty_time":
        values, mask, selection = values[:0], mask[:0], selection[:0]
    elif damage == "no_training_rows":
        selection[:] = False
    elif damage == "no_eligible_values":
        mask[:] = False
    else:
        selection[1] = False
    with pytest.raises(ValueError):
        fit(values, mask, selection)


def test_fit_does_not_mutate_or_retain_views_of_input_arrays():
    values = np.array([1, 3, 100], dtype=float)[:, None, None]
    mask = np.ones(values.shape, dtype=bool)
    selection = np.array([True, True, False])
    originals = [array.copy() for array in (values, mask, selection)]
    state = fit(values, mask, selection)
    for actual, expected in zip((values, mask, selection), originals):
        np.testing.assert_array_equal(actual, expected)
    values[:] = 999
    mask[:] = False
    selection[:] = False
    assert state.mean[0, 0] == 2 and state.scale[0, 0] == 1


@pytest.mark.parametrize("damage", ["mask_shape", "numeric_mask", "window_shape", "numeric_window"])
def test_masks_must_have_boolean_dtype_and_exact_shapes(damage):
    values = np.ones((3, 2, 1))
    mask = np.ones(values.shape, dtype=bool)
    selection = np.ones(3, dtype=bool)
    if damage == "mask_shape":
        mask = mask[:, :, 0]
    elif damage == "numeric_mask":
        mask = mask.astype(int)
    elif damage == "window_shape":
        selection = selection[:, None]
    else:
        selection = selection.astype(int)
    with pytest.raises(ValueError):
        fit(values, mask, selection)


@pytest.mark.parametrize("shape", [(3, 2), (3, 2, 1, 1), (3, 0, 1), (3, 2, 0)])
def test_values_require_three_dimensions_and_nonempty_node_feature_axes(shape):
    values = np.ones(shape)
    with pytest.raises(ValueError):
        transforms.fit_standardizer(
            values, np.ones(shape, dtype=bool), np.ones(3, dtype=bool),
            node_ids=("A", "B"), feature_names=("return",),
        )


@pytest.mark.parametrize("axes", [
    {"node_ids": ("A",)},
    {"feature_names": ("x",)},
    {"node_ids": ("A", "A")},
    {"feature_names": ("x", "x")},
])
def test_axis_identity_must_be_unique_and_match_values(axes):
    with pytest.raises(ValueError):
        fit(np.ones((3, 2, 2)), **axes)
