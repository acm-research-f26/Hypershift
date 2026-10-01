"""Walk-forward schedules advance complete test blocks with explicit history policy."""

import pandas as pd
import pytest

from experiments import splits as splits_module
from test.test_forecast_splits import fixture_samples


if not hasattr(splits_module, "make_walk_forward_folds"):
    pytest.skip("Add make_walk_forward_folds to experiments/splits.py", allow_module_level=True)


def schedule_arguments():
    start = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    return dict(
        train_start=start,
        fit_cutoff=start + pd.Timedelta("60min"),
        validation_end=start + pd.Timedelta("90min"),
        test_end=start + pd.Timedelta("120min"),
        final_test_end=start + pd.Timedelta("210min"),
        train_policy="expanding",
    )


def build(**changes):
    return splits_module.make_walk_forward_folds(**dict(schedule_arguments(), **changes))


def test_expanding_training_keeps_start_and_advances_cutoffs():
    arguments = schedule_arguments()
    folds = build()
    assert len(folds) == 4
    for number, fold in enumerate(folds):
        shift = number * pd.Timedelta("30min")
        assert list(fold) == ["train_start", "fit_cutoff", "validation_end", "test_end"]
        assert fold["train_start"] == arguments["train_start"]
        for name in ("fit_cutoff", "validation_end", "test_end"):
            assert fold[name] == arguments[name] + shift


def test_rolling_training_keeps_initial_elapsed_duration():
    arguments = schedule_arguments()
    folds = build(train_policy="rolling")
    assert len(folds) == 4
    for number, fold in enumerate(folds):
        shift = number * pd.Timedelta("30min")
        for name in ("train_start", "fit_cutoff", "validation_end", "test_end"):
            assert fold[name] == arguments[name] + shift
        assert fold["fit_cutoff"] - fold["train_start"] == pd.Timedelta("60min")


@pytest.mark.parametrize("policy", ["expanding", "rolling"])
def test_test_blocks_are_contiguous_and_half_open(policy):
    folds = build(train_policy=policy)
    for before, after in zip(folds, folds[1:]):
        assert before["test_end"] == after["validation_end"]
        boundary = before["test_end"]
        assert not before["validation_end"] <= boundary < before["test_end"]
        assert after["validation_end"] <= boundary < after["test_end"]


def test_incomplete_final_block_is_omitted_without_shortening():
    arguments = schedule_arguments()
    stop = arguments["final_test_end"] - pd.Timedelta("10min")
    folds = build(final_test_end=stop)
    assert len(folds) == 3
    assert folds[-1]["test_end"] == arguments["test_end"] + pd.Timedelta("60min")
    assert all(fold["test_end"] - fold["validation_end"] == pd.Timedelta("30min") for fold in folds)
    assert folds[-1]["test_end"] < stop


def test_initial_test_end_equal_to_final_end_produces_one_fold():
    arguments = schedule_arguments()
    folds = build(final_test_end=arguments["test_end"])
    assert folds == [{name: arguments[name] for name in ("train_start", "fit_cutoff", "validation_end", "test_end")}]


def test_test_duration_controls_advance_independently_of_validation_duration():
    arguments = schedule_arguments()
    folds = build(validation_end=arguments["fit_cutoff"] + pd.Timedelta("15min"))
    assert len(folds) == 3
    assert all(fold["validation_end"] - fold["fit_cutoff"] == pd.Timedelta("15min") for fold in folds)
    assert all(fold["test_end"] - fold["validation_end"] == pd.Timedelta("45min") for fold in folds)
    assert folds[1]["fit_cutoff"] - folds[0]["fit_cutoff"] == pd.Timedelta("45min")


def test_timezone_conversion_and_aware_strings_give_utc_boundaries():
    arguments = schedule_arguments()
    alternate = {
        name: value.tz_convert("America/New_York").isoformat() if isinstance(value, pd.Timestamp) else value
        for name, value in arguments.items()
    }
    folds = splits_module.make_walk_forward_folds(**alternate)
    assert folds == build()
    assert all(str(value.tz) == "UTC" for fold in folds for value in fold.values())


def test_clock_day_is_elapsed_24_hours_even_across_daylight_saving():
    start = pd.Timestamp("2024-03-08 09:30", tz="America/New_York")
    folds = splits_module.make_walk_forward_folds(
        train_start=start,
        fit_cutoff=start + pd.Timedelta("1D"),
        validation_end=start + pd.Timedelta("2D"),
        test_end=start + pd.Timedelta("3D"),
        final_test_end=start + pd.Timedelta("4D"),
        train_policy="rolling",
    )
    assert len(folds) == 2
    assert folds[1]["fit_cutoff"] - folds[0]["fit_cutoff"] == pd.Timedelta("24h")
    assert folds[1]["train_start"].tz_convert("America/New_York").hour == 9
    assert folds[1]["fit_cutoff"].tz_convert("America/New_York").hour == 10


@pytest.mark.parametrize("policy", ["expanding", "rolling"])
def test_each_fold_reuses_split_information_deadlines(policy):
    panel, samples, boundaries = fixture_samples()
    folds = splits_module.make_walk_forward_folds(
        **boundaries,
        final_test_end=boundaries["test_end"] + pd.Timedelta("45min"),
        train_policy=policy,
    )
    assert len(folds) == 2
    groups = [splits_module.split_forecast_samples(samples, **fold) for fold in folds]
    assert [sample.target_start for sample in groups[0]["test"]] == list(panel.timestamps[[7, 8, 9]])
    assert [sample.target_start for sample in groups[1]["test"]] == list(panel.timestamps[[10]])
    first_origins = {sample.origin_time for sample in groups[0]["test"]}
    second_origins = {sample.origin_time for sample in groups[1]["test"]}
    assert first_origins.isdisjoint(second_origins)
    for fold, split in zip(folds, groups):
        assert split["train"]
        assert all(fold["train_start"] <= sample.origin_time < fold["fit_cutoff"] for sample in split["train"])
        assert all(sample.target_availability <= fold["fit_cutoff"] for sample in split["train"])
        assert all(sample.target_availability <= fold["validation_end"] for sample in split["validation"])
    assert not any(sample is samples[1] for sample in groups[0]["train"])
    assert any(sample is samples[1] for sample in groups[1]["train"])


def test_fold_dictionaries_are_independent():
    folds = build()
    second_start = folds[1]["train_start"]
    folds[0]["train_start"] = folds[0]["train_start"] - pd.Timedelta("1h")
    assert folds[1]["train_start"] == second_start


@pytest.mark.parametrize("name", ["train_start", "fit_cutoff", "validation_end", "test_end", "final_test_end"])
@pytest.mark.parametrize("damage", ["naive", "missing"])
def test_missing_or_naive_boundaries_are_rejected(name, damage):
    value = schedule_arguments()[name]
    changed = value.tz_localize(None) if damage == "naive" else pd.NaT
    with pytest.raises(ValueError):
        build(**{name: changed})


@pytest.mark.parametrize("name, previous", [
    ("fit_cutoff", "train_start"),
    ("validation_end", "fit_cutoff"),
    ("test_end", "validation_end"),
])
@pytest.mark.parametrize("change", ["equal", "earlier"])
def test_invalid_initial_order_is_rejected(name, previous, change):
    value = schedule_arguments()[previous]
    if change == "earlier":
        value -= pd.Timedelta("1min")
    with pytest.raises(ValueError):
        build(**{name: value})


def test_final_end_before_first_complete_test_block_is_rejected():
    with pytest.raises(ValueError):
        build(final_test_end=schedule_arguments()["test_end"] - pd.Timedelta("1s"))


@pytest.mark.parametrize("policy", ["fixed", "Expanding", "", None])
def test_unknown_training_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        build(train_policy=policy)


def test_training_policy_must_be_supplied_explicitly():
    arguments = schedule_arguments()
    del arguments["train_policy"]
    with pytest.raises(TypeError):
        splits_module.make_walk_forward_folds(**arguments)


@pytest.mark.parametrize("policy", ["expanding", "rolling"])
def test_extending_horizon_preserves_every_previous_fold(policy):
    original = build(train_policy=policy)
    stop = schedule_arguments()["final_test_end"]
    partial_extension = build(train_policy=policy, final_test_end=stop + pd.Timedelta("29min"))
    full_extension = build(train_policy=policy, final_test_end=stop + pd.Timedelta("30min"))
    assert partial_extension == original
    assert full_extension[:-1] == original
    assert full_extension[-1]["test_end"] == stop + pd.Timedelta("30min")


@pytest.mark.parametrize("policy", ["expanding", "rolling"])
def test_boundaries_do_not_imply_observations_exist_in_a_fold(policy):
    _, samples, arguments = fixture_samples()
    later = {name: value + pd.Timedelta("10D") for name, value in arguments.items()}
    folds = splits_module.make_walk_forward_folds(
        **later, final_test_end=later["test_end"], train_policy=policy,
    )
    assert len(folds) == 1
    assert splits_module.split_forecast_samples(samples, **folds[0]) == {
        "train": [], "validation": [], "test": [],
    }


@pytest.mark.parametrize("policy", ["expanding", "rolling"])
def test_old_test_observations_can_train_later_only_when_labels_are_known(policy):
    panel, samples, arguments = fixture_samples()
    folds = splits_module.make_walk_forward_folds(
        **arguments,
        final_test_end=arguments["test_end"] + pd.Timedelta("90min"),
        train_policy=policy,
    )
    first = splits_module.split_forecast_samples(samples, **folds[0])
    third = splits_module.split_forecast_samples(samples, **folds[2])
    assert [sample.target_start for sample in first["test"]] == list(panel.timestamps[[7, 8, 9]])
    trained_origins = {sample.target_start for sample in third["train"]}
    assert set(panel.timestamps[[7, 8]]).issubset(trained_origins)
    # The last old test label arrives 30 seconds after this fold's fit cutoff.
    assert panel.timestamps[9] not in trained_origins
    if policy == "rolling":
        assert panel.timestamps[2] not in trained_origins
        assert folds[2]["train_start"] == panel.timestamps[6]
    else:
        assert panel.timestamps[2] in trained_origins


def test_boundaries_in_different_timezones_still_describe_one_schedule():
    arguments = schedule_arguments()
    names = ("train_start", "fit_cutoff", "validation_end", "test_end", "final_test_end")
    zones = ("America/New_York", "Asia/Tokyo", "Europe/London", "UTC", "America/Chicago")
    for name, zone in zip(names, zones):
        arguments[name] = arguments[name].tz_convert(zone)
    assert splits_module.make_walk_forward_folds(**arguments) == build()
