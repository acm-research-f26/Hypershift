"""Chronological splits for prepared forecast samples."""
import pandas as pd


def split_forecast_samples(samples, *, train_start, fit_cutoff, validation_end, test_end):
    """Split ordered samples into half-open prediction-origin intervals."""
    bounds = [
        pd.Timestamp(value)
        for value in (train_start, fit_cutoff, validation_end, test_end)
    ]
    if any(pd.isna(time) or time.tzinfo is None for time in bounds):
        raise ValueError("Split boundaries must be present and timezone-aware")
    train_start, fit_cutoff, validation_end, test_end = (
        time.tz_convert("UTC") for time in bounds
    )
    if not train_start < fit_cutoff < validation_end < test_end:
        raise ValueError("Split boundaries must be strictly increasing")

    splits = {"train": [], "validation": [], "test": []}
    previous = None
    for sample in samples:
        origin, end, known = (pd.Timestamp(value) for value in (sample.origin_time, sample.target_end, sample.target_availability))
        if any(pd.isna(time) or time.tzinfo is None for time in (origin, end, known)):
            raise ValueError("Sample times must be present and timezone-aware")
        if not origin < end <= known:
            raise ValueError("A future target must end after prediction and before availability")
        if previous is not None and origin <= previous:
            raise ValueError("Samples must have unique, increasing prediction times")
        previous = origin

        if train_start <= origin < fit_cutoff:
            # A training label must already be known when fitting starts.
            usable = sample.eligible_nodes & sample.target_mask
            if known <= fit_cutoff and usable.any():
                splits["train"].append(sample)
        elif fit_cutoff <= origin < validation_end:
            # Validation results must be available before test-time selection.
            if known <= validation_end:
                splits["validation"].append(sample)
        elif validation_end <= origin < test_end:
            # Test labels are used later to score already-issued predictions.
            splits["test"].append(sample)

    return splits

def make_walk_forward_folds(
    *,
    train_start,
    fit_cutoff,
    validation_end,
    test_end,
    final_test_end,
    train_policy
):
    """Return complete folds advancing by one elapsed test-block duration."""
    bounds = [
        pd.Timestamp(value)
        for value in (train_start, fit_cutoff, validation_end, test_end, final_test_end)
    ]
    if any(pd.isna(time) or time.tzinfo is None for time in bounds):
        raise ValueError("Fold boundaries must be present and timezone-aware")

    train_start, fit_cutoff, validation_end, test_end, final_test_end = (
        time.tz_convert("UTC") for time in bounds
    )
    if not train_start < fit_cutoff < validation_end < test_end <= final_test_end:
        raise ValueError(
            "Require train_start < fit_cutoff < validation_end "
            "< test_end <= final_test_end"
        )
    if train_policy not in ("expanding", "rolling"):
        raise ValueError("train_policy must be 'expanding' or 'rolling'")

    step = test_end - validation_end
    folds = []
    while True:
        folds.append({
            "train_start": train_start,
            "fit_cutoff": fit_cutoff,
            "validation_end": validation_end,
            "test_end": test_end,
        })

        # Stop before creating a shortened or out-of-range test block.
        if final_test_end - test_end < step:
            break

        fit_cutoff += step
        validation_end += step
        test_end += step
        if train_policy == "rolling": # ts only moves for rolling history
            train_start += step

    return folds

