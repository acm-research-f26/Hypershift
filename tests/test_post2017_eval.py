"""Plan Task 4: locked 2018-2023 inference panel. Synthetic only (no network, no real data)."""
import numpy as np
import pandas as pd
import pytest

from hypershift.data.alpaca_panel import (assert_frozen_order, build_eval_panel, daily_tie_flags, target_dates,
                                          topk_names)
from hypershift.data.rsr import FILL
from hypershift.train.loop import gather_batch, window_offsets

CAL = pd.bdate_range("2016-01-04", periods=120)
ORDER = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
TEST0 = CAL[80]
SCALE_END = CAL[60]


def closes(seed, n=len(CAL)):
    r = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + 0.01 * r.standard_normal(n)), index=CAL)


def panel(series, order=ORDER, **kw):
    return build_eval_panel(series, order, ORDER, CAL, TEST0, SCALE_END, **kw)


def test_date_alignment_and_returns():
    s = {t: closes(i) for i, t in enumerate(ORDER)}
    d = panel(s)
    assert d.tickers == ORDER
    assert d.test_index == 80 and pd.Timestamp(d.timestamps[80]) == TEST0
    k, t = 2, 95
    assert d.gt[k, t] == pytest.approx(s["CCC"].iloc[t] / s["CCC"].iloc[t - 1] - 1, rel=1e-5)
    assert (target_dates(d) == CAL[80:].values).all()
    offs = window_offsets(d, 16, "test")
    _, m, _, g = gather_batch(d, offs[:1], 16)
    assert offs[0] + 16 == 80 and g[0, k] == pytest.approx(d.gt[k, 80])


def test_permuted_tickers_rejected():
    s = {t: closes(i) for i, t in enumerate(ORDER)}
    with pytest.raises(ValueError):
        panel(s, order=ORDER[::-1])
    with pytest.raises(ValueError):
        assert_frozen_order(ORDER[:-1], ORDER)
    assert_frozen_order(list(ORDER), ORDER)


def test_missing_date_masked_not_filled():
    s = {t: closes(i) for i, t in enumerate(ORDER)}
    s["BBB"] = s["BBB"].drop(CAL[90])
    d = panel(s)
    assert d.mask[1, 90] == 0 and (d.features[1, 90] == FILL).all()
    assert d.gt[1, 90] == 0 and d.gt[1, 91] == 0          # neither side of the gap is a real return
    assert d.mask[1, 91] == 1 and d.mask[1, 89] == 1
    offs = np.array([75, 91])                              # window 75..91 contains the gap, window 91..107 does not
    _, m, _, _ = gather_batch(d, offs, 16)
    assert m[0, 1] == 0 and m[1, 1] == 1 and m[0, 0] == 1


def test_delisted_masked_after_last_bar_never_zero_filled():
    s = {t: closes(i) for i, t in enumerate(ORDER)}
    s["DDD"] = s["DDD"].iloc[:96]                          # last bar CAL[95]
    d = panel(s)
    assert (d.mask[3, 96:] == 0).all() and d.mask[3, 95] == 1
    assert (d.features[3, 96:] == FILL).all()
    pred = np.zeros((6, 5))
    pred[3] = 9.0                                          # delisted name has the best score every day
    mask = d.mask[:, 94:99]
    assert 3 not in topk_names(pred[:, 2], mask[:, 2], 5)  # day 96: masked, never selected
    assert 3 in topk_names(pred[:, 0], mask[:, 0], 5)      # day 94: observed, selected


def test_identity_failed_or_absent_ticker_keeps_slot_fully_masked():
    s = {t: closes(i) for i, t in enumerate(ORDER) if t != "EEE"}
    d = panel(s)
    assert d.tickers[4] == "EEE" and (d.mask[4] == 0).all()


def test_reappearance_after_long_gap_truncated():
    s = {t: closes(i) for i, t in enumerate(ORDER)}
    s["FFF"] = s["FFF"].drop(CAL[85:100])                  # 15-day gap
    d = panel(s, gap_days=10, gap_from=CAL[60])
    assert (d.mask[5, 85:] == 0).all()                     # cut from the start of the gap
    d2 = panel(s)                                          # default 60-day rule: gap too short, stays
    assert d2.mask[5, 110] == 1


def test_tie_flags():
    pred = np.array([[3., 3., 1.], [2., 2., 1.], [1., 2., 1.], [0.5, 1., 1.]])
    f = daily_tie_flags(pred, np.ones_like(pred), 2)
    assert f.tolist() == [False, True, True]
    pred2 = np.array([[1., 1.], [1., 1.], [1., 1.]])
    assert daily_tie_flags(pred2, np.ones_like(pred2), 2).tolist() == [True, True]
