"""Economic and information-boundary checks for the bounded strategy runner."""
import numpy as np
import pytest

pytest.importorskip('numba')

from experiments.strategy_search.execution import replay
from experiments.strategy_search.features import event_targets, window_sum, moments
from experiments.strategy_search.statistics import daily_rows, scales


def market(t=40, n=3):
    price = np.full((t, n), 100.)
    return price, np.zeros(t, np.int64), np.zeros((t, n), bool), np.zeros(n, np.int64)


def run(base, price=None, fee=0., delay=2, **kw):
    p, sessions, actions, sector = market(*base.shape)
    if price is not None:
        p = price
    return replay(base, p, p, sessions, actions, sector, False, np.ones(1), fee, .03, delay, **kw)


def test_delayed_entry_and_exit_hold_shares_without_minute_reranking():
    base = np.zeros((40, 3))
    base[1:11, 0] = .04
    path, closed, duration, failures, breaches = run(base)
    assert path[:3, 10].sum() == 0
    assert path[3, 10] == 1
    assert path[4:13, 10].sum() == 0
    assert path[13, 10] == 1
    assert path[:, 10].sum() == 2
    assert abs(duration / closed - 10) < 1e-10
    assert failures == breaches == 0


def test_hand_calculated_cost_funded_round_trip():
    base = np.zeros((40, 3))
    base[1:11, 0] = .04
    path, *_ = run(base, fee=10.)
    expected = 1 - .00008 / 1.00004
    np.testing.assert_allclose(path[-1, 1], expected, rtol=0, atol=1e-13)
    assert path[:, 8].sum() > 0


def test_overlapping_pending_entries_do_not_cancel_an_unfilled_sleeve():
    base = np.zeros((40, 3))
    base[1:15, 0] = .04
    base[2:15, 1] = .04
    path, *_ = run(base)
    np.testing.assert_allclose(path[5, 3], .08)


def test_five_minute_delay_and_closing_flattening():
    base = np.full((40, 3), .04)
    path, *_ = run(base, delay=5)
    assert path[:5, 10].sum() == 0
    assert path[5, 10] == 3
    assert path[-1, 3] == 0


def test_unavailable_liquidation_is_carried_and_invalidated():
    base = np.zeros((40, 3))
    base[:, 0] = .04
    price = np.full_like(base, 100.)
    price[-7:, 0] = np.nan
    path, _, _, failures, _ = run(base, price=price)
    assert failures > 0 and path[-1, 14] > 0
    assert path[-1, 3] > 0


def test_total_daily_turnover_cap_includes_liquidation():
    base = np.full((80, 20), .04)
    path, _, _, failures, breaches = run(base, inventory=True, day_cap=.3, partial=.5, interval=5)
    assert path[:, 7].sum() <= .3 + 1e-8
    assert path[-1, 3] == 0
    assert failures == breaches == 0


def test_future_prices_cannot_change_earlier_orders_or_nav():
    base = np.zeros((40, 3))
    base[1:30, 0] = .04
    price = np.full_like(base, 100.)
    altered = price.copy()
    altered[20:] *= 2
    a, *_ = run(base, price=price)
    b, *_ = run(base, price=altered)
    np.testing.assert_array_equal(a[:20], b[:20])


def test_windows_reset_and_future_perturbations_cannot_change_features():
    values = np.arange(24., dtype=float).reshape(12, 2)
    valid = np.ones_like(values, bool)
    sessions = np.repeat(np.arange(2), 6)
    a, ok = window_sum(values, valid, sessions, 3)
    altered = values.copy()
    altered[8:] = -999
    b, _ = window_sum(altered, valid, sessions, 3)
    np.testing.assert_array_equal(a[:8], b[:8])
    assert not ok[6:8].any()
    np.testing.assert_array_equal(a[8], values[6:9].sum(0))


def test_fit_statistics_exclude_development_values():
    values = np.arange(400., dtype=float).reshape(200, 2)
    valid = np.ones_like(values, bool)
    train = np.arange(200) < 120
    first = moments(values, valid, train)
    values[120:] = 1e20
    second = moments(values, valid, train)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_sparse_suppressed_scores_never_fill_rank_tails():
    score = np.zeros((10, 10))
    eligible = np.ones_like(score, bool)
    base, events = event_targets(score, eligible, np.zeros_like(eligible), np.zeros_like(eligible),
                                np.full_like(score, .01), np.zeros(10, np.int64), np.zeros(10, np.int64),
                                np.zeros(10, np.int64), True, 5, 10, 0, False, 1, False, np.ones(10, bool))
    assert events == 0
    assert not base.any()


def test_risk_scaling_uses_only_preceding_daily_returns():
    daily = np.zeros((40, 7))
    daily[:, 1] = np.sin(np.arange(40)) * .01
    a = scales(daily)
    daily[30:, 1] = .5
    b = scales(daily)
    np.testing.assert_array_equal(a[:31], b[:31])
    assert not a[:20].any()


def test_higher_fee_reduces_flat_price_round_trip_nav():
    base = np.zeros((40, 3))
    base[1:11, 0] = .04
    low, *_ = run(base, fee=.01)
    high, *_ = run(base, fee=1.)
    assert high[-1, 1] < low[-1, 1]
