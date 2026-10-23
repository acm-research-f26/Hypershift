import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from hypershift.strategy import engine as EN
from hypershift.strategy import pit_engine as PE
from hypershift.data import pit_sp500 as PS


def _data(N=8, T=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    open_ = np.concatenate([close[:, :1], close[:, :-1] * (1 + .002 * rng.normal(size=(N, T - 1)))], 1)
    return SimpleNamespace(close=close, open=open_, observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                           beta=np.ones((N, T)), sectors=np.zeros(N, int), dates=np.arange('2020-01-01', T, dtype='datetime64[D]'))


def _scores(N, days, seed=1):
    rng = np.random.default_rng(seed)
    return {t: rng.normal(size=N) for t in days}


@pytest.mark.parametrize('mode', ['LO', 'LS'])
def test_run_pit_equals_engine_when_always_eligible(mode):
    d = _data(N=24, T=80)
    sc = _scores(24, range(20, 80, 5))
    a = EN.run(sc, d, mode=mode, top=.2, band=.4, cost_bp=10, borrow_bp=50)
    b = PE.run_pit(sc, d, mode=mode, top=.2, band=.4, cost_bp=10, borrow_bp=50)
    for k in ('gross', 'traded', 'borrow', 'net'):
        assert np.allclose(a[k], b[k])


def test_run_pit_exits_name_that_left_index():
    d = _data(N=8, T=60)
    sc = {20: np.r_[10., np.zeros(7)], 25: np.r_[10., np.zeros(7)]}
    d.eligible[0, 23:] = False                     # name 0 leaves the index after the first trade
    old = EN.run(sc, d, mode='LO', top=.125, band=.125, trace=True, cost_bp=0)
    new = PE.run_pit(sc, d, mode='LO', top=.125, band=.125, trace=True, cost_bp=0)
    k = 30 - 20
    assert old['w_start'][k][0] > .5               # frozen engine: stuck weight
    assert new['w_start'][k][0] == 0               # PIT engine: sold at the next rebalance


def test_run_pit_liquidates_delisted_at_last_close_with_terminal_return():
    d = _data(N=8, T=60)
    sc = {20: np.r_[10., np.zeros(7)]}
    d.observed[0, 30:] = False
    d.close[0, 30:] = np.nan
    d.open[0, 30:] = np.nan
    base = PE.run_pit(sc, d, mode='LO', top=.125, band=.125, cost_bp=0, trace=True)
    hit = PE.run_pit(sc, d, mode='LO', top=.125, band=.125, cost_bp=0, trace=True, terminal=np.r_[-1., np.zeros(7)])
    k = 30 - 20
    assert base['w_start'][k + 1][0] == 0 and hit['w_start'][k + 1][0] == 0
    assert np.isclose(base['gross'][k], 0) and np.isclose(hit['gross'][k], -base['w_start'][k][0])
    assert base['delisted_weight'][k] > .99


def test_intervals_split_reused_ticker():
    sd = np.array(['2016-01-04', '2017-09-01', '2019-04-02'], dtype='datetime64[D]')
    snaps = [frozenset({'DOW', 'A'}), frozenset({'A', 'DWDP'}), frozenset({'A', 'DWDP', 'DOW'})]
    iv = PS.intervals(sd, snaps, np.datetime64('2016-01-04'), np.datetime64('2023-12-29'))
    dow = sorted((s, e) for t, s, e in iv if t == 'DOW')
    assert dow == [(np.datetime64('2016-01-04'), np.datetime64('2017-09-01')),
                   (np.datetime64('2019-04-02'), np.datetime64('2023-12-30'))]
    assert [(s, e) for t, s, e in iv if t == 'DWDP'] == [(np.datetime64('2017-09-01'), np.datetime64('2023-12-30'))]


def test_group_entities_merges_rescaled_copy_not_share_class():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2020-01-01', periods=300)
    a = pd.Series(100 * np.cumprod(1 + .01 * rng.normal(size=300)), idx)
    b = a.iloc[50:] * .97                           # same company, different adjustment scale, shorter
    c = a * np.cumprod(1 + .003 * rng.normal(size=300))   # sister share class
    g = PS.group_entities({'FB': a, 'META': b, 'GOOG': c})
    assert sorted(map(sorted, g)) == [['FB', 'META'], ['GOOG']]


def test_member_from_intervals():
    dates = np.arange('2020-01-01', '2020-01-11', dtype='datetime64[D]')
    m = PS.member_from_intervals([[(np.datetime64('2020-01-03'), np.datetime64('2020-01-05')),
                                   (np.datetime64('2020-01-08'), np.datetime64('2020-01-09'))]], dates)
    assert m[0].tolist() == [False, False, True, True, False, False, False, True, False, False]


def test_sector_proxy_recovers_blocks_and_leaves_self_out():
    rng = np.random.default_rng(3)
    T, k = 200, 3
    f = rng.normal(0, .01, (k, T))
    sec = np.repeat(np.arange(k), 6)
    ret = f[sec] + rng.normal(0, .004, (len(sec), T))
    ref = sec.copy()
    ref[[0, 7]] = -1                                # unknown sector: must still be assigned
    out = PS.sector_proxy(ret, np.ones_like(ret, bool), ref, window=126)
    assert (out == sec).all()


def test_failure_flags():
    close = np.ones((2, 100)) * 100
    close[0, 80:] = 30                               # collapses before its last bar
    obs = np.ones_like(close, bool)
    last = np.array([99, 99])
    assert PS.failure_flags(close, obs, last).tolist() == [True, False]
