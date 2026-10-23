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


def _toy_pit_fb(N=48, T=400, seed=7):
    rng = np.random.default_rng(seed)
    dates = np.busday_offset('2017-06-01', np.arange(T), roll='forward')
    return SimpleNamespace(X=rng.normal(size=(N, T, 20)).astype(np.float32), y=rng.normal(0, .02, (N, T)),
                           y_ok=np.ones((N, T), bool), eligible=np.ones((N, T), bool), beta=rng.normal(1, .2, (N, T)),
                           sectors=np.repeat(np.arange(4), N // 4), horizon=5, dates=dates)


def test_train_signals_only_matured_and_pit_eligible():
    from hypershift.strategy import pit as SP
    fb = _toy_pit_fb()
    fb.eligible[:, 100:120] = False
    fb.eligible[:5, 100:120] = True                   # too few eligible names on those dates
    tr = SP.train_signals(fb, asof=300)
    assert tr.max() + fb.horizon <= 300 and tr.min() >= 63
    assert not np.isin(np.arange(100, 120), tr).any()


def test_annual_scores_shape_and_eligibility():
    from hypershift.strategy import pit as SP
    fb = _toy_pit_fb()
    fb.eligible[3, :] = False
    sc = SP.annual_scores(fb, 'ridge', years=(2018,))
    days = sorted(sc)
    assert days[0] - 1 == int(np.searchsorted(fb.dates, np.datetime64('2018-01-01')))
    assert all(b - a == 5 for a, b in zip(days, days[1:]))
    assert all(fb.dates[d - 1].astype('datetime64[Y]') == np.datetime64('2018', 'Y') for d in days)
    assert np.isnan(sc[days[0]][3]) and np.isfinite(np.delete(sc[days[0]], 3)).all()


def test_clean_returns_rules():
    obs = np.ones((1, 12), bool)
    raw_c = np.array([[100., 100, 65, 66, 66, 33, 33, 25, 25, 26, 6.5, 6.6]])
    raw_o = np.array([[100., 100, 64, 66, 66, 33.2, 33, 25.5, 25, 25.5, 6.4, 6.6]])
    # Alpaca factors: a spin recorded as a 1:3 reverse split before day 2 (absurd), a genuine 4:1 split before day 10
    all_c = raw_c * np.where(np.arange(12) < 10, .25, 1.)[None] * np.where(np.arange(12) < 2, 3.06, 1.)[None]
    r, kind = PS.clean_returns(raw_c, raw_o, all_c, obs)
    assert kind[0, 2] == 'neutralised' and r[0, 2] == 0                         # bad adjustment -> value-neutral day
    assert kind[0, 5] == 'neutralised' and r[0, 5] == 0                         # unflagged 1:1 distribution
    assert kind[0, 7] == 'raw' and np.isclose(r[0, 7], 25 / 33 - 1)              # -24% gap, no flag: kept
    assert kind[0, 10] == 'adjusted' and np.isclose(r[0, 10], all_c[0, 10] / all_c[0, 9] - 1)   # genuine split
    assert kind[0, 11] == 'raw' and np.isclose(r[0, 11], 6.6 / 6.5 - 1)


def test_clean_returns_keeps_crash_near_two_thirds():
    obs = np.ones((1, 3), bool)
    raw_c = np.array([[267.83, 267.83, 106.04]])          # SIVB 2023-03-09: gap to 176.55 is 0.659 of the close
    raw_o = np.array([[267.83, 267.83, 176.55]])
    r, kind = PS.clean_returns(raw_c, raw_o, raw_c.copy(), obs)
    assert kind[0, 2] == 'raw' and np.isclose(r[0, 2], 106.04 / 267.83 - 1)


def test_clean_returns_rejects_adjustment_without_price_event_or_overcorrection():
    obs = np.ones((2, 3), bool)
    raw_c = np.array([[50., 50, 50], [40., 40, 31.6]])     # IR 2020-03-02: no raw move; XRX 2017-01-03: -21% spin
    raw_o = raw_c.copy()
    all_c = raw_c * np.array([[1 / .882, 1 / .882, 1.], [1 / 1.518, 1 / 1.518, 1.]])
    r, kind = PS.clean_returns(raw_c, raw_o, all_c, obs)
    assert kind[0, 2] == 'neutralised' and r[0, 2] == 0
    assert kind[1, 2] == 'neutralised' and r[1, 2] == 0
