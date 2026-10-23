import numpy as np
from types import SimpleNamespace

from hypershift.strategy import lazyprices as LP, pit_engine as PE


def _cal():
    return np.busday_offset('2020-01-01', np.arange(120), roll='forward')


def test_formation_days_are_first_trading_days_of_months():
    cal = _cal()
    f = LP.formation_days(cal, np.datetime64('2020-02-01'), np.datetime64('2020-04-30'))
    assert [str(cal[i]) for i in f] == ['2020-02-03', '2020-03-02', '2020-04-01']


def test_latest_signal_respects_availability_and_expiry():
    cal = _cal()
    F = int(np.searchsorted(cal, np.datetime64('2020-05-01')))
    # rows: entity, available trading-day index, similarity, form
    rows = [(0, int(np.searchsorted(cal, np.datetime64('2020-04-30'))), .9, '10-Q'),   # available the day before F: used
            (1, F, .1, '10-Q'),                                                          # available on F: not yet
            (1, int(np.searchsorted(cal, np.datetime64('2020-02-03'))), .5, '10-K'),    # Feb: within 3 months -> used
            (2, int(np.searchsorted(cal, np.datetime64('2020-01-15'))), .7, '10-K')]    # January: expired for May
    sim, form = LP.signals_at(rows, n=3, F=F, cal=cal)
    assert sim[0] == .9 and form[0] == '10-Q'
    assert sim[1] == .5 and form[1] == '10-K'
    assert np.isnan(sim[2]) and form[2] == ''


def test_changers_bottom_fifth_within_form():
    sim = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1., .05, .95, np.nan])
    form = np.array(['10-Q'] * 10 + ['10-K'] * 2 + [''])
    elig = np.ones(13, bool)
    elig[3] = False
    ch = LP.changers(sim, form, elig)
    # 10-Q eligible with signal: 9 names -> round(1.8) = 2 lowest (.1, .2); 10-K: 2 names -> round(.4) = 0
    assert np.flatnonzero(ch).tolist() == [0, 1]


def test_avoid_weights_keep_industry_weights_and_sum_to_one():
    elig = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    ch = np.array([1, 0, 0, 1, 0, 0, 0], bool)
    ind = np.array([1, 1, 1, 2, 2, 3, 3])
    b, a = LP.avoid_weights(elig, ch, ind)
    assert np.isclose(b.sum(), 1) and np.isclose(a.sum(), 1)
    for g in (1, 2, 3):
        assert np.isclose(a[ind == g].sum(), b[ind == g].sum())
    assert a[0] == 0 and a[3] == 0 and np.isclose(a[1], b[1] * 1.5) and np.isclose(a[4], b[4] * 2)
    solo = LP.avoid_weights(np.array([1, 1], bool), np.array([1, 0], bool), np.array([1, 2]))[1]
    assert np.isclose(solo[0], .5)          # an industry with no non-changer keeps its changer


def test_run_targets_equals_equal_weight_benchmark():
    rng = np.random.default_rng(0)
    N, T = 12, 70
    close = 100 * np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    open_ = np.concatenate([close[:, :1], close[:, :-1] * (1 + .002 * rng.normal(size=(N, T - 1)))], 1)
    d = SimpleNamespace(close=close, open=open_, observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                        beta=np.ones((N, T)), sectors=np.zeros(N, int), dates=np.arange('2020-01-01', T, dtype='datetime64[D]'))
    d.eligible[3, 30:] = False
    days = list(range(20, 70, 10))
    bench = PE.benchmark_ew({t: np.zeros(N) for t in days}, d, cost_bp=10)
    tg = {t: d.eligible[:, t - 1] / d.eligible[:, t - 1].sum() for t in days}
    mine = PE.run_targets(tg, d, cost_bp=10)
    for k in ('gross', 'traded', 'net'):
        assert np.allclose(bench[k], mine[k])
