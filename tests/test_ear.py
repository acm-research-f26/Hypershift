import numpy as np
from types import SimpleNamespace

from hypershift.strategy import ear as EA


def _block(rows):
    keys = ('accessionNumber', 'form', 'filingDate', 'acceptanceDateTime', 'items')
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def test_events_original_8k_item_202_only_and_collapsed():
    blk = _block([('a1', '8-K', '2020-04-28', '2020-04-29T04:05:00.000Z', '2.02,9.01'),
                  ('a2', '8-K/A', '2020-04-29', '2020-04-30T04:05:00.000Z', '2.02'),
                  ('a3', '8-K', '2020-04-30', '2020-04-30T20:05:00.000Z', '2.02,7.01'),      # within 5 trading days
                  ('a4', '8-K', '2020-05-15', '2020-05-15T20:05:00.000Z', '5.02'),
                  ('a5', '10-Q', '2020-05-01', '2020-05-01T20:05:00.000Z', ''),
                  ('a6', '8-K', '2020-07-28', '2020-07-28T22:05:00.000Z', '2.02')])
    cal = np.busday_offset('2020-01-01', np.arange(200), roll='forward')
    ev = EA.events(blk, cal)
    assert [e['accession'] for e in ev] == ['a1', 'a6']
    assert cal[ev[0]['d0']] == np.datetime64('2020-04-28')


def test_event_day_uses_official_filing_date_next_trading_day_if_closed():
    cal = np.array(['2020-04-24', '2020-04-27', '2020-04-28'], dtype='datetime64[D]')
    assert EA.event_day('2020-04-27', cal) == 1
    assert EA.event_day('2020-04-25', cal) == 1          # Saturday filing date -> Monday


def test_ear_window_and_placebo_window_are_disjoint_and_market_adjusted():
    T = 20
    r = np.zeros((3, T))
    r[0, 10] = .05                                        # event-day jump for stock 0
    r[:, 10] += .01                                       # market move on that day for everyone
    r[0, 6] = -.02                                        # pre-event move (placebo window d0-4..d0-2)
    close = np.cumprod(1 + r, 1)
    obs = np.ones((3, T), bool)
    s = EA.window_ar(close, obs, obs, 0, 10, (-1, 1))
    p = EA.window_ar(close, obs, obs, 0, 10, (-4, -2))
    assert np.isclose(s, .05 - .05 / 3, atol=1e-9)        # market = EW of members incl. the stock itself
    assert np.isclose(p, -.02 + .02 / 3, atol=1e-9)


def test_window_ar_requires_all_bars():
    close = np.ones((2, 12))
    obs = np.ones((2, 12), bool)
    obs[0, 6] = False
    assert np.isnan(EA.window_ar(close, obs, obs, 0, 6, (-1, 1)))


def test_rank_bins_and_tilt_weights_keep_industry_sums():
    sig = np.array([.1, -.2, .3, np.nan, -.5, .6, .0, .2, -.1, .4])
    elig = np.ones(10, bool)
    bot, top = EA.rank_bins(sig, elig, frac=.2)
    # 9 names with a signal -> round(1.8) = 2 per bin
    assert np.flatnonzero(bot).tolist() == [1, 4] and np.flatnonzero(top).tolist() == [5, 9]
    ind = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    b, a = EA.tilt_weights(elig, bot, top, ind)
    for g in (1, 2, 3):
        assert np.isclose(a[ind == g].sum(), b[ind == g].sum())
    assert a[1] == 0 and a[4] == 0 and np.isclose(a[5], 2 * b[5])
    assert np.isclose(a[0], b[0] * 4 / 3) and np.isclose(a[3], b[3] * 4 / 3)   # no top name: to non-bottom names
    assert np.isclose(a[9], b[9]) and np.isclose(a[8], b[8])                  # no bottom name: unchanged


def test_event_bins_use_only_trailing_breakpoints():
    rng = np.random.default_rng(0)
    rows = [(k % 7, 300 + k, float(rng.normal()), 'EAR') for k in range(400)]
    extra = (0, 650, 99., 'EAR')                            # an extreme event cannot change bins assigned before it
    b1 = EA.event_bins(rows + [extra], lookback=252, min_n=100)
    b2 = EA.event_bins(rows, lookback=252, min_n=100)
    early = [k for k, r in enumerate(rows) if r[1] <= 650]
    assert [b1[k][2] for k in early] == [b2[k][2] for k in early]
    assert all(np.isnan(r[2]) for r in b2[:100])            # fewer than min_n prior events: no bin
    vals = np.array([r[2] for r in b1 if np.isfinite(r[2])])
    assert set(np.unique(vals)) <= {-1., 0., 1.} and b1[-1][2] == 1.


def test_tilt_weights_cap():
    elig = np.ones(10, bool)
    bot = np.zeros(10, bool); bot[:6] = True
    top = np.zeros(10, bool); top[6] = True
    ind = np.ones(10, int)
    b, a = EA.tilt_weights(elig, bot, top, ind, cap=2.)
    assert np.isclose(a.sum(), 1) and a.max() <= 2 * b.max() + 1e-12
    assert np.isclose(a[6], .2) and np.allclose(a[7:], .2)          # top first, then the other non-bottom names
    assert np.allclose(a[:6], .2 / 6)                                # what cannot move stays on the bottom names
