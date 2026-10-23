"""WP-A study runner shared by the exploration and the pseudo-holdout (identical code; frozen by hash).

Candidate PRICE3 and every rule are predeclared in docs/superpowers/plans/2026-10-22-strategy.md.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hypershift.strategy import signals as SG, engine as EN, factors as FA

PANEL = 'data/fresh/sp500_daily/screen_sp500_2015_2026.npz'
OHLCV = 'data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz'
MEMBERS = 'data/audit/membership/components.csv'
COSTS = (5, 10, 20, 30)
BASE_COST = 10
BORROW = 50
ARMS = ('PRICE3', 'MOM', 'LOWVOL', 'RR21')


def load_data():
    from hypershift.screening.data import load_panel
    p = load_panel(PANEL, 2026, train_stride=5)
    z, o = np.load(PANEL), np.load(OHLCV)
    ident = z['identity_ok'].astype(bool)
    ohl = o['ohlcv'][ident][:, :len(p.dates)].astype(np.float64)
    assert (o['tickers'][ident] == p.tickers).all()
    return SimpleNamespace(close=ohl[..., 3], open=ohl[..., 0], volume=ohl[..., 4], price=p.price.astype(np.float64),
                           observed=p.observed, eligible=p.eligible, beta=p.beta.astype(np.float64),
                           sectors=p.sectors, dates=p.dates, tickers=p.tickers)


def pit_eligible(d, path=MEMBERS):
    """eligible & member of the S&P 500 on the date (latest snapshot <= date; ticker '.' -> '-')."""
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    snaps = sorted((np.datetime64(r['date']), {t.strip().upper().replace('.', '-') for t in r['tickers'].split(',')}) for r in rows)
    sd = np.array([s[0] for s in snaps])
    tick = [str(t).upper() for t in d.tickers]
    out = np.zeros_like(d.eligible)
    for t, day in enumerate(d.dates):
        k = np.searchsorted(sd, day, side='right') - 1
        if k >= 0:
            mem = snaps[k][1]
            out[:, t] = [x in mem for x in tick]
    return d.eligible & out


def scores_for(d, arm, start, end, eligible=None):
    """{execution day r: score [N]} for month starts r in [start, end]; signals at the close of r-1."""
    el = d.eligible if eligible is None else eligible
    out = {}
    for r in EN.month_starts(d.dates):
        if not (start <= d.dates[r] <= end) or r - 1 < 253:
            continue
        s = r - 1
        parts = {'MOM': SG.momentum(d.price, d.observed, s), 'LOWVOL': SG.lowvol(d.price, d.observed, s),
                 'RR21': SG.rr21(d.price, d.observed, d.beta, d.sectors, el, s)}
        e = el[:, s]
        if arm == 'PRICE3':
            out[r] = SG.composite([parts['MOM'], parts['LOWVOL'], parts['RR21']], e)
        else:
            out[r] = np.where(e, parts[arm], np.nan)
    return out


def net(res, cost):
    return res['gross'] - cost * 1e-4 * res['traded'] - res['borrow']


def stationary_idx(n, block, B, rng):
    restart = rng.random((B, n)) < 1. / block
    restart[:, 0] = True
    fresh = rng.integers(n, size=(B, n))
    t = np.arange(n)[None]
    start = np.maximum.accumulate(np.where(restart, t, 0), 1)
    return (np.take_along_axis(fresh, start, 1) + t - start) % n


def ir_stats(x, B=2000, seed=0):
    x = np.asarray(x, float)
    ir = x.mean() / max(x.std(ddof=1), 1e-12) * np.sqrt(252)
    idx = stationary_idx(len(x), 21, B, np.random.default_rng(seed))
    bs = x[idx].mean(1) / np.maximum(x[idx].std(1, ddof=1), 1e-12) * np.sqrt(252)
    cum = np.cumsum(np.log1p(x))
    mdd = float(np.min(np.exp(cum - np.maximum.accumulate(cum)) - 1))
    return {'ann_mean': float(x.mean() * 252), 'ann_vol': float(x.std(ddof=1) * np.sqrt(252)), 'ir': float(ir),
            'ci': tuple(np.quantile(bs, [.025, .975])), 'mdd': mdd}


def per_year(x, dates):
    yrs = dates.astype('datetime64[Y]').astype(int) + 1970
    return {int(y): float(np.sum(x[yrs == y])) for y in np.unique(yrs)}


def run_all(d, start, end, execution='open', eligible=None):
    """Run PRICE3/MOM/LOWVOL/RR21 long-only, PRICE3 long-short, and the EW benchmark on one window."""
    end_idx = int(np.searchsorted(d.dates, end, side='right') - 1)
    res = {}
    bench_scores = scores_for(d, 'PRICE3', start, end, eligible)
    res['BENCH'] = EN.benchmark_ew(bench_scores, d, execution=execution, cost_bp=0, end=end_idx)
    for arm in ARMS:
        sc = scores_for(d, arm, start, end, eligible)
        res[f'{arm}-LO'] = EN.run(sc, d, mode='LO', execution=execution, cost_bp=0, borrow_bp=0, end=end_idx)
    res['PRICE3-LS'] = EN.run(bench_scores, d, mode='LS', execution=execution, cost_bp=0, borrow_bp=BORROW, end=end_idx)
    return res


def decision(lo_active_10, alpha, ir_20, worst_year, turnover):
    """Predeclared exploration GO rule for PRICE3-LO."""
    checks = {'IR>=0.30 @10bp': lo_active_10 >= .30, 'FF alpha >= 0': alpha >= 0, 'IR>0 @20bp': ir_20 > 0,
              'worst year > -10%': worst_year > -.10, 'traded < 300%/yr': turnover < 3.0}
    return all(checks.values()), checks
