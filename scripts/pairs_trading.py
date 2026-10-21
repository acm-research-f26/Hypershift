"""Study X2: distance-method pairs trading on graph structure, pairwise (clique) vs star expansion.

Predeclared in docs/screening_20261019/RESEARCH_SCREENING_PLAN.md (Study X amendment). Gatev-Goetzmann-Rouwenhorst
style: 252-session formation, 126-session trading, a new portfolio every month start, daily return = mean over the
active overlapping portfolios, scored only on dates inside the evaluation years. Every grouping is rebuilt at each
formation from formation data only.

  pairwise: all within-group pairs; top-N by SSD of normalised prices; spread = P_i - P_j
  star:     each stock vs a buy-and-hold basket of the other members of its group; top-N by SSD

  python scripts/pairs_trading.py --out docs/screening_20261019 --results results/PAIRS_SP500_20261021
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, 'scripts')
from screen_analysis import bootstrap_means, boot_p, write_csv  # noqa: E402
from hypershift.eval.stats import holm  # noqa: E402

FORM, TRADE = 252, 126
GROUPINGS = ('cluster', 'sector', 'all', 'random')


def load(path):
    z = np.load(path, allow_pickle=False)
    ident = z['identity_ok'].astype(bool)
    price = z['features'][ident][:, :, -1].astype(float)   # adjusted close / per-stock constant
    return price, z['mask'][ident].astype(bool), z['dates'].astype('datetime64[D]'), z['sectors'][ident].astype(int)


def groupings(price, sectors, u, start, m, draws):
    r = price[u, start+1:m] / price[u, start:m-1] - 1
    r = r / np.maximum(r.std(1, keepdims=True), 1e-4)
    cluster = KMeans(n_clusters=16, random_state=7, n_init=5).fit_predict(r)
    out = {'cluster': [cluster], 'sector': [sectors[u]], 'all': [np.zeros(len(u), int)]}
    rng = np.random.default_rng(int(m))
    out['random'] = [rng.permutation(cluster) for _ in range(draws)]
    return out


def candidates(x, labels, family, n):
    """Return (A, B) leg index-series over formation+trading and the chosen count. x: normalised prices [k, T]."""
    f = x[:, :FORM]
    if family == 'pairwise':
        sq = (f**2).sum(1)
        d = sq[:, None] + sq[None] - 2 * f @ f.T
        ok = (labels[:, None] == labels[None]) & np.triu(np.ones_like(d, bool), 1)
        d = np.where(ok, d, np.inf)
        flat = np.argsort(d, axis=None)[:n]
        flat = flat[np.isfinite(d.ravel()[flat])]
        i, j = np.unravel_index(flat, d.shape)
        return x[i], x[j]
    a, b = [], []
    for g in np.unique(labels):
        ix = np.flatnonzero(labels == g)
        if len(ix) < 3:
            continue
        total = x[ix].sum(0)
        for i in ix:
            a.append(x[i]); b.append((total - x[i]) / (len(ix) - 1))
    a, b = np.array(a), np.array(b)
    ssd = ((a[:, :FORM] - b[:, :FORM])**2).sum(1)
    keep = np.argsort(ssd)[:n]
    return a[keep], b[keep]


def simulate(a, b, n, wait):
    """Gross daily pnl and leg-trade counts over the trading window (committed capital = n pairs)."""
    s = a - b
    mu, sd = s[:, :FORM].mean(1), np.maximum(s[:, :FORM].std(1), 1e-12)
    z = (s - mu[:, None]) / sd[:, None]
    ra = a[:, FORM:] / a[:, FORM-1:-1] - 1
    rb = b[:, FORM:] / b[:, FORM-1:-1] - 1
    zt = z[:, FORM:]
    k, L = zt.shape
    target, pos = np.zeros((k, L)), np.zeros(k)
    for d in range(L):                                              # desired position after close d
        new = pos.copy()
        new[(pos == 0) & (zt[:, d] < -2)] = 1                       # spread cheap: long A, short B
        new[(pos == 0) & (zt[:, d] > 2)] = -1
        new[((pos == 1) & (zt[:, d] >= 0)) | ((pos == -1) & (zt[:, d] <= 0))] = 0
        target[:, d] = pos = new
    zero = np.zeros((k, 1))
    act = np.concatenate([zero, target[:, :-1]], 1) if wait else target.copy()   # wait: execute at next close
    act[:, -1] = 0                                                  # everything closed at the window end
    held = np.concatenate([zero, act[:, :-1]], 1)                   # position over (close d-1, close d]
    pnl = (held * (ra - rb)).sum(0) / n
    legs = 2 * (np.abs(act - held) > 0).sum(0)
    return pnl, legs


def run(price, observed, dates, sectors, years, n, wait, draws, families=('pairwise', 'star')):
    # Missing closes are carried forward: a stock that stops trading earns 0 thereafter.
    filled = price.copy()
    for t in range(1, filled.shape[1]):
        filled[:, t] = np.where(observed[:, t], filled[:, t], filled[:, t-1])
    months = np.flatnonzero(np.r_[True, dates[1:].astype('datetime64[M]') != dates[:-1].astype('datetime64[M]')])
    series = {}
    for y in years:
        lo = int(np.searchsorted(dates, np.datetime64(f'{y}-01-01'))); hi = int(np.searchsorted(dates, np.datetime64(f'{y+1}-01-01')))
        starts = [m for m in months if m + TRADE > lo and m < hi and m - FORM - 1 >= 0]
        acc = {(fam, g): np.zeros(hi-lo) for fam in families for g in GROUPINGS}
        legs = {k: np.zeros(hi-lo) for k in acc}
        active = np.zeros(hi-lo)
        for m in starts:
            start = m - FORM
            u = np.flatnonzero(observed[:, start:m].all(1))
            end = min(m + TRADE, filled.shape[1])
            x = filled[u, start:end] / filled[u, start:start+1]
            gs = groupings(price, sectors, u, start, m, draws)
            span = np.arange(m, end)
            inside = (span >= lo) & (span < hi)
            active[span[inside]-lo] += 1
            for fam in families:
                for g, labs in gs.items():
                    p_sum, l_sum = 0., 0.
                    for lab in labs:
                        a, b = candidates(x, lab, fam, n)
                        p, l = simulate(a, b, n, wait)
                        p_sum = p_sum + p; l_sum = l_sum + l
                    acc[(fam, g)][span[inside]-lo] += p_sum[inside] / len(labs)
                    legs[(fam, g)][span[inside]-lo] += l_sum[inside] / len(labs) / n
        for k in acc:
            series[(k[0], k[1], y)] = (acc[k] / np.maximum(active, 1), legs[k] / np.maximum(active, 1))
    return series


def net(series, fam, g, y, bp):
    gross, legs = series[(fam, g, y)]
    return gross - legs * bp * 1e-4


def pooled(series_by_year, B):
    means = [s.mean() for s in series_by_year]
    boots = np.mean([bootstrap_means(s, 10, B, np.random.default_rng(i)) for i, s in enumerate(series_by_year)], 0)
    m = float(np.mean(means))
    return m, np.percentile(boots, [2.5, 97.5]), boot_p(m, boots)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', default='data/fresh/sp500_daily/screen_sp500_2015_2026.npz')
    p.add_argument('--out', default='docs/screening_20261019')
    p.add_argument('--results', default='results/PAIRS_SP500_20261021')
    p.add_argument('--years', nargs='+', type=int, default=[2020, 2022, 2023])
    p.add_argument('--draws', type=int, default=20)
    p.add_argument('--boot', type=int, default=5000)
    args = p.parse_args()
    assert all(y < 2024 for y in args.years), '2024+ is reserved'
    price, observed, dates, sectors = load(args.panel)
    keep = dates < np.datetime64(f'{max(args.years)+1}-01-01')            # nothing after the last scored year
    price, observed, dates = price[:, keep], observed[:, keep], dates[keep]
    r = np.where(observed[:, 1:] & observed[:, :-1], price[:, 1:] / price[:, :-1] - 1, 0)
    print('daily |r| > 40%:', int((np.abs(r) > .4).sum()), 'max', float(np.abs(r).max()), flush=True)
    specs = {'primary': dict(n=20, wait=True), 'same_close': dict(n=20, wait=False), 'top50': dict(n=50, wait=True)}
    res = Path(args.results); res.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); (out/'figures').mkdir(parents=True, exist_ok=True)
    allser = {}
    for name, kw in specs.items():
        allser[name] = run(price, observed, dates, sectors, args.years, draws=args.draws, **kw)
        np.savez_compressed(res/f'{name}.npz', **{f'{f}_{g}_{y}_{k}': v[i] for (f, g, y), v in allser[name].items()
                                                  for i, k in enumerate(('gross', 'legs'))})
        print('done', name, flush=True)

    rows, lines = [], ['# Study X2 — pairs trading on graph structure (exploratory)', '',
                       f'Panel `{args.panel}`; years {args.years}; formation {FORM}, trading {TRADE} sessions, monthly '
                       f'overlapping portfolios; random null = mean of {args.draws} random partitions with cluster sizes. '
                       f'Stationary bootstrap block 10, B={args.boot}, years pooled at equal weight. Survivorship-biased '
                       'panel: absolute returns are an upper bound.', '']
    for name in specs:
        S = allser[name]
        for bp in (0, 10, 25):
            for fam in ('pairwise', 'star'):
                for g in GROUPINGS:
                    ys = [net(S, fam, g, y, bp) for y in args.years]
                    m, ci, pv = pooled(ys, args.boot)
                    row = {'spec': name, 'bp': bp, 'family': fam, 'grouping': g, 'pooled_daily_mean': m,
                           'ci_lo': ci[0], 'ci_hi': ci[1], 'p': pv, 'ann_return': m*252}
                    for y, s in zip(args.years, ys):
                        row[f'sharpe_{y}'] = float(s.mean() / max(s.std(), 1e-12) * np.sqrt(252))
                        row[f'ann_{y}'] = float(s.mean()*252)
                        row[f'legs_per_day_{y}'] = float(S[(fam, g, y)][1].mean())
                    if g in ('cluster', 'sector'):
                        d = [a - b for a, b in zip(ys, [net(S, fam, 'random', y, bp) for y in args.years])]
                        dm, dci, dp = pooled(d, args.boot)
                        row.update({'vs_random_mean': dm, 'vs_random_lo': dci[0], 'vs_random_hi': dci[1], 'vs_random_p': dp})
                    if fam == 'star':
                        d = [a - b for a, b in zip(ys, [net(S, 'pairwise', g, y, bp) for y in args.years])]
                        dm, dci, dp = pooled(d, args.boot)
                        row.update({'star_minus_pairwise': dm, 'smp_lo': dci[0], 'smp_hi': dci[1], 'smp_p': dp})
                    rows.append(row)
    prim = [r for r in rows if r['spec'] == 'primary' and r['bp'] == 10]
    ha = holm({(r['family'], r['grouping']): r['p'] for r in prim if r['grouping'] != 'random'})
    hb = holm({(r['family'], r['grouping']): r['vs_random_p'] for r in prim if r['grouping'] in ('cluster', 'sector')})
    for r in prim:
        k = (r['family'], r['grouping'])
        r['holm_abs'] = ha.get(k, np.nan); r['holm_struct'] = hb.get(k, np.nan)
        allpos = all(r[f'sharpe_{y}'] > 0 for y in args.years)
        a_ok = k in ha and ha[k] < .05 and r['ci_lo'] > 0
        b_ok = k in hb and hb[k] < .05 and r['vs_random_lo'] > 0
        r['label'] = ('calibration' if r['grouping'] == 'random' else 'STRUCTURE EFFECT' if a_ok and b_ok and allpos
                      else 'POSITIVE ONLY' if a_ok else 'NO EVIDENCE')
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(out/'x2_PAIRS_RESULTS.csv', rows, cols)

    ys = args.years
    lines += ['## Primary: wait-one-day execution, top-20, 10 bp per leg per side', '',
              '| Family | Grouping | Ann. net return (pooled) [95% CI] | Sharpe ' + ' / '.join(map(str, ys)) +
              ' | Holm p (abs, 6) | vs random null, ann. [95% CI] | Holm p (struct, 4) | Label |', '|---|---|---|---|---|---|---|---|']
    for r in prim:
        vr = (f"{r['vs_random_mean']*252:+.3f} [{r['vs_random_lo']*252:+.3f}, {r['vs_random_hi']*252:+.3f}]"
              if 'vs_random_mean' in r and r['grouping'] in ('cluster', 'sector') else '—')
        lines.append(f"| {r['family']} | {r['grouping']} | {r['ann_return']:+.3f} [{r['ci_lo']*252:+.3f}, {r['ci_hi']*252:+.3f}] | "
                     + ' / '.join(f"{r[f'sharpe_{y}']:+.2f}" for y in ys)
                     + f" | {r['holm_abs']:.3f} | {vr} | {r['holm_struct']:.3f} | {r['label']} |".replace('| nan |', '| — |'))
    lines += ['', '## Star minus pairwise (primary spec, 10 bp; descriptive)', '', '| Grouping | Ann. diff [95% CI] | p |', '|---|---|---|']
    for r in prim:
        if r['family'] == 'star':
            lines.append(f"| {r['grouping']} | {r['star_minus_pairwise']*252:+.3f} [{r['smp_lo']*252:+.3f}, {r['smp_hi']*252:+.3f}] | {r['smp_p']:.3f} |")
    lines += ['', '## Sensitivities (pooled ann. net return; Sharpe by year)', '',
              '| Spec | bp | Family | Grouping | Ann. return [95% CI] | Sharpe ' + ' / '.join(map(str, ys)) + ' | legs/day |',
              '|---|---|---|---|---|---|---|']
    for r in rows:
        if (r['spec'], r['bp']) == ('primary', 10):
            continue
        lines.append(f"| {r['spec']} | {r['bp']} | {r['family']} | {r['grouping']} | {r['ann_return']:+.3f} "
                     f"[{r['ci_lo']*252:+.3f}, {r['ci_hi']*252:+.3f}] | " + ' / '.join(f"{r[f'sharpe_{y}']:+.2f}" for y in ys)
                     + f" | {np.mean([r[f'legs_per_day_{y}'] for y in ys]):.3f} |")
    (out/'x2_PAIRS_TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:30]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    S = allser['primary']
    colors = {'cluster': '#0072B2', 'sector': '#009E73', 'all': '#D55E00', 'random': '#7f7f7f'}
    fig, axes = plt.subplots(2, len(ys), figsize=(4*len(ys), 5.6), sharey='row', squeeze=False)
    for i, fam in enumerate(('pairwise', 'star')):
        for j, y in enumerate(ys):
            ax = axes[i][j]
            for g in GROUPINGS:
                ax.plot(np.cumsum(net(S, fam, g, y, 10)), color=colors[g], lw=1.2, ls='--' if g == 'random' else '-',
                        label=g if (i, j) == (0, 0) else None)
            ax.axhline(0, color='black', lw=.6); ax.set_title(f'{fam}, {y}')
        axes[i][0].set_ylabel('cumulative net return (10 bp)')
    axes[0][0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/'figures'/'x2_pairs_cumulative.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
