"""Study FB tables and verdicts (plan: docs/superpowers/plans/2026-10-22-feature-benchmark.md).

  python scripts/featbench_analysis.py --root results/FEATBENCH_SP500_20261022 --out docs/featbench_20261022 --tag fb
Daily paired series on common dates per fold; stationary bootstrap block max(10, 2h), B 5000; pooled = mean of
fold means; Holm within each family. SUPPORTED = Holm p < .05, CI excludes 0, >= 80% of fold means share the
pooled sign; else NO EVIDENCE. Exploratory: every year 2019-2023 was inspected before.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from screen_analysis import bootstrap_means, boot_p, write_csv  # noqa: E402
from hypershift.eval.stats import holm  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023]


def _daily(root, year, arm, seed=7, key='neutral_RankIC'):
    p = Path(root) / str(year) / arm.replace(':', '__') / f'seed_{seed}' / 'daily.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return np.asarray(z[key], float), np.asarray(z['dates']).astype(str)


def _horizon(arm):
    parts = arm.split(':')
    return int(parts[3]) if len(parts) > 3 and parts[3] not in ('', '-') else 5


def paired(root, folds, a, b=None, key='neutral_RankIC', seed_a=7, seed_b=7, B=5000):
    """a - b (or a alone when b is None) on common dates per fold. None if any fold is missing."""
    block = max(10, 2 * _horizon(a))
    per, boots = {}, []
    for f in folds:
        da = _daily(root, f, a, seed_a, key)
        db = _daily(root, f, b, seed_b, key) if b else None
        if da is None or (b and db is None):
            return None
        if b:
            common = sorted(set(da[1]) & set(db[1]))
            ia = {d: i for i, d in enumerate(da[1])}; ib = {d: i for i, d in enumerate(db[1])}
            d = np.array([da[0][ia[c]] - db[0][ib[c]] for c in common])
        else:
            d = da[0]
        bt = bootstrap_means(d, block, B, np.random.default_rng((0, f)))
        per[f] = float(d.mean()); boots.append(bt)
    pb = np.mean(boots, 0)
    pooled = float(np.mean(list(per.values())))
    return {'per_fold': per, 'pooled': pooled, 'ci': tuple(np.quantile(pb, [.025, .975])), 'p': boot_p(pooled, pb),
            'block': block}


def verdict(holm_p, ci, fold_means):
    pooled = np.mean(fold_means)
    agree = np.mean([np.sign(m) == np.sign(pooled) for m in fold_means])
    if holm_p < .05 and (ci[0] > 0 or ci[1] < 0) and agree >= .8:
        return 'SUPPORTED'
    return 'NO EVIDENCE'


def fmt(r):
    return f"{r['pooled']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


FAMILIES = {
    'primary': [('lgbm:rich', 'ridge:rich'), ('ridge:rich', 'block:reversal16'), ('lgbm:rich', 'block:reversal16')],
    'secondary': [('lgbm:rich', 'lgbm:price'), ('mlp:rich', 'ridge:rich'), ('ridge:lags', 'block:reversal16')],
    'targets': [('ridge:rich:sector', 'ridge:rich'), ('ridge:rich:residual', 'ridge:rich'),
                ('lgbm:rich:sector', 'lgbm:rich'), ('lgbm:rich:residual', 'lgbm:rich')],
    'representations': [(f'{m}:rich:raw:5:{r}', f'{m}:rich') for m in ('ridge', 'lgbm')
                        for r in ('raw', 'cs_rank', 'sector_z', 'volscaled_z')],
}
ABSOLUTE = ['block:l1', 'block:l2_5', 'block:l6_20', 'block:l21_60', 'block:reversal16', 'ridge:lags', 'ridge:price',
            'ridge:rich', 'lgbm:price', 'lgbm:rich', 'mlp:rich', 'ridge:rich:sector', 'ridge:rich:residual',
            'lgbm:rich:sector', 'lgbm:rich:residual'] + [a for a, _ in FAMILIES['representations']]
HORIZONS = [(f'{m}:rich' if h == 5 else f'{m}:rich:raw:{h}') for h in (1, 5, 10, 20) for m in ('ridge', 'lgbm')] + \
           [f'block:reversal16:-:{h}' for h in (1, 10, 20)]


def scalar(root, year, arm, key, seed=7):
    p = Path(root) / str(year) / arm.replace(':', '__') / f'seed_{seed}' / 'metrics.json'
    return json.load(open(p)).get(key) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='fb')
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    L = [f'# Study FB tables ({a.tag})', '', f'Root `{root}`. Exploratory (2019–2023 inspected before). Primary metric: '
         'factor-neutral RankIC (h-day return minus beta × market, sector-demeaned). Paired daily series on common dates; '
         'stationary bootstrap block max(10, 2h), B 5000; pooled = mean of 5 fold means. SUPPORTED = Holm p < .05, CI '
         'excludes 0, ≥ 4/5 fold means share the pooled sign.', '']
    # absolute
    L += ['## Absolute (seed 7; h = 5 unless named)', '',
          '| Arm | Neutral RankIC pooled [95% CI] | Fold means 2019–2023 | Raw RankIC | Style-neutral | Net Sharpe (mean of folds) | Turnover |',
          '|---|---|---|---|---|---|---|']
    for arm in ABSOLUTE:
        r = paired(root, FOLDS, arm)
        if r is None:
            continue
        raw = paired(root, FOLDS, arm, key='RankIC'); sty = paired(root, FOLDS, arm, key='style_neutral_RankIC')
        sh = [scalar(root, f, arm, 'net_Sharpe') for f in FOLDS]; to = [scalar(root, f, arm, 'turnover') for f in FOLDS]
        L.append(f"| {arm} | {fmt(r)} | {' / '.join(f'{v:+.4f}' for v in r['per_fold'].values())} | {raw['pooled']:+.4f} | "
                 f"{sty['pooled']:+.4f} | {np.mean(sh):+.2f} | {np.mean(to):.2f} |" if None not in sh else
                 f"| {arm} | {fmt(r)} | {' / '.join(f'{v:+.4f}' for v in r['per_fold'].values())} | {raw['pooled']:+.4f} | {sty['pooled']:+.4f} | — | — |")
    rows = []
    for fam, pairs in FAMILIES.items():
        res = {f'{x} − {y}': paired(root, FOLDS, x, y) for x, y in pairs}
        res = {k: v for k, v in res.items() if v is not None}
        if not res:
            continue
        hp = holm({k: v['p'] for k, v in res.items()})
        L += ['', f'## Family: {fam} (Holm over {len(res)} of {len(pairs)} predeclared)', '',
              '| Contrast | ΔNeutral RankIC pooled [95% CI] | Fold means | Folds > 0 | Boot p | Holm p | Verdict |',
              '|---|---|---|---|---|---|---|']
        for k, r in res.items():
            fm = list(r['per_fold'].values())
            v = verdict(hp[k], r['ci'], fm)
            L.append(f"| {k} | {fmt(r)} | {' / '.join(f'{x:+.4f}' for x in fm)} | {sum(x > 0 for x in fm)}/5 | "
                     f"{r['p']:.3f} | {hp[k]:.3f} | {v} |")
            rows.append({'family': fam, 'contrast': k, 'pooled': r['pooled'], 'ci_lo': r['ci'][0], 'ci_hi': r['ci'][1],
                         'boot_p': r['p'], 'holm_p': hp[k], 'verdict': v,
                         **{f'fold_{f}': x for f, x in r['per_fold'].items()}})
    # horizons (descriptive)
    hrows = [(arm, paired(root, FOLDS, arm)) for arm in HORIZONS]
    hrows = [(arm, r) for arm, r in hrows if r is not None]
    if hrows:
        L += ['', '## Horizons (descriptive; no cross-horizon verdicts)', '',
              '| Arm | h | Neutral RankIC pooled [95% CI] | Block | Days / h (effective obs.) | vs reversal16 at same h |', '|---|---|---|---|---|---|']
        for arm, r in hrows:
            h = _horizon(arm)
            days = sum(len(_daily(root, f, arm)[0]) for f in FOLDS)
            ref = 'block:reversal16' if h == 5 else f'block:reversal16:-:{h}'
            c = paired(root, FOLDS, arm, ref) if not arm.startswith('block') else None
            L.append(f"| {arm} | {h} | {fmt(r)} | {r['block']} | {days / h:.0f} | {fmt(c) if c else '—'} |")
    # LightGBM seed sensitivity
    seeds = [s for s in (7, 19, 42) if all(_daily(root, f, 'lgbm:rich', s) for f in FOLDS)]
    if len(seeds) > 1:
        L += ['', '## LightGBM seed sensitivity (lgbm:rich, descriptive)', '', '| Seed | Neutral RankIC pooled [95% CI] | vs ridge:rich |', '|---|---|---|']
        for s in seeds:
            r = paired(root, FOLDS, 'lgbm:rich', seed_a=s)
            c = paired(root, FOLDS, 'lgbm:rich', 'ridge:rich', seed_a=s)
            L.append(f'| {s} | {fmt(r)} | {fmt(c)} |')
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.tag}_TABLES.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    if rows:
        cols = sorted({k for r in rows for k in r}, key=lambda c: (not c.startswith(('family', 'contrast')), c))
        write_csv(out / f'{a.tag}_PAIRED.csv', rows, cols)
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
