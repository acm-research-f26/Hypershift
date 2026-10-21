"""Study J analysis (plan: Study J amendment in RESEARCH_SCREENING_PLAN.md).

Primary: folds 2025/2026, {kronos, kronos_static} vs static, Wilcoxon over (fold, seed) deltas + Holm over 2 + the
stationary date bootstrap; verdict as in the deep test. Everything else is descriptive. All absolute numbers come from
daily series on the dates common to the compared arms (fold 2024 holds kronos only from 2024-07-01).

  python scripts/j_analysis.py --root results/SCREEN_SP500_20261021_j --out docs/screening_20261019
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from screen_analysis import Store, load_runs, paired_arm, bootstrap_means, boot_p, write_csv  # noqa: E402
from deep_analysis import deep_verdict  # noqa: E402
from hypershift.eval.stats import holm, wilcoxon_one_sample  # noqa: E402
from hypershift.screening.metrics import rank_metrics  # noqa: E402

PRIMARY = [2025, 2026]


class DateStore(Store):
    """Store whose daily series keep only dates >= cutoff (sensitivity s2)."""
    def __init__(self, runs, cutoff):
        super().__init__(runs); self.cutoff = np.datetime64(cutoff)

    def daily(self, fold, arm, seed, key='RankIC'):
        v, d = super().daily(fold, arm, seed, key)
        keep = d.astype('datetime64[D]') >= self.cutoff
        return v[keep], d[keep]


def family(store, folds, cands, parent='static', B=5000):
    res, wil = {}, {}
    for c in cands:
        r = paired_arm(store, folds, c, parent, B=B)
        if r is None:
            continue
        r['deltas'] = [d for f in r['per_fold'].values() for d in f['seed_delta'].values()]
        wil[c] = wilcoxon_one_sample(r['deltas']); res[c] = r
    return res, wil


def row(c, r, p, hp=None, verdict=None):
    fm = {f: x['mean'] for f, x in r['per_fold'].items()}
    pos = ' / '.join(f"{f}: {sum(d > 0 for d in x['seed_delta'].values())}/{len(x['seed_delta'])}" for f, x in r['per_fold'].items())
    cells = [c, f"{r['pooled']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]", ' / '.join(f'{f}: {m:+.4f}' for f, m in fm.items()),
             pos, str(len(r['deltas'])), f'{p:.3f}']
    if hp is not None:
        cells += [f'{hp:.3f}', verdict]
    return '| ' + ' | '.join(cells) + ' |'


def common_dates(store, fold, arms):
    sets = [set(map(str, store.daily(fold, a, s)[1])) for a in arms for s in store.seeds(fold, a)]
    return sorted(set.intersection(*sets))


def absolute(store, fold, arm, dates, key='RankIC'):
    out = []
    for s in store.seeds(fold, arm):
        v, d = store.daily(fold, arm, s, key)
        pos = {str(x): i for i, x in enumerate(d)}
        out.append(np.array([v[pos[c]] for c in dates]))
    return np.mean(out, 0)


def pooled_ci(series_by_fold, seed=0):
    boots = np.mean([bootstrap_means(s, 10, 5000, np.random.default_rng((seed, f))) for f, s in series_by_fold.items()], 0)
    m = float(np.mean([s.mean() for s in series_by_fold.values()]))
    return m, np.quantile(boots, [.025, .975]), boot_p(m, boots)


def ensemble_daily(store, fold, dates):
    """5-seed kronos ensemble (mean prediction) daily RankIC on dates."""
    preds = [np.load(store.by[(fold, 'kronos', s)]['dir']/'predictions.npz') for s in store.seeds(fold, 'kronos')]
    p = np.mean([z['prediction'] for z in preds], 0)
    _, daily = rank_metrics(p, preds[0]['target'], preds[0]['mask'])
    pos = {str(x): i for i, x in enumerate(preds[0]['dates'])}
    return np.array([daily['RankIC'][pos[c]] for c in dates])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='j')
    args = p.parse_args()
    out = Path(args.out)
    runs, status, folds, arms = load_runs(args.root)
    folds = [f for f in folds if f >= 2024]             # 2022/23 hold only the memorisation diagnostic
    store = Store(runs)
    print('complete', len(runs), 'not complete', sum(v != 'complete' for v in status.values()), 'folds', folds, 'arms', arms)
    L = [f'# Study J — Kronos zero-shot ({args.tag})', '',
         f'Root `{args.root}`. Exploratory; reserved period spent for J only (see the Study J amendment). '
         'Bootstrap: stationary, block 10, B=5000 per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided on (fold, seed) deltas.', '']
    head = '| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p |'
    # ---- primary
    res, wil = family(store, PRIMARY, ['kronos', 'kronos_static'])
    hp = holm(wil)
    L += ['## Primary: folds 2025, 2026 vs static (Holm over 2)', '', head + ' Holm p | Verdict |', '|---' * 8 + '|']
    csv = []
    for c, r in res.items():
        v = deep_verdict(hp[c], r['ci'], [x['mean'] for x in r['per_fold'].values()])
        if v != 'NO EVIDENCE' and r['pooled'] < 0:
            v += ' (−)'                                # rule is sign-blind below STRONG; mark harmful direction
        L.append(row(c, r, wil[c], hp[c], v))
        csv += [{'family': 'primary', 'candidate': c, 'parent': 'static', 'fold': f, 'seed': s, 'dRankIC': d, 'pooled': r['pooled'],
                 'ci_lo': r['ci'][0], 'ci_hi': r['ci'][1], 'wilcoxon_p': wil[c], 'holm_p': hp[c], 'verdict': v}
                for f, x in r['per_fold'].items() for s, d in x['seed_delta'].items()]
    L += ['', '(−) = the predeclared rule (Holm p < .05, not STRONG) fired in the harmful direction: evidence of no improvement, '
          'not of a gain. All seeds score the same dates, so the Wilcoxon reflects seed noise; the date-bootstrap CI is the date uncertainty.']
    # ---- absolute on common dates
    L += ['', '## Absolute RankIC on common dates (seed mean; pooled CI over primary folds)', '',
          '| Arm | ' + ' | '.join(map(str, folds)) + ' | pooled 2025-26 [95% CI] | p (mean = 0) |', '|---' * (len(folds) + 3) + '|']
    shown = [a for a in ['static', 'none', 'kronos', 'kronos_static', 'reversal', 'momentum', 'ridge', 'random'] if a in arms]
    for a in shown:
        cells, prim = [], {}
        for f in folds:
            if not store.seeds(f, a) or not store.seeds(f, 'kronos'):
                cells.append('—'); continue
            dates = common_dates(store, f, [a, 'kronos'])
            s = absolute(store, f, a, dates); cells.append(f'{s.mean():+.4f}')
            if f in PRIMARY:
                prim[f] = s
        m, ci, pv = pooled_ci(prim) if len(prim) == len(PRIMARY) else (np.nan, (np.nan, np.nan), np.nan)
        L.append(f'| {a} | ' + ' | '.join(cells) + f' | {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {pv:.3f} |')
    ens = {f: ensemble_daily(store, f, common_dates(store, f, ['kronos'])) for f in PRIMARY}
    m, ci, pv = pooled_ci(ens)
    L.append(f'| kronos {len(store.seeds(PRIMARY[0], "kronos"))}-seed ensemble | ' + ' | '.join(f'{ens[f].mean():+.4f}' if f in ens else '—' for f in folds)
             + f' | {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {pv:.3f} |')
    p1 = {f: absolute(store, f, 'kronos', common_dates(store, f, ['kronos']), 'RankIC_path1') for f in PRIMARY}
    m, ci, pv = pooled_ci(p1)
    L.append('| kronos single path (path 1) | ' + ' | '.join(f'{p1[f].mean():+.4f}' if f in p1 else '—' for f in folds)
             + f' | {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {pv:.3f} |')
    # ---- descriptive contrasts (no verdicts)
    L += ['', '## Descriptive contrasts, primary folds (no verdicts; raw p)', '', head, '|---' * 6 + '|']
    desc = [('kronos', 'none'), ('kronos_static', 'kronos'), ('none', 'static')]
    for a, b in desc:
        r, w = family(store, PRIMARY, [a], parent=b)
        if a in r:
            L.append(row(f'{a} − {b}', r[a], w[a]))
    for base in ('reversal', 'momentum', 'ridge'):
        if base in arms:
            r = paired_arm(store, PRIMARY, base, 'kronos')   # single-seed arm vs kronos seed mean; sign flipped below
            if r:
                fm = ' / '.join(f"{f}: {-x['mean']:+.4f}" for f, x in r['per_fold'].items())
                L.append(f"| kronos − {base} | {-r['pooled']:+.4f} [{-r['ci'][1]:+.4f}, {-r['ci'][0]:+.4f}] | {fm} | — | — | boot {r['p']:.3f} |")
    # ---- sensitivities
    L += ['', '## Sensitivities (descriptive)', '', head, '|---' * 6 + '|']
    if 2024 in folds:
        r, w = family(store, [2024] + PRIMARY, ['kronos', 'kronos_static'])
        for c in r:
            L.append(row(f'(s1) {c}, + fold 2024H2', r[c], w[c]))
    r, w = family(DateStore(runs, '2025-07-01'), PRIMARY, ['kronos', 'kronos_static'])
    for c in r:
        L.append(row(f'(s2) {c}, signal dates ≥ 2025-07-01', r[c], w[c]))
    for c in ('kronos_base', 'kronos_mini', 'kronos_lb64'):
        if c in arms:
            r, w = family(store, PRIMARY, [c])
            if c in r:
                L.append(row(f'(s3/s4) {c} (seed 7; Wilcoxon on 2 deltas, uninformative)', r[c], w[c]))
    # ---- trading (primary folds: identical date ranges, so metrics.json scalars are comparable)
    for f in PRIMARY:                                   # scalars are comparable only if every run scores the same dates
        ref = None
        for a in shown:
            for s in store.seeds(f, a):
                d = np.load(store.by[(f, a, s)]['dir']/'predictions.npz')['dates']
                ref = d if ref is None else ref
                assert d.shape == ref.shape and (d == ref).all(), (f, a, s)
    L += ['', '## Trading block, primary folds (seed mean of metrics.json; top-10, 10 bp; dates asserted identical across arms)', '',
          '| Arm | ' + ' | '.join(f'{f} net Sharpe' for f in PRIMARY) + ' | neutral RankIC ' + ' / '.join(map(str, PRIMARY)) + ' |', '|---' * 4 + '|']
    for a in shown:
        if all(store.seeds(f, a) for f in PRIMARY):
            sh = [np.mean([store.scalar(f, a, s, 'net_Sharpe') for s in store.seeds(f, a)]) for f in PRIMARY]
            nr = [np.mean([store.scalar(f, a, s, 'neutral_RankIC') for s in store.seeds(f, a)]) for f in PRIMARY]
            L.append(f'| {a} | ' + ' | '.join(f'{x:+.2f}' for x in sh) + ' | ' + ' / '.join(f'{x:+.4f}' for x in nr) + ' |')
    # ---- memorisation diagnostic
    mem = [f for f in (2022, 2023) if store.seeds(f, 'kronos')]
    if mem:
        L += ['', '## Memorisation diagnostic (inside pretraining window; contaminated; descriptive)', '',
              '| Fold | kronos RankIC (seed 7) [95% CI] |', '|---|---|']
        for f in mem:
            v, _ = store.daily(f, 'kronos', store.seeds(f, 'kronos')[0])
            b = bootstrap_means(v, 10, 5000, np.random.default_rng((1, f)))
            L.append(f'| {f} | {v.mean():+.4f} [{np.quantile(b, .025):+.4f}, {np.quantile(b, .975):+.4f}] |')
    (out/f'{args.tag}_TABLES.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    if csv:
        write_csv(out/f'{args.tag}_PAIRED.csv', csv, list(csv[0]))
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
