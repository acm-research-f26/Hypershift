"""Deep-test analysis for screen candidates (plan: deep-test amendment in RESEARCH_SCREENING_PLAN.md).

Per candidate vs parent: per-fold paired dRankIC (matched seeds, common dates) with the screen's stationary
bootstrap of the seed-averaged daily series; seed-level two-sided Wilcoxon on all (fold, seed) deltas; Holm over
candidates on the Wilcoxon p; verdict STRONG / SEED-ROBUST ONLY / NO EVIDENCE exactly as predeclared.
--compare ROOT adds a reproducibility table for every (fold, arm, seed) present in both roots.

  python scripts/deep_analysis.py --root results/SCREEN_SP500_20261019_v3_deep --out docs/screening_20261019 \
      --tag deep --compare results/SCREEN_SP500_20261019_v3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'scripts')
from screen_analysis import Store, load_runs, paired_arm, write_csv  # noqa: E402
from hypershift.eval.stats import holm, wilcoxon_one_sample  # noqa: E402


def deep_verdict(p_holm, ci, fold_means):
    if p_holm < .05:
        return 'STRONG' if ci[0] > 0 and all(m > 0 for m in fold_means) else 'SEED-ROBUST ONLY'
    return 'NO EVIDENCE'


def compare(store_a, store_b):
    rows = []
    for (f, a, s), r in sorted(store_a.by.items()):
        if (f, a, s) not in store_b.by:
            continue
        pa = np.load(r['dir']/'predictions.npz'); pb = np.load(store_b.by[(f, a, s)]['dir']/'predictions.npz')
        m = pa['mask'] & pb['mask']
        corr = float(np.corrcoef(pa['prediction'][m], pb['prediction'][m])[0, 1])
        ra, rb = r['m']['RankIC'], store_b.by[(f, a, s)]['m']['RankIC']
        rows.append({'fold': f, 'arm': a, 'seed': s, 'RankIC_a': ra, 'RankIC_b': rb, 'diff': ra - rb,
                     'prediction_corr': corr, 'identical': bool(np.array_equal(pa['prediction'], pb['prediction']))})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='deep')
    p.add_argument('--candidates', nargs='+', default=['two_residual', 'pretrain'])
    p.add_argument('--parent', default='static')
    p.add_argument('--compare')
    p.add_argument('--boot', type=int, default=5000)
    args = p.parse_args()
    out = Path(args.out); (out/'figures').mkdir(parents=True, exist_ok=True)
    runs, status, folds, arms = load_runs(args.root)
    store = Store(runs)
    print('complete', len(runs), 'not complete', sum(v != 'complete' for v in status.values()), 'folds', folds)
    res, wil = {}, {}
    for c in args.candidates:
        r = paired_arm(store, folds, c, args.parent, B=args.boot)
        deltas = [d for f in r['per_fold'].values() for d in f['seed_delta'].values()]
        r['deltas'] = deltas
        wil[c] = wilcoxon_one_sample(deltas)
        res[c] = r
    hp = holm(wil)
    lines = [f'# Deep tests ({args.tag})', '', f'Root `{args.root}`; folds {folds}; parent `{args.parent}`. '
             f'Bootstrap: stationary, block 10, B={args.boot}, per fold; pooled = equal-weight fold mean. '
             'Wilcoxon: two-sided signed-rank on all (fold, seed) deltas; Holm over candidates. Exploratory.', '',
             '| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |',
             '|---|---|---|---|---|---|---|---|']
    rows = []
    for c, r in res.items():
        fm = {f: x['mean'] for f, x in r['per_fold'].items()}
        pos = {f: f"{sum(d > 0 for d in x['seed_delta'].values())}/{len(x['seed_delta'])}" for f, x in r['per_fold'].items()}
        v = deep_verdict(hp[c], r['ci'], list(fm.values()))
        lines.append(f"| {c} | {r['pooled']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                     + ' / '.join(f'{f}: {m:+.4f}' for f, m in fm.items()) + ' | '
                     + ' / '.join(f'{f}: {s}' for f, s in pos.items())
                     + f" | {len(r['deltas'])} | {wil[c]:.3f} | {hp[c]:.3f} | {v} |")
        for f, x in r['per_fold'].items():
            for s, d in x['seed_delta'].items():
                rows.append({'candidate': c, 'fold': f, 'seed': s, 'dRankIC': d, 'fold_mean': x['mean'],
                             'fold_ci_lo': x['ci'][0], 'fold_ci_hi': x['ci'][1], 'pooled': r['pooled'],
                             'pooled_ci_lo': r['ci'][0], 'pooled_ci_hi': r['ci'][1], 'wilcoxon_p': wil[c],
                             'holm_p': hp[c], 'verdict': v})
    write_csv(out/f'{args.tag}_PAIRED.csv', rows, list(rows[0]))
    lines += ['', '## Absolute RankIC (seed mean per fold)', '', '| Arm | ' + ' | '.join(map(str, folds)) + ' | seeds |',
              '|---|' + '---|'*len(folds) + '---|']
    for a in [args.parent] + args.candidates:
        lines.append(f'| {a} | ' + ' | '.join(f"{np.mean([store.scalar(f, a, s, 'RankIC') for s in store.seeds(f, a)]):+.4f}"
                                              for f in folds) + f' | {len(store.seeds(folds[0], a))} |')
    if args.compare:
        rep = compare(store, Store(load_runs(args.compare)[0]))
        if rep:
            write_csv(out/f'{args.tag}_REPRO.csv', rep, list(rep[0]))
            d = np.array([x['diff'] for x in rep]); c_ = np.array([x['prediction_corr'] for x in rep])
            lines += ['', f'## Reproducibility vs `{args.compare}`', '',
                      f'{len(rep)} matching runs; identical predictions: {sum(x["identical"] for x in rep)}; '
                      f'RankIC diff mean {d.mean():+.4f}, mean |diff| {np.abs(d).mean():.4f}, max |diff| {np.abs(d).max():.4f}; '
                      f'prediction correlation median {np.median(c_):.3f} (min {c_.min():.3f}). Per run: `{args.tag}_REPRO.csv`.']
    (out/f'{args.tag}_TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(res), figsize=(4.2*len(res), 3.2), sharey=True, squeeze=False)
    for ax, (c, r) in zip(axes[0], res.items()):
        for i, (f, x) in enumerate(r['per_fold'].items()):
            d = list(x['seed_delta'].values())
            ax.scatter(np.full(len(d), i) + np.linspace(-.15, .15, len(d)), d, s=14, color='#0072B2', alpha=.8)
            ax.errorbar(i + .3, x['mean'], yerr=[[x['mean'] - x['ci'][0]], [x['ci'][1] - x['mean']]],
                        fmt='o', color='black', capsize=3, ms=4)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(len(r['per_fold']))); ax.set_xticklabels(list(r['per_fold']))
        ax.set_title(f"{c} − {args.parent}: pooled {r['pooled']:+.4f}, Holm p {hp[c]:.2f}")
    axes[0][0].set_ylabel('ΔRankIC (dots: seeds; bar: fold mean, date-bootstrap 95% CI)')
    fig.tight_layout(); fig.savefig(out/'figures'/f'{args.tag}_seed_deltas.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
