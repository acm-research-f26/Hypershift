"""Study FB Task 7: portfolio rules and subgroup RankIC on frozen h=5 predictions (descriptive; no retraining).

  python scripts/featbench_portfolio.py --root results/FEATBENCH_SP500_20261022 --out docs/featbench_20261022
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from screen_analysis import bootstrap_means  # noqa: E402
from hypershift.featbench.data import load_fb  # noqa: E402
from hypershift.featbench.portfolio import RULES, simulate, subgroup_rankic, weights_rule  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023]
ARMS = ['block:reversal16', 'ridge:rich', 'lgbm:rich', 'lgbm:rich:residual']  # last one added post hoc (best arm)
CHARS = ['log_dvol_20', 'vol_60', 'beta', 'amihud_20']
sharpe = lambda x: float(x.mean() / max(x.std(ddof=1), 1e-12) * np.sqrt(252))


def pooled_ci(series_by_fold, seed=0):
    boots = np.mean([bootstrap_means(s, 10, 5000, np.random.default_rng((seed, f))) for f, s in series_by_fold.items()], 0)
    m = float(np.mean([s.mean() for s in series_by_fold.values()]))
    return m, np.quantile(boots, [.025, .975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    port = {(arm, rule): {} for arm in ARMS for rule in RULES}
    sub = {(arm, c): {} for arm in ARMS for c in CHARS}
    check, hold_all = [], {}
    for f in FOLDS:
        fb = load_fb(f, 5)
        for arm in ARMS:
            run = root / str(f) / arm.replace(':', '__') / 'seed_7'
            z = np.load(run / 'predictions.npz')
            pred, sig = z['prediction'].astype(np.float64), z['signals']
            for rule in RULES:
                port[(arm, rule)][f] = simulate(weights_rule(pred, fb, sig, rule), fb, sig)
            check.append((f, arm, json.load(open(run / 'metrics.json'))['net_Sharpe'], port[(arm, 'A_top10')][f]['net_Sharpe']))
            for c in CHARS:
                sub[(arm, c)][f] = subgroup_rankic(pred, fb, sig, c)
        ev = fb.evaluate
        w = fb.eligible[:, ev].T.astype(float); w /= w.sum(1, keepdims=True)
        hold_all[f] = simulate(w, fb, ev, cost=0.)
        print('fold', f, 'done', flush=True)
    P = ['# Study FB portfolio construction on frozen predictions (descriptive)', '',
         'h = 5 predictions, seed 7, folds 2019–2023, no retraining. Daily book = mean of the last 5 daily sleeve '
         'target weights (no drift); cost 10 bp × turnover; Sharpe annualised (√252), no risk-free rate. Pooled Sharpe is '
         'computed on the concatenated daily net returns of the 5 folds. `lgbm:rich:residual` was added post hoc because '
         'it had the highest neutral RankIC. Rules: A equal-weight top 10; B long top 20% / short bottom 20% within each '
         'sector; C top 10 weighted by 1/vol_60; D long top decile / short bottom decile (gross 1 each) plus an equal-weight hedge across eligible stocks that sets net beta to 0 (revised after the final review: the first version scaled the short leg by a beta ratio, up to about 5x); '
         'E top-10 entry, keep until rank > 20.', '',
         '| Arm | Rule | Pooled net Sharpe | Fold net Sharpe 2019–2023 | Ann. net return | Turnover / day | Beta to market | Max DD (worst fold) |',
         '|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        for rule in RULES:
            r = port[(arm, rule)]
            net = np.concatenate([r[f]['net'] for f in FOLDS])
            P.append(f"| {arm} | {rule} | {sharpe(net):+.2f} | {' / '.join(f'{r[f]['net_Sharpe']:+.2f}' for f in FOLDS)} | "
                     f"{net.mean()*252:+.3f} | {np.mean([r[f]['turnover'] for f in FOLDS]):.3f} | "
                     f"{np.mean([r[f]['beta_to_market'] for f in FOLDS]):+.2f} | {min(r[f]['max_drawdown'] for f in FOLDS):+.2f} |")
    net = np.concatenate([hold_all[f]['net'] for f in FOLDS])
    P.append(f"| hold-all (equal weight, no cost) | — | {sharpe(net):+.2f} | {' / '.join(f'{hold_all[f]['net_Sharpe']:+.2f}' for f in FOLDS)} | {net.mean()*252:+.3f} | — | {np.mean([hold_all[f]['beta_to_market'] for f in FOLDS]):+.2f} | {min(hold_all[f]['max_drawdown'] for f in FOLDS):+.2f} |")
    P += ['', '## Sanity: rule A vs the screening sleeve backtest (`trading()`, net 10 bp)', '',
          '| Fold | Arm | trading() net Sharpe | rule A net Sharpe |', '|---|---|---|---|']
    P += [f'| {f} | {arm} | {x:+.2f} | {y:+.2f} |' for f, arm, x, y in check]
    (out / 'fb_PORTFOLIO.md').write_text('\n'.join(P) + '\n', encoding='utf-8')
    S = ['# Study FB subgroup predictability (descriptive, hypothesis-generating only)', '',
         'Neutral RankIC within per-date terciles of a characteristic measured at the signal close (no thresholds fitted). '
         'Pooled = mean of 5 fold means; CI = stationary bootstrap (block 10, B 5000) averaged over folds. '
         '`log_dvol_20` is the size/liquidity proxy (no shares outstanding available).', '',
         '| Arm | Characteristic | Low tercile | Mid | High | High − low [95% CI] |', '|---|---|---|---|---|---|']
    for arm in ARMS:
        for c in CHARS:
            s = sub[(arm, c)]
            cells = []
            for g in ('low', 'mid', 'high'):
                m, ci = pooled_ci({f: s[f][g] for f in FOLDS})
                cells.append(f'{m:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')
            m, ci = pooled_ci({f: s[f]['high'] - s[f]['low'] for f in FOLDS}, 1)
            S.append(f"| {arm} | {c} | {' | '.join(cells)} | {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    (out / 'fb_SUBGROUP.md').write_text('\n'.join(S) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(P + [''] + S))


if __name__ == '__main__':
    main()
