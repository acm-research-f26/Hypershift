"""BC review follow-ups (ledgered): paired model − style-only placebo, and a random-seed check of the book accounting.

  python scripts/bcreview_followup.py   -> appends to docs/audit_20261022/A7_STYLE_RANK_RR.md and A8_COST.md
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.featbench.data import load_fb, OHLCV, PANEL  # noqa: E402
from hypershift.featbench.prospective import fit_fixed  # noqa: E402
from hypershift.fbaudit import stats as S, diag as D, diag2 as D2, book  # noqa: E402
from hypershift.fbaudit.refit import target_values  # noqa: E402
from fbaudit_refit import corrected_fb  # noqa: E402
from bcreview_style import style_inputs  # noqa: E402

COR = Path('results/FBAUDIT_20261022/corrected')
OUT = Path('docs/audit_20261022')
SEEDS = 20


def sharpe(x):
    return float(x.mean() / max(x.std(ddof=1), 1e-12) * np.sqrt(252))


def main():
    o = np.load(OHLCV); z = np.load(PANEL)
    ohlcv_all = o['ohlcv'][z['identity_ok'].astype(bool)].astype(np.float64)
    ser, rnd = {}, {}
    for f in S.FOLDS:
        fb = corrected_fb(load_fb(f, 5), f)
        ev = fb.evaluate
        mask = (fb.y_ok[:, ev] & fb.eligible[:, ev]).T
        neutral = fb.neutral[:, ev].T.astype(np.float64)
        targets = {'neutral': neutral, 'LIN': D.neutralise_pred(neutral, D.exposures(fb, ev, 'full'), mask),
                   'NL': D2.nl_resid(neutral, fb, ev, mask)}
        ytr, mtr = target_values(fb, fb.train, 'neutral')
        placebo = fit_fixed('lgbm', style_inputs(fb, fb.train), ytr, mtr & fb.eligible[:, fb.train].T,
                            style_inputs(fb, ev), {'n_estimators': 50})
        res = np.load(COR / str(f) / 'lgbm_rich_residual_es.npz')['pred'].astype(np.float64)
        rr = D2.residual_reversal(fb, ev, 21)
        for k, tgt in targets.items():
            a, p, r = (S.spearman_daily(x, tgt, mask) for x in (res, placebo, rr))
            for name, s in (('model', a), ('placebo', p), ('RR', r), ('model − placebo', a - p), ('RR − placebo', r - p)):
                ser.setdefault((name, k), {})[f] = s
        T = fb.price.shape[1]
        for sd in range(SEEDS):
            pr = np.random.default_rng((f, sd, 99)).normal(size=res.shape)
            for sc in ('daily', 'weekly_buffer'):
                for ex in ('close1', 'open1'):
                    g = book.simulate_book(pr, fb, ev, sc, ex, ohlcv=ohlcv_all[:, :T])['gross']
                    rnd.setdefault((sc, ex, sd), []).append(g)
        print('fold', f, flush=True)
    fmt = lambda r: f"{r['pooled']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
    folds = lambda r: ' / '.join(f'{v:+.4f}' for v in r['per_fold'].values())
    A = ['', '## Follow-up: paired increments over the style-only placebo (ledgered after advisor review)', '',
         'The placebo is the matched learner (same LightGBM, 50 rounds, neutral target) with style-only inputs. '
         '"Model" is the leak-free residual LightGBM.', '',
         '| Series | Metric | Pooled [95% CI] | Fold means |', '|---|---|---|---|']
    for name in ('placebo', 'model', 'RR', 'model − placebo', 'RR − placebo'):
        for k in ('neutral', 'LIN', 'NL'):
            r = S.pooled(ser[(name, k)], 'stationary', 10)
            A.append(f'| {name} | {k} | {fmt(r)} | {folds(r)} |')
    with open(OUT / 'A7_STYLE_RANK_RR.md', 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(A) + '\n')
    B = ['', f'## Follow-up: accounting check with {SEEDS} random-score seeds (gross Sharpe, pooled over folds)', '',
         'Random scores carry no information, so gross Sharpe should centre on 0 for every schedule and execution. A '
         'consistently non-zero mean would indicate an accounting or price-data asymmetry.', '',
         '| Schedule | Exec | Mean gross Sharpe over seeds | SD over seeds | Min | Max |', '|---|---|---|---|---|---|']
    for sc in ('daily', 'weekly_buffer'):
        for ex in ('close1', 'open1'):
            v = np.array([sharpe(np.concatenate(rnd[(sc, ex, sd)])) for sd in range(SEEDS)])
            B.append(f'| {sc} | {ex} | {v.mean():+.2f} | {v.std(ddof=1):.2f} | {v.min():+.2f} | {v.max():+.2f} |')
    with open(OUT / 'A8_COST.md', 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(A + B))


if __name__ == '__main__':
    main()
