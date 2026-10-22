"""BC review Task 1: does the style-residual metric leak style information nonlinearly, and does the model add to RR?

  python scripts/bcreview_style.py   -> docs/audit_20261022/A7_STYLE_RANK_RR.md
Exploratory (2019-2023 inspected). Leak-free inputs (corrected dollar volume) everywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.featbench.data import load_fb, transform  # noqa: E402
from hypershift.featbench.prospective import fit_fixed  # noqa: E402
from hypershift.fbaudit import stats as S, diag as D, diag2 as D2  # noqa: E402
from hypershift.fbaudit.refit import target_values  # noqa: E402
from fbaudit_refit import corrected_fb  # noqa: E402

COR = Path('results/FBAUDIT_20261022/corrected')
FB_ROOT = Path('results/FEATBENCH_SP500_20261022')
OUT = Path('docs/audit_20261022')


def style_inputs(fb, signals):
    xs = np.stack([np.stack([D2._style(fb, s, n) for n in D2.STYLES], -1) for s in signals])      # [S, N, 4]
    z = transform(xs, fb.eligible[:, signals].T, 'cs_z', fb)
    sec = (fb.sectors[:, None] == np.unique(fb.sectors)[None]).astype(np.float32)
    return np.concatenate([z, np.broadcast_to(sec, (len(signals),) + sec.shape)], -1)


def main():
    ser, cor = {}, {}
    for f in S.FOLDS:
        fb = corrected_fb(load_fb(f, 5), f)
        ev = fb.evaluate
        mask = (fb.y_ok[:, ev] & fb.eligible[:, ev]).T
        neutral = fb.neutral[:, ev].T.astype(np.float64)
        full = D.exposures(fb, ev, 'full')
        Q = D2.quintile_dummies(fb, ev)
        t_lin = D.neutralise_pred(neutral, full, mask)
        t_nl = D.neutralise_pred(neutral, Q, mask)
        ytr, mtr = target_values(fb, fb.train, 'neutral')
        placebo = fit_fixed('lgbm', style_inputs(fb, fb.train), ytr, mtr & fb.eligible[:, fb.train].T,
                            style_inputs(fb, ev), {'n_estimators': 50})
        raw = np.load(COR / str(f) / 'lgbm_rich_raw_es.npz')['pred'].astype(np.float64)
        res = np.load(COR / str(f) / 'lgbm_rich_residual_es.npz')['pred'].astype(np.float64)
        rr = D2.residual_reversal(fb, ev, 21)
        arms = {'residual LightGBM': res, 'raw LightGBM neutralised (full)': D.neutralise_pred(raw, full, mask),
                'RR (residual reversal, 21d)': rr, 'RR5 (sensitivity)': D2.residual_reversal(fb, ev, 5),
                'reversal16': S.load_pred(FB_ROOT, f, 'block:reversal16')['prediction'].astype(np.float64),
                'l1': S.load_pred(FB_ROOT, f, 'block:l1')['prediction'].astype(np.float64),
                'style-only placebo': placebo, 'random': np.random.default_rng(f).normal(size=raw.shape)}
        for a, p in arms.items():
            put = lambda v, s: ser.setdefault((a, v), {}).__setitem__(f, s)
            put('LIN', S.spearman_daily(p, t_lin, mask))
            put('NL', S.spearman_daily(p, t_nl, mask))
            put('NLpred', S.spearman_daily(D.neutralise_pred(p, Q, mask), t_nl, mask))
        put = lambda v, s: ser.setdefault(('residual LightGBM', v), {}).__setitem__(f, s)
        put('NL_partial_RR', S.spearman_daily(D.neutralise_pred(res, np.concatenate([Q, rr[..., None]], -1), mask), t_nl, mask))
        put('NL_partial_RR_rev16', S.spearman_daily(D.neutralise_pred(
            res, np.concatenate([Q, rr[..., None], arms['reversal16'][..., None], arms['l1'][..., None]], -1), mask), t_nl, mask))
        cor.setdefault('res~RR', []).append(np.nanmean(S.spearman_daily(res, rr, mask)))
        cor.setdefault('res~rev16', []).append(np.nanmean(S.spearman_daily(res, arms['reversal16'], mask)))
        cor.setdefault('RR~rev16', []).append(np.nanmean(S.spearman_daily(rr, arms['reversal16'], mask)))
        print('fold', f, flush=True)
    P = lambda a, v: S.pooled(ser[(a, v)], 'stationary', 10)
    Dd = lambda a, b, v: S.pooled({f: ser[(a, v)][f] - ser[(b, v)][f] for f in S.FOLDS}, 'stationary', 10)
    fmt = lambda r: f"{r['pooled']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
    folds = lambda r: ' / '.join(f'{v:+.4f}' for v in r['per_fold'].values())
    L = ['# A7 — Style-residual metric under nonlinear residualisation; residual reversal as a baseline', '',
         'BC pre-execution review, exploratory (2019–2023 inspected). Leak-free inputs throughout.', '',
         '**Metrics:**',
         '- **LIN** = Spearman of the prediction vs the factor-neutral 5-day return linearly residualised per date on '
         'sector dummies + z(beta, log_dvol_20, ret_60, vol_60) (A2c).',
         '- **NL** = the same, residualised on sector dummies + quintile dummies of the four styles (predeclared D2).',
         '- **NLpred** = NL after also residualising the *prediction* on the same dummies.', '',
         '**Arms:**',
         '- **style-only placebo:** LightGBM, 50 rounds, inputs only the four styles (cs_z) + sector one-hot, trained '
         'on the neutral target. A positive score means the metric still contains predictable style information (D3).',
         '- **RR:** predeclared D1.', '',
         'Generated by `scripts/bcreview_style.py`.', '',
         '| Arm | LIN | NL | NLpred |', '|---|---|---|---|']
    arms = ['residual LightGBM', 'raw LightGBM neutralised (full)', 'RR (residual reversal, 21d)', 'RR5 (sensitivity)',
            'reversal16', 'l1', 'style-only placebo', 'random']
    for a in arms:
        L.append(f"| {a} | {fmt(P(a, 'LIN'))} | {fmt(P(a, 'NL'))} | {fmt(P(a, 'NLpred'))} |")
    pl, pn = P('style-only placebo', 'LIN'), P('style-only placebo', 'NL')
    leak_lin = pl['ci95'][0] > 0 or pl['ci95'][1] < 0
    leak_nl = pn['ci95'][0] > 0 or pn['ci95'][1] < 0
    choice = 'NL' if not leak_nl else ('LIN' if not leak_lin else 'none (STOP)')
    L += ['', f"**D3 placebo verdict:** LIN {'LEAKS style' if leak_lin else 'clean'} (placebo {fmt(pl)}); "
          f"NL {'LEAKS style' if leak_nl else 'clean'} (placebo {fmt(pn)}). Metric for BC: **{choice}**.", '',
          '## Incremental value of the model (NL metric)', '', '| Contrast | Pooled [95% CI] | Fold means |', '|---|---|---|']
    for a, b in (('residual LightGBM', 'RR (residual reversal, 21d)'), ('residual LightGBM', 'reversal16'),
                 ('residual LightGBM', 'raw LightGBM neutralised (full)'), ('RR (residual reversal, 21d)', 'reversal16'),
                 ('RR (residual reversal, 21d)', 'l1')):
        r = Dd(a, b, 'NL')
        L.append(f'| {a} − {b} | {fmt(r)} | {folds(r)} |')
    for v, lab in (('NL_partial_RR', 'residual LightGBM, prediction also residualised on RR'),
                   ('NL_partial_RR_rev16', 'residual LightGBM, prediction also residualised on RR, reversal16, l1')):
        L.append(f"| {lab} (vs 0) | {fmt(P('residual LightGBM', v))} | {folds(P('residual LightGBM', v))} |")
    L += ['', '## Mean per-date Spearman between predictions', '', '| Pair | Mean (fold average) |', '|---|---|']
    for k, v in cor.items():
        L.append(f'| {k} | {np.mean(v):+.3f} |')
    (OUT / 'A7_STYLE_RANK_RR.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
