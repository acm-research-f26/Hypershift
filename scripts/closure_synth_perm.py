"""Closure Task 2b: validate the dimension-matched conditional-permutation control on synthetic data (ledgered).

  python scripts/closure_synth_perm.py   -> appends to docs/audit_20261022/A12_SYNTHETIC.md
Increment = full − full_perm, where full_perm has the 6 non-style columns permuted within date inside 20 bins of the
style-only model's prediction (same learner, same input dimension). Scenarios: S (style only), I (b=0.15), I5 (b=0.5).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.featbench.prospective import fit_fixed  # noqa: E402
from hypershift.fbaudit import stats as S, matched as M  # noqa: E402
from closure_synth import inputs, blocks, TRAIN, EVAL, REPS  # noqa: E402

OUT = Path('docs/audit_20261022')


def main():
    rows = {}
    for sc, b in (('S', .15), ('I', .15), ('I5', .5)):
        for rep in REPS:
            p = M.synth_panel('S' if sc == 'S' else 'I', rep, b=b)
            y = (p['y'] - p['y'].mean(0, keepdims=True)).T
            m = np.ones_like(y, bool)
            tr, ev = slice(0, TRAIN), slice(TRAIN, TRAIN + EVAL)
            Xs, Xf = inputs(p, [0, 1, 2, 3]), inputs(p, list(range(10)))
            both = np.concatenate([Xs[tr], Xs[ev]])
            base = fit_fixed('lgbm', Xs[tr], y[tr], m[tr], both, {'n_estimators': 50}, seed=7, threads=8)
            Xp = Xf.copy()
            allT = np.r_[np.arange(TRAIN), np.arange(TRAIN, TRAIN + EVAL)]
            Xp[allT, :, 4:10] = M.permute_within_bins(Xf[allT][:, :, 4:10], base, m[allT], bins=20, seed=rep)
            full = fit_fixed('lgbm', Xf[tr], y[tr], m[tr], Xf[ev], {'n_estimators': 50}, seed=7, threads=8)
            perm = fit_fixed('lgbm', Xp[tr], y[tr], m[tr], Xp[ev], {'n_estimators': 50}, seed=7, threads=8)
            ye, me = y[ev], m[ev]
            inc = S.spearman_daily(full, ye, me) - S.spearman_daily(perm, ye, me)
            rows.setdefault(sc, []).append(S.pooled(blocks(inc), 'stationary', 10))
            print(sc, rep, flush=True)
    fmt = lambda r: f"{r['pooled']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
    ex0 = lambda r: r['ci95'][0] > 0 or r['ci95'][1] < 0
    s_ok = all(not ex0(r) for r in rows['S'])
    i_ok = all(r['ci95'][0] > 0 for r in rows['I'])
    i5_ok = all(r['ci95'][0] > 0 for r in rows['I5'])
    L = ['', '## Follow-up (ledgered): dimension-matched conditional-permutation control', '',
         'Why the naive matched increment failed: in S, adding 6 uninformative columns *lowers* the 50-round model\'s score '
         '(dilution/capacity), so full − style is negative even with no information. In I it stays negative, so the planted '
         'term is outweighed. Encompassing is positive in S because two learners capture different parts of the same '
         'nonlinear style signal. Neither is a valid test.', '',
         'Control: `full_perm` has the same 10 inputs, but the 6 non-style columns are permuted jointly within date '
         'inside 20 bins of the style-only model\'s prediction (`matched.permute_within_bins`). Increment = full − '
         'full_perm (raw RankIC). This matches input dimension and learner capacity and keeps the extra columns\' '
         'association with style; only stock-specific information is destroyed.', '',
         '| Scenario | Planted b (standalone IC ≈) | ' + ' | '.join(f'rep {r}' for r in REPS) + ' |', '|---|---|' + '---|' * len(REPS)]
    for sc, lab in (('S', '0 (none)'), ('I', '0.15 (≈0.015)'), ('I5', '0.5 (≈0.05)')):
        L.append(f'| {sc} | {lab} | ' + ' | '.join(fmt(r) for r in rows[sc]) + ' |')
    L += ['', f"- No false increment in S: **{'yes' if s_ok else 'no'}**. Detects b = 0.15: **{'yes' if i_ok else 'no'}**. "
          f"Detects b = 0.5: **{'yes' if i5_ok else 'no'}**.",
          f"- **Permutation-control verdict:** {'VALID (no false positive; detects planted information)' if s_ok and (i_ok or i5_ok) else 'NOT VALID'}"
          f"{'; power limited to effects ≳ 0.05 standalone IC with this learner' if s_ok and i5_ok and not i_ok else ''}."]
    with open(OUT / 'A12_SYNTHETIC.md', 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
