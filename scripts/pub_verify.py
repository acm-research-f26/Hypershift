"""Publication claims verification: recompute headline numbers from saved outputs and compare with cited values.

  python scripts/pub_verify.py   -> docs/publication/CLAIMS_VERIFICATION.md
No model is fitted: only saved predictions, metrics, arrays and the audit raw download are read.
MATCH = recomputed value rounds to the cited value at the cited precision (half a unit in the last digit).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
OUT = Path('docs/publication/CLAIMS_VERIFICATION.md')


def check(claimed, recomputed, decimals):
    if recomputed is None:
        return 'NOT FOUND'
    return 'MATCH' if abs(claimed - recomputed) <= .5 * 10 ** -decimals + 1e-12 else 'MISMATCH'


def _seeds(folder):
    return sorted(glob.glob(f'{folder}/seed_*/metrics.json'))


def leakfree_sharpe(folder):
    v = [json.load(open(p))['test']['sr'] for p in _seeds(folder)]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def hold_all(folder):
    p = Path(folder) / 'seed_0'
    if not (p / 'test_gt.npy').exists():
        return None
    gt, m = np.load(p / 'test_gt.npy'), np.load(p / 'test_mask.npy').astype(bool)
    d = np.array([gt[m[:, t], t].mean() for t in range(gt.shape[1])]) if gt.shape[0] == m.shape[0] else None
    return float(d.mean() / d.std() * np.sqrt(252)) if d is not None else None   # ddof=0, as eval.metrics.sharpe


def ks_ic(folder):
    v = [json.load(open(p))['test_ic'] for p in sorted(glob.glob(f'{folder}/seed_*/known_signal.json'))]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def main():
    rows = []
    add = lambda cid, what, claimed, dec, rec, n, src: rows.append((cid, what, claimed, rec, n, check(claimed, rec, dec), src))
    # THINK reproduction (2017, full NYSE)
    v, n = leakfree_sharpe('results/R5_g2/THINK_paperProtocol')
    add('T1', 'THINK (HH) leak-free Sharpe 2017, old setup (wd 5e-4, level, norm=paper)', 0.089, 3, v, n, 'results/R5_g2/THINK_paperProtocol/*/metrics.json')
    v, n = leakfree_sharpe('results/R5_g2/EH')
    add('T1', 'EH leak-free Sharpe 2017, old setup', 0.383, 3, v, n, 'results/R5_g2/EH/*/metrics.json')
    add('T1', 'Equal-weight hold-all Sharpe 2017 (same mask)', 1.531, 3, hold_all('results/R5_g2/THINK_paperProtocol'), 237,
        'results/R5_g2/THINK_paperProtocol/seed_0/test_{gt,mask}.npy')
    for norm, hh, eh in (('train', 1.687, 1.436), ('paper', 1.929, 1.146)):
        v, n = leakfree_sharpe(f'results/R5_f_{norm}/HH')
        add('T3', f'Corrected THINK leak-free Sharpe 2017, norm={norm}', hh, 3, v, n, f'results/R5_f_{norm}/HH')
        v, n = leakfree_sharpe(f'results/R5_f_{norm}/EH')
        add('T3', f'Corrected EH leak-free Sharpe 2017, norm={norm}', eh, 3, v, n, f'results/R5_f_{norm}/EH')
    # weight-decay collapse (planted signal, high level, relative inputs; oracle IC 0.279 from F_learnability.md)
    a = json.load(open('results/known_signal_analysis.json'))
    base = [r['test_ic'] for r in a['runs']['high|relative']['HH_hyper'].values()]
    add('T2', 'Planted signal: THINK test IC with coupled wd 5e-4 (D baseline)', 0.074, 3, float(np.mean(base)), len(base),
        'results/known_signal_analysis.json runs[high|relative][HH_hyper]')
    eh = [r['test_ic'] for r in a['runs']['high|relative']['EH_hyper'].values()]
    add('T2', 'Planted signal: EH test IC with coupled wd 5e-4 (F_learnability.md table)', 0.090, 3, float(np.mean(eh)), len(eh),
        'results/known_signal_analysis.json runs[high|relative][EH_hyper]')
    v, n = ks_ic('results/f_f2_c1_high_relative/EH_hyper')
    add('T2', 'Planted signal: EH test IC with wd 0 (c1)', 0.202, 3, v, n, 'results/f_f2_c1_high_relative/EH_hyper/*/known_signal.json')
    v, n = ks_ic('results/f_f2_c1_high_relative/HH_hyper')
    add('T2', 'Planted signal: THINK test IC with wd 0 (c1)', 0.174, 3, v, n, 'results/f_f2_c1_high_relative/HH_hyper/*/known_signal.json')
    # Study FB and audit (neutral RankIC, pooled mean of 2019-2023 fold means)
    from hypershift.fbaudit import stats as S
    root = Path('results/FEATBENCH_SP500_20261022')
    try:
        d = {f: S.load_daily(root, f, 'lgbm:rich:residual')[0] - S.load_daily(root, f, 'lgbm:rich')[0] for f in S.FOLDS}
        add('F1', 'FB lgbm residual − raw target (contaminated features)', 0.0140, 4, float(np.mean([x.mean() for x in d.values()])), 5, str(root))
        cor = Path('results/FBAUDIT_20261022/corrected')
        dc = {f: np.load(cor / str(f) / 'lgbm_rich_residual_es.npz')['daily'] - np.load(cor / str(f) / 'lgbm_rich_raw_es.npz')['daily'] for f in S.FOLDS}
        add('F2', 'Leak-free lgbm residual − raw target', 0.0089, 4, float(np.mean([x.mean() for x in dc.values()])), 5, str(cor))
        r = np.mean([np.load(cor / str(f) / 'lgbm_rich_residual_es.npz')['daily'].mean() for f in S.FOLDS])
        add('F2', 'Leak-free lgbm residual target, neutral RankIC', 0.0136, 4, float(r), 5, str(cor))
    except FileNotFoundError as e:
        add('F1', f'FB/audit series missing: {e}', 0, 4, None, 0, str(root))
    # labels and masks from FB saved predictions (same evaluation dates and eligibility for every arm)
    lab = {f: S.load_pred(root, f, 'lgbm:rich') for f in S.FOLDS}
    # matched 2x2 (A11): seed-averaged daily neutral RankIC, full − style
    mat = Path('results/FBAUDIT_20261022/matched')
    if mat.exists():
        for tgt, claimed in (('residual', -0.0017), ('neutral', -0.0011)):
            per = []
            for f in S.FOLDS:
                z = lab[f]
                dd = [S.spearman_daily(np.load(mat / str(f) / f'{w}_{tgt}_s{s}.npz')['pred'], z['neutral'], z['mask'])
                      for w in ('full', 'style') for s in (7, 19, 42)]
                per.append((np.mean(dd[:3], 0) - np.mean(dd[3:], 0)).mean())
            add('M1', f'Matched full − style features, {tgt} target (seed-avg)', claimed, 4, float(np.mean(per)), 5, str(mat))
        for cell, claimed in (('style_neutral', 0.0259), ('full_residual', 0.0132)):
            per = [np.mean([S.spearman_daily(np.load(mat / str(f) / f'{cell}_s{s}.npz')['pred'], lab[f]['neutral'], lab[f]['mask']).mean()
                            for s in (7, 19, 42)]) for f in S.FOLDS]
            add('M2', f'Matched cell {cell}, neutral RankIC (seed-avg)', claimed, 4, float(np.mean(per)), 5, str(mat))
    # dividend leak term (A5): IC of L and of its future-only part (L residualised on trailing yield)
    raw = Path('data/audit/yf_unadjusted/unadjusted.npz')
    if raw.exists():
        z = np.load(raw)
        Lk = np.log(z['adj'] / z['close'])
        TY = np.full_like(Lk, np.nan); TY[:, 252:] = Lk[:, 252:] - Lk[:, :-252]
        icL, icF = [], []
        for f in S.FOLDS:
            sig = lab[f]['signals']
            m = lab[f]['mask'] & np.isfinite(TY[:, sig].T) & np.isfinite(Lk[:, sig].T)
            lk, ty = np.nan_to_num(Lk[:, sig].T), np.nan_to_num(TY[:, sig].T)
            res = np.zeros_like(lk)
            for d in range(len(sig)):
                mm = m[d]
                A = np.column_stack([np.ones(mm.sum()), ty[d, mm]])
                c, *_ = np.linalg.lstsq(A, lk[d, mm], rcond=None)
                res[d, mm] = lk[d, mm] - A @ c
            icL.append(S.spearman_daily(lk, lab[f]['neutral'], m).mean()); icF.append(S.spearman_daily(res, lab[f]['neutral'], m).mean())
        add('L1', 'Dividend-adjustment term L as a score (restricted mask)', 0.0189, 4, float(np.mean(icL)), 5, str(raw))
        add('L1', 'Future-only part of L (residual on trailing yield)', 0.0305, 4, float(np.mean(icF)), 5, str(raw))
    doc_only = [
        ('S1', 'Synthetic learner probe: FB LightGBM 50 rounds gains −0.0043 from planted IC-0.046 feature', 'A12_SYNTHETIC.md; rerun: scripts/closure_synth*.py (deterministic seeds)'),
        ('E1', 'Style/residual buffered book net Sharpe +1.11 at 5 bp (close1)', 'A13_LF_AND_ECON.md; rerun: scripts/closure_econ.py'),
        ('E2', 'A8 residual LightGBM buffered book net Sharpe +0.56 [−0.32, +1.44]', 'A8_COST.md; rerun: scripts/bcreview_cost.py'),
        ('P1', 'Prospective v1 power ≈ 0.07 at θ = 0.006, 250 days', 'A1_RECOMPUTE_POWER.md; rerun: scripts/fbaudit_stats.py'),
        ('T4', 'THINK 2017 Sharpe ≈ 2 at 88–93rd percentile of random top-5 baskets', 'docs/phase1_5a/REPORT_2017.md; rerun: scripts/forensic_2017.py'),
        ('T5', 'Phase 2 protocol matrix: ordering never holds with both gaps > 2 SE in 88 cells', 'docs/phase2/REPORT_CLOSURE.md (Kaggle-era outputs)'),
        ('T6', 'Post-2017 frozen THINK Sharpe 0.43 vs hold-all 0.51 (2018–2023)', 'docs/phase1_5b/REPORT_POST2017.md; outputs in docs/phase1_5b/post2017_results.json'),
    ]
    L = ['# Claims verification (recomputed from saved outputs)', '',
         'Generated by `scripts/pub_verify.py`. No model is fitted. MATCH means the recomputed value rounds to the cited value '
         'at the cited precision. Labels are re-read from the FB saved predictions (`predictions.npz`: neutral, mask, signals).', '',
         '| Claim ID | Quantity | Cited | Recomputed | n | Status | Source |', '|---|---|---|---|---|---|---|']
    for cid, what, claimed, rec, n, st, src in rows:
        L.append(f"| {cid} | {what} | {claimed} | {'—' if rec is None else f'{rec:.4f}'} | {n} | **{st}** | `{src}` |")
    L += ['', '## Doc-only claims (not recomputed by this script; regenerable by the listed command)', '',
          '| Claim ID | Quantity | Where / how to regenerate |', '|---|---|---|']
    for cid, what, where in doc_only:
        L.append(f'| {cid} | {what} | {where} |')
    n_bad = sum(r[5] != 'MATCH' for r in rows)
    L += ['', f'Summary: {len(rows) - n_bad} of {len(rows)} recomputed claims MATCH; {n_bad} not matched or not found.', '',
          '## Notes on mismatches', '',
          '- **EH planted-signal IC under coupled decay.** `docs/phase1_5/F_learnability.md` cites +0.090. The saved '
          'per-seed values in `results/known_signal_analysis.json` average 0.0984 (5 seeds; the first 3 seeds average '
          '0.0999). The source of the cited value could not be located in the local files. The manuscript cites only the '
          'THINK value (0.074, MATCH) and the qualitative EH pattern; the cited 0.090 is not used.', '',
          '## What this verification establishes, and what it does not', '',
          '- **Establishes:** the cited summary statistics equal the values recomputed by independent aggregation code '
          'from the saved per-run outputs (metrics, daily series, predictions, the audit download). This checks '
          'transcription and aggregation.',
          '- **Does not establish:** that the models, data pipelines, feature builders or evaluators that produced those '
          'outputs are correct; that the saved outputs are free of the limitations in `LEAKAGE_AUDIT.md`; or that the '
          'results replicate on other data or with another implementation. Those rest on the project unit tests, its '
          'internal audits (`docs/phase1_5/`, `docs/audit_20261022/`) and, ultimately, on independent re-implementation, '
          'which has not been done.',
          '- **Doc-only rows** (below) are not checked here at all, beyond their regenerating commands existing.']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    sys.stdout.reconfigure(encoding='utf-8')
    print(chr(10).join(L))


if __name__ == '__main__':
    main()
