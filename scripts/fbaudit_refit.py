"""FB audit Tasks 4-5: matched refits (fixed rounds, seeds, target ladder) and dividend-corrected refits.

  python scripts/fbaudit_refit.py run main          # Task 4 jobs
  python scripts/fbaudit_refit.py run corrected     # Task 5 jobs (needs fbaudit_dividends.py analyze first)
  python scripts/fbaudit_refit.py report            # -> docs/audit_20261022/A4_TARGETS.md
  python scripts/fbaudit_refit.py report-corrected  # -> docs/audit_20261022/A5_DIVIDENDS_part2.md
Outputs: results/FBAUDIT_20261022/<tag>/<year>/<job>.npz (skipped if present). All exploratory (2019-2023 inspected).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
from hypershift.featbench.data import load_fb, FEATURES  # noqa: E402
from hypershift.fbaudit import refit, stats as S  # noqa: E402

RES = Path('results/FBAUDIT_20261022')
FB_ROOT = Path('results/FEATBENCH_SP500_20261022')
OUT = Path('docs/audit_20261022')
THREADS = min(8, os.cpu_count() or 4)
# job name -> (model, features, target, rounds, seed)
MAIN = {f'lgbm_rich_{t}_r{r}': ('lgbm', 'rich', t, r, 7) for t in ('raw', 'residual') for r in (50, 300)}
MAIN.update({f'lgbm_rich_{t}_r50': ('lgbm', 'rich', t, 50, 7) for t in ('sector', 'demean', 'sector_beta', 'neutral')})
MAIN.update({f'lgbm_rich_residual_es_s{s}': ('lgbm', 'rich', 'residual', None, s) for s in (19, 42)})
MAIN.update({f'ridge_rich_{t}_sel': ('ridge', 'rich', t, None, 7) for t in ('demean', 'sector_beta', 'neutral')})
CORRECTED = {'lgbm_rich_raw_es': ('lgbm', 'rich', 'raw', None, 7), 'lgbm_rich_residual_es': ('lgbm', 'rich', 'residual', None, 7),
             'lgbm_price_residual_es': ('lgbm', 'price', 'residual', None, 7)}


def corrected_fb(fb, year):
    arr = np.load(Path('data/audit/yf_unadjusted') / f'corrected_{year}.npy')
    fb.X = fb.X.copy()
    fb.X[..., FEATURES.index('log_dvol_20')] = arr[..., 0]
    fb.X[..., FEATURES.index('amihud_20')] = arr[..., 1]
    return fb


def run(tag):
    jobs = MAIN if tag == 'main' else CORRECTED
    for year in S.FOLDS:
        todo = {k: v for k, v in jobs.items() if not (RES / tag / str(year) / f'{k}.npz').exists()}
        if not todo:
            continue
        fb = load_fb(year, 5)
        if tag == 'corrected':
            fb = corrected_fb(fb, year)
        for name, (model, feats, tgt, rounds, seed) in todo.items():
            out = refit.run_arm(fb, model, feats, tgt, rounds=rounds, seed=seed, threads=THREADS)
            p = RES / tag / str(year)
            p.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p / f'{name}.npz', daily=out['daily'], dates=out['dates'], pred=out['pred'].astype(np.float32),
                                info=json.dumps(out['info'], default=float))
            print(year, name, f"{out['daily'].mean():+.4f}", out['info'], flush=True)


def load(tag, name):
    return {f: np.load(RES / tag / str(f) / f'{name}.npz')['daily'] for f in S.FOLDS}


def fb_series(arm, seed=7):
    return {f: S.load_daily(FB_ROOT, f, arm, seed)[0] for f in S.FOLDS}


def diff(a, b):
    return {f: a[f] - b[f] for f in S.FOLDS}


def row(label, ser):
    r = S.pooled(ser, 'stationary', 10)
    fm = list(r['per_fold'].values())
    return (f"| {label} | {r['pooled']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
            f"{' / '.join(f'{v:+.4f}' for v in fm)} | {sum(v > 0 for v in fm)}/5 |"), r


def report():
    rev = fb_series('block:reversal16')
    H = '| Series | Pooled [95% CI] | Fold means | Folds > 0 |\n|---|---|---|---|'
    L = ['# A4 — Matched target tests at fixed rounds, seeds and a target ladder', '',
         'Exploratory audit refits (`scripts/fbaudit_refit.py`). LightGBM with FB parameters and cs_z inputs on the 20 '
         'rich features; only the target (and, where stated, the number of rounds) changes. Fixed-round fits use '
         '`prospective.fit_fixed` on the FB training window (no validation step); "es" = FB early stopping on validation '
         'neutral RankIC. Score: factor-neutral RankIC, pooled mean of 5 fold means, stationary bootstrap block 10.', '',
         '## T4: residual − raw at fixed rounds (predeclared)', '', H]
    verdict = []
    for r in (50, 300):
        s, res = row(f'residual − raw, {r} rounds', diff(load('main', f'lgbm_rich_residual_r{r}'), load('main', f'lgbm_rich_raw_r{r}')))
        L.append(s)
        verdict.append(res['pooled'] >= .007 and sum(v > 0 for v in res['per_fold'].values()) >= 4)
    for r in (50, 300):
        for t in ('raw', 'residual'):
            L.append(row(f'{t}, {r} rounds (absolute)', load('main', f'lgbm_rich_{t}_r{r}'))[0])
    L += ['', f"**T4 verdict: {'target effect robust to rounds' if all(verdict) else 'NOT robust to rounds'}** "
          f"(50 rounds: {'pass' if verdict[0] else 'fail'}; 300 rounds: {'pass' if verdict[1] else 'fail'}).", '']
    L += ['## Development estimate of the frozen specification (lgbm:rich:residual, 50 rounds, descriptive)', '', H,
          row('frozen spec (50 rounds) vs 0', load('main', 'lgbm_rich_residual_r50'))[0],
          row('frozen spec (50 rounds) − reversal16', diff(load('main', 'lgbm_rich_residual_r50'), rev))[0],
          row('FB early-stopped residual (reported +0.0127) − reversal16', diff(fb_series('lgbm:rich:residual'), rev))[0], '']
    L += ['## Seeds (early-stopped, as in FB)', '', H]
    res_s = {7: fb_series('lgbm:rich:residual'), 19: load('main', 'lgbm_rich_residual_es_s19'), 42: load('main', 'lgbm_rich_residual_es_s42')}
    raw_s = {s: fb_series('lgbm:rich', s) for s in (7, 19, 42)}
    for s in (7, 19, 42):
        L.append(row(f'residual − raw, seed {s}', diff(res_s[s], raw_s[s]))[0])
    avg = {f: np.mean([res_s[s][f] - raw_s[s][f] for s in (7, 19, 42)], 0) for f in S.FOLDS}
    L.append(row('residual − raw, mean of 3 seeds', avg)[0])
    L.append(row('residual (mean of 3 seeds) vs 0', {f: np.mean([res_s[s][f] for s in (7, 19, 42)], 0) for f in S.FOLDS})[0])
    L += ['', '## Target ladder', '', 'LightGBM at 50 fixed rounds; Ridge with FB alpha selection (raw/sector/residual Ridge rows are the FB runs).', '', H]
    base = load('main', 'lgbm_rich_raw_r50')
    for t in ('raw', 'demean', 'sector', 'sector_beta', 'neutral', 'residual'):
        s = load('main', f'lgbm_rich_{t}_r50')
        L.append(row(f'lgbm {t}', s)[0])
    for t in ('demean', 'sector', 'sector_beta', 'neutral', 'residual'):
        L.append(row(f'lgbm {t} − raw', diff(load('main', f'lgbm_rich_{t}_r50'), base))[0])
    rr = fb_series('ridge:rich')
    ridge = {'raw': rr, 'sector': fb_series('ridge:rich:sector'), 'residual': fb_series('ridge:rich:residual'),
             **{t: load('main', f'ridge_rich_{t}_sel') for t in ('demean', 'sector_beta', 'neutral')}}
    for t in ('raw', 'demean', 'sector', 'sector_beta', 'neutral', 'residual'):
        L.append(row(f'ridge {t}', ridge[t])[0])
    for t in ('demean', 'sector', 'sector_beta', 'neutral', 'residual'):
        L.append(row(f'ridge {t} − raw', diff(ridge[t], rr))[0])
    OUT.joinpath('A4_TARGETS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


def report_corrected():
    H = '| Series | Pooled [95% CI] | Fold means | Folds > 0 |\n|---|---|---|---|'
    orig = {'raw': fb_series('lgbm:rich'), 'res': fb_series('lgbm:rich:residual'), 'price': fb_series('lgbm:price')}
    cor = {'raw': load('corrected', 'lgbm_rich_raw_es'), 'res': load('corrected', 'lgbm_rich_residual_es'),
           'pres': load('corrected', 'lgbm_price_residual_es')}
    L = ['## Corrected refits (dividend-free dollar volume in features and in the residual regressor; early stopping, seed 7)', '', H]
    out = {}
    for lbl, ser in (('original lgbm:rich (raw)', orig['raw']), ('corrected lgbm:rich (raw)', cor['raw']),
                     ('original lgbm:rich:residual', orig['res']), ('corrected lgbm:rich:residual', cor['res']),
                     ('lgbm:price:residual (no volume inputs; corrected regressor)', cor['pres']),
                     ('original residual − raw', diff(orig['res'], orig['raw'])),
                     ('corrected residual − raw', diff(cor['res'], cor['raw'])),
                     ('original rich − price (raw target)', diff(orig['raw'], orig['price'])),
                     ('corrected rich − price (raw target)', diff(cor['raw'], orig['price'])),
                     ('corrected rich residual − price residual', diff(cor['res'], cor['pres']))):
        s, r = row(lbl, ser)
        L.append(s)
        out[lbl] = r['pooled']
    moves = {'residual − raw': out['corrected residual − raw'] - out['original residual − raw'],
             'rich − price': out['corrected rich − price (raw target)'] - out['original rich − price (raw target)']}
    L += ['', 'Shift corrected − original: ' + '; '.join(f'{k} {v:+.4f}' for k, v in moves.items()) +
          f" (T5 criterion 3 material if any |shift| > 0.005: {'MATERIAL' if any(abs(v) > .005 for v in moves.values()) else 'not material'}).", '']
    OUT.joinpath('A5_DIVIDENDS_part2.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        report_corrected()
