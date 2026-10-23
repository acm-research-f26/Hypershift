"""Publication figures and tables, generated only from saved outputs (no fitting, no backtest, no new data).

  python scripts/pubfig.py   -> docs/publication/figures/*.png, docs/publication/tables/*.md
Inputs: results/R5_g2, results/R5_f_{train,paper} (metrics.json), results/known_signal_analysis.json,
results/f_f2_c1_high_relative (known_signal.json), results/FEATBENCH_SP500_20261022 (predictions/daily),
results/FBAUDIT_20261022/matched (predictions), data/audit/yf_unadjusted/unadjusted.npz,
docs/publication/figures/leak_mechanism.json (from scripts/pub_leak_mechanism.py).
Palette: dataviz reference categorical slots in fixed order (blue, orange, aqua, yellow); light surface.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, 'src')
from hypershift.fbaudit import stats as S  # noqa: E402

FIG, TAB = Path('docs/publication/figures'), Path('docs/publication/tables')
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FB_ROOT = Path('results/FEATBENCH_SP500_20261022')


def style(ax, title):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc='left')


def save(fig, name):
    fig.patch.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, facecolor=SURF)
    plt.close(fig)


def metrics(folder):
    rows = [json.load(open(p)) for p in sorted(glob.glob(f'{folder}/seed_*/metrics.json'))]
    return np.array([r['test']['sr'] for r in rows]), np.array([r['test_oracle_sr'] for r in rows])


def fig_repro():
    groups = [('Original setup\n(wd 5e-4, level, full-series norm)', 'results/R5_g2/THINK_paperProtocol', 'results/R5_g2/EH'),
              ('Corrected\n(wd 0, relative, train-period norm)', 'results/R5_f_train/HH', 'results/R5_f_train/EH'),
              ('Corrected\n(wd 0, relative, full-series norm)', 'results/R5_f_paper/HH', 'results/R5_f_paper/EH')]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rng = np.random.default_rng(0)
    L = ['| Setup | Arm | Seeds | Validation-selected Sharpe (mean ± SD) | Test-selected Sharpe (mean, diagnostic) |', '|---|---|---|---|---|']
    for g, (lab, hh, eh) in enumerate(groups):
        for k, (arm, path, col) in enumerate((('THINK', hh, BLUE), ('EH', eh, ORANGE))):
            v, o = metrics(path)
            x = g * 3 + k + rng.uniform(-.15, .15, len(v))
            ax.scatter(x, v, s=22, color=col, edgecolor=SURF, linewidth=.8, zorder=3, label=arm if g == 0 else None)
            ax.scatter([g * 3 + k], [o.mean()], marker='D', s=42, facecolor='none', edgecolor=col, linewidth=1.5, zorder=3,
                       label=f'{arm} test-selected mean' if g == 0 else None)
            L.append(f"| {lab.replace(chr(10), ' ')} | {arm} | {len(v)} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | {o.mean():.3f} |")
    ax.axhline(1.531, color=INK2, linewidth=1, linestyle='--')
    ax.text(-.4, 1.58, 'equal-weight hold-all 1.53', color=INK2, fontsize=8, ha='left')
    ax.set_xticks([.5, 3.5, 6.5]); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel('Top-5 Sharpe, 2017 (annualised)', color=INK2, fontsize=9)
    style(ax, 'THINK vs EH, NYSE 2017: seeds (dots) and test-selected mean (diamonds)')
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper left', bbox_to_anchor=(0, -0.18))
    save(fig, 'fig1_reproduction.png')
    (TAB / 'tab1_reproduction.md').write_text('# Table 1. THINK reproduction, NYSE 2017 (source: metrics.json)\n\n' + '\n'.join(L) +
                                             '\n\nHold-all equal-weight Sharpe 1.531 (recomputed, `CLAIMS_VERIFICATION.md`). Test-selected values select on the test year (diagnostic only).\n', encoding='utf-8')


def fig_planted():
    a = json.load(open('results/known_signal_analysis.json'))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    L = ['| Arm | Coupled wd 5e-4: IC (seeds) | wd 0: IC (seeds) | Oracle IC |', '|---|---|---|---|']
    for k, (arm, lab) in enumerate((('HH_hyper', 'THINK'), ('EH_hyper', 'EH'))):
        base = np.array([r['test_ic'] for r in a['runs']['high|relative'][arm].values()])
        c1 = np.array([json.load(open(p))['test_ic'] for p in sorted(glob.glob(f'results/f_f2_c1_high_relative/{arm}/seed_*/known_signal.json'))])
        for j, (v, col, name) in enumerate(((base, ORANGE, 'coupled weight decay 5e-4'), (c1, BLUE, 'weight decay 0'))):
            x = k * 3 + j
            ax.bar(x, v.mean(), width=.8, color=col, label=name if k == 0 else None)
            ax.scatter(np.full(len(v), x), v, s=12, color=INK, zorder=3)
        L.append(f'| {lab} | {base.mean():.3f} ({len(base)}) | {c1.mean():.3f} ({len(c1)}) | 0.279 |')
    ax.axhline(.279, color=INK2, linestyle='--', linewidth=1)
    ax.text(4.4, .285, 'oracle IC 0.279', color=INK2, fontsize=8, ha='right')
    ax.set_ylim(0, .31)
    ax.set_xticks([.5, 3.5]); ax.set_xticklabels(['THINK (hyperbolic)', 'EH (Euclidean temporal)'], fontsize=9)
    ax.set_ylabel('Test IC on planted signal', color=INK2, fontsize=9)
    style(ax, 'Planted-signal recovery (309 stocks, real graph, "high" signal)')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, .9))
    save(fig, 'fig2_planted_signal.png')
    (TAB / 'tab2_planted_signal.md').write_text('# Table 2. Planted-signal recovery (relative inputs, "high" signal)\n\n' + '\n'.join(L) +
                                               '\n\nSources: `results/known_signal_analysis.json` (coupled decay), `results/f_f2_c1_high_relative/*/seed_*/known_signal.json` (wd 0).\n', encoding='utf-8')


def labels():
    return {f: S.load_pred(FB_ROOT, f, 'lgbm:rich') for f in S.FOLDS}


def fig_leak(lab):
    mech = json.load(open(FIG / 'leak_mechanism.json'))
    z = np.load('data/audit/yf_unadjusted/unadjusted.npz')
    Lk = np.log(z['adj'] / z['close'])
    TY = np.full_like(Lk, np.nan); TY[:, 252:] = Lk[:, 252:] - Lk[:, :-252]
    ty_ser, res_ser = {}, {}
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
        ty_ser[f] = S.spearman_daily(ty, lab[f]['neutral'], m)
        res_ser[f] = S.spearman_daily(res, lab[f]['neutral'], m)
    rows = [('Trailing distribution yield TY (known at t)', S.pooled(ty_ser, 'stationary', 10), INK2),
            ('L = log(AdjClose/Close), full ex-post term', mech['L'], ORANGE),
            ('L residualised on TY (regression residual)', S.pooled(res_ser, 'stationary', 10), ORANGE),
            ('L_div: future distributions at today\'s price', mech['L_div'], ORANGE),
            ('L_path: future price-path component', mech['L_path'], BLUE),
            ('L(end): distributions after 2024-01-11', mech['L_end'], ORANGE)]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    T = ['| Score (all scored against 5-day factor-neutral returns, 2019–2023) | Rank IC pooled | 95% CI |', '|---|---|---|']
    for i, (name, r, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r['ci95'][0], r['ci95'][1]], [y, y], color=col, linewidth=2)
        ax.scatter([r['pooled']], [y], s=40, color=col, zorder=3)
        ax.text(r['ci95'][1] + .003, y, f"{r['pooled']:+.3f}", va='center', fontsize=8, color=INK)
        T.append(f"| {name} | {r['pooled']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlabel('Rank IC vs 5-day factor-neutral return (95% CI, stationary bootstrap)', color=INK2, fontsize=9)
    style(ax, 'Back-adjustment term as a score')
    ax.grid(axis='x', color=GRID, linewidth=.8); ax.grid(axis='y', visible=False)
    save(fig, 'fig3_adjustment_leak.png')
    (TAB / 'tab3_adjustment_leak.md').write_text('# Table 3. Back-adjustment term and components as scores (462 stocks, 2019–2023 signals; exploratory)\n\n' +
                                                '\n'.join(T) + '\n\nSources: `data/audit/yf_unadjusted/unadjusted.npz`, FB saved labels; components from `scripts/pub_leak_mechanism.py`.\n', encoding='utf-8')


def fig_matched(lab):
    mat = Path('results/FBAUDIT_20261022/matched')
    cells = ['full_residual', 'style_residual', 'full_neutral', 'style_neutral']
    ser = {c: {f: np.mean([S.spearman_daily(np.load(mat / str(f) / f'{c}_s{s}.npz')['pred'], lab[f]['neutral'], lab[f]['mask'])
                           for s in (7, 19, 42)], 0) for f in S.FOLDS} for c in cells}
    res = {c: S.pooled(ser[c], 'stationary', 10) for c in cells}
    inc = {t: S.pooled({f: ser[f'full_{t}'][f] - ser[f'style_{t}'][f] for f in S.FOLDS}, 'stationary', 10) for t in ('residual', 'neutral')}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={'width_ratios': [3, 2]})
    ax = axes[0]
    for k, c in enumerate(cells):
        col = BLUE if c.startswith('full') else ORANGE
        r = res[c]
        ax.plot([k, k], r['ci95'], color=col, linewidth=2); ax.scatter([k], [r['pooled']], s=40, color=col, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(['full\nresidual', 'style\nresidual', 'full\nneutral', 'style\nneutral'], fontsize=8)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel('Neutral rank IC (3-seed mean, 95% CI)', color=INK2, fontsize=9)
    style(ax, 'Matched cells (blue = 20 features, orange = 4 styles)')
    ax = axes[1]
    for k, t in enumerate(('residual', 'neutral')):
        r = inc[t]
        ax.plot([k, k], r['ci95'], color=AQUA, linewidth=2); ax.scatter([k], [r['pooled']], s=40, color=AQUA, zorder=3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['residual target', 'neutral target'], fontsize=8)
    style(ax, 'Full − style (paired)')
    save(fig, 'fig4_matched_2x2.png')
    T = ['| Series | Neutral rank IC pooled [95% CI] |', '|---|---|'] + \
        [f"| {c} | {res[c]['pooled']:+.4f} [{res[c]['ci95'][0]:+.4f}, {res[c]['ci95'][1]:+.4f}] |" for c in cells] + \
        [f"| full − style, {t} target | {inc[t]['pooled']:+.4f} [{inc[t]['ci95'][0]:+.4f}, {inc[t]['ci95'][1]:+.4f}] |" for t in inc]
    (TAB / 'tab4_matched.md').write_text('# Table 4. Matched 2×2 (LightGBM 50 rounds, seeds 7/19/42, leak-corrected dollar volume; 2019–2023; exploratory)\n\n' +
                                        '\n'.join(T) + '\n\nSource: `results/FBAUDIT_20261022/matched/`, FB saved labels.\n', encoding='utf-8')


def fig_power():
    pairs = [('residual LightGBM − raw LightGBM', 'lgbm:rich:residual', 'lgbm:rich', BLUE),
             ('residual LightGBM − reversal16', 'lgbm:rich:residual', 'block:reversal16', ORANGE),
             ('residual LightGBM vs 0', 'lgbm:rich:residual', None, AQUA)]
    days = np.arange(250, 6001, 50)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    T = ['| Contrast | LRV per day | MDE at 250 d | MDE at 1,250 d | Days for MDE 0.01 |', '|---|---|---|---|---|']
    for name, a, b, col in pairs:
        ser = S.paired_by_fold(FB_ROOT, a, b)
        lrv = float(np.mean([S.pooled({0: x}, 'stationary', 10, 3000)['se'] ** 2 * len(x) for x in ser.values()]))
        mde = 2.8 * np.sqrt(lrv / days)
        ax.plot(days, mde, color=col, linewidth=2, label=name)
        T.append(f'| {name} | {lrv:.4f} | {2.8 * np.sqrt(lrv / 250):.4f} | {2.8 * np.sqrt(lrv / 1250):.4f} | {lrv * (2.8 / .01) ** 2:.0f} |')
    ax.axhline(.01, color=INK2, linestyle='--', linewidth=1); ax.text(5900, .0105, 'effect 0.01', color=INK2, fontsize=8, ha='right')
    ax.set_xlabel('Evaluation days (single series)', color=INK2, fontsize=9)
    ax.set_ylabel('MDE, rank IC (80% power, α .05)', color=INK2, fontsize=9)
    style(ax, 'Minimum detectable effect vs test length (long-run variance from 2019–2023)')
    ax.legend(frameon=False, fontsize=8)
    save(fig, 'fig5_power.png')
    (TAB / 'tab5_power.md').write_text('# Table 5. Power by contrast (FB original features; stationary bootstrap block 10)\n\n' + '\n'.join(T) +
                                      '\n\nSource: FB `daily.npz`. LRV varies about 2.8× across years (A9); treat as indicative.\n', encoding='utf-8')


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    fig_repro(); print('fig1', flush=True)
    fig_planted(); print('fig2', flush=True)
    lab = labels()
    fig_leak(lab); print('fig3', flush=True)
    fig_matched(lab); print('fig4', flush=True)
    fig_power(); print('fig5', flush=True)


if __name__ == '__main__':
    main()
