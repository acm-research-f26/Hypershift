"""The single predeclared 2016-2023 evaluation of S&P 500 changer avoidance (docs/strategy_lazyprices_20261023/PREREG.md).

  python scripts/lazy_eval.py   -> docs/strategy_lazyprices_20261023/EVAL.md
Run once, after PRECHECKS.md passes. No parameter is varied.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.data import edgar as ED  # noqa: E402
from hypershift.strategy import lazyprices as LP, pit as SP, pit_engine as PE, study as ST, factors as FA  # noqa: E402
from lazy_checks import signal_rows  # noqa: E402

OUT = Path('docs/strategy_lazyprices_20261023/EVAL.md')
START, END = np.datetime64('2016-07-01'), np.datetime64('2023-12-31')


def main():
    d, _ = SP.load()
    labels = list(d.labels)
    N = len(labels)
    cal = d.dates
    cmap = json.loads(Path('data/edgar/cik_map.json').read_text())
    ind = np.array([ED.ff12(cmap.get(l, {}).get('sic')) if cmap.get(l, {}).get('cik') else 12 for l in labels])   # SIC of the latest CIK
    rows, _ = signal_rows(cal, labels)
    tA, tB, tC, tN = {}, {}, {}, {}
    n_ch, max_act, form_k = [], [], []
    for F in LP.formation_days(cal, START, END):
        e = d.eligible[:, F - 1]
        sim, form = LP.signals_at(rows, N, F, cal)
        ch = LP.changers(sim, form, e)
        top = LP.changers(-np.where(np.isfinite(sim), sim, np.nan), form, e)     # top 20% similarity within form
        b, a = LP.avoid_weights(e, ch, ind)
        tA[F], tB[F] = a, b
        tC[F] = ch / max(ch.sum(), 1)
        tN[F] = top / max(top.sum(), 1)
        n_ch.append(int(ch.sum()))
        max_act.append(float((a - b).max()))
        form_k.append(float((ch & (form == '10-K')).sum() / max(ch.sum(), 1)))
    run = {k: PE.run_targets(t, d, cost_bp=0, end=len(cal) - 1) for k, t in (('A', tA), ('B', tB), ('C', tC), ('N', tN))}
    dates = run['A']['dates']
    ff = FA.load_ff('data/factors')
    yrs = len(dates) / 252

    def net(k, c):
        return run[k]['gross'] - c * 1e-4 * run[k]['traded']
    act = {c: net('A', c) - net('B', c) for c in (10, 20, 30)}
    s = {c: ST.ir_stats(act[c]) for c in act}
    att = FA.attribute(act[10], dates, ff)
    py = ST.per_year(act[10], dates)
    trA, trB = run['A']['traded'].sum() / yrs, run['B']['traded'].sum() / yrs
    legC = net('C', 10) - net('B', 10)
    legN = net('N', 10) - net('B', 10)
    ls = net('N', 10) - net('C', 10) - 50e-4 / 252
    sC, sN, sL = ST.ir_stats(legC), ST.ir_stats(legN), ST.ir_stats(ls)
    aL = FA.attribute(ls, dates, ff)
    L = ['# E/F. S&P 500 changer avoidance: the single predeclared 2016–2023 evaluation', '',
         'Spec `PREREG.md` (frozen before download); checks `PRECHECKS.md`. Generated once by `scripts/lazy_eval.py`. '
         f'Window {dates[0]} to {dates[-1]} ({yrs:.1f} years), {len(tA)} monthly formations, next-open execution, '
         'weights drift between rebalances, delisted names liquidated at their last close.', '',
         '## Primary: active return of avoidance (A) over the equal-weight PIT S&P 500 (B)', '',
         '| Cost (bp per unit traded) | Active ann. return | Tracking error | **Active IR** [95% CI] | Max active drawdown |',
         '|---|---|---|---|---|']
    for c in (10, 20, 30):
        L.append(f"| {c} | {s[c]['ann_mean']:+.4f} | {s[c]['ann_vol']:.4f} | **{s[c]['ir']:+.2f}** [{s[c]['ci'][0]:+.2f}, "
                 f"{s[c]['ci'][1]:+.2f}] | {s[c]['mdd']:+.3f} |")
    b = att['betas']
    L += ['', f"- **Active FF5 + MOM + ST_Rev alpha:** {att['alpha_ann']:+.4f}/yr (t {att['t']:+.2f}); exposures "
          + ', '.join(f'{k} {v:+.3f}' for k, v in b.items()) + f"; R² {att['r2']:.2f}.",
          f'- **Turnover:** A {trA:.2f}× NAV/yr, B {trB:.2f}× NAV/yr; the avoidance overlay adds {trA - trB:.2f}× NAV/yr of trading.',
          f'- **Concentration:** changers per formation median {int(np.median(n_ch))} (range {min(n_ch)}–{max(n_ch)}); '
          f'maximum active weight {max(max_act):.4f}; 10-K share of changers median {np.median(form_k):.2f}.',
          f"- **Absolute:** A net {net('A', 10).mean() * 252:+.4f}/yr, B net {net('B', 10).mean() * 252:+.4f}/yr at 10 bp.", '',
          '| Year | ' + ' | '.join(str(y) for y in sorted(py)) + ' |', '|---|' + '---|' * len(py),
          '| Active @10 bp | ' + ' | '.join(f'{py[y]:+.4f}' for y in sorted(py)) + ' |', '',
          '## Declared diagnostics (non-gating)', '',
          '| Portfolio | Ann. net vs B (10 bp) | IR [95% CI] |', '|---|---|---|',
          f"| Changers (bottom 20% within form, EW) − B | {sC['ann_mean']:+.4f} | {sC['ir']:+.2f} [{sC['ci'][0]:+.2f}, {sC['ci'][1]:+.2f}] |",
          f"| Stable reporters (top 20% within form, EW) − B | {sN['ann_mean']:+.4f} | {sN['ir']:+.2f} [{sN['ci'][0]:+.2f}, {sN['ci'][1]:+.2f}] |",
          f"| Long-short: stable − changers, 50 bp/yr borrow | {sL['ann_mean']:+.4f} | {sL['ir']:+.2f} [{sL['ci'][0]:+.2f}, {sL['ci'][1]:+.2f}] |",
          '', f"Long-short FF alpha {aL['alpha_ann']:+.4f} (t {aL['t']:+.2f}), market beta {aL['betas']['Mkt-RF']:+.2f}. "
          'The diagnostics are not industry-neutral and are reported once, for interpretation only.', '']
    ir, t = s[10]['ir'], att['t']
    dec = 'CONTINUE (licenses one protected 2024–2026 scoring + implementation-only paper trade; not validation)' \
        if ir >= .3 and t >= 1. else 'STOP (S&P 500 changer avoidance)'
    L += ['## Predeclared decision', '', f'Net active IR at 10 bp {ir:+.2f}; active alpha t {t:+.2f} → **{dec}**.']
    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
