"""The single predeclared 2016-2023 EAR evaluation (docs/strategy_ear_20261023/PREREG.md). Run once, after the
PREREG commit; no parameter is varied.

  python scripts/ear_eval.py   -> docs/strategy_ear_20261023/EVAL.md
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.strategy import pit as SP, pit_engine as PE, study as ST, factors as FA  # noqa: E402
from ear_common import signal_rows, targets  # noqa: E402

OUT = Path('docs/strategy_ear_20261023/EVAL.md')


def main():
    d, _ = SP.load()
    assert d.dates.max() <= np.datetime64('2023-12-31')
    tg = {'EAR': targets(d, signal_rows(d, (-1, 1))), 'control': targets(d, signal_rows(d, (-4, -2)))}
    end = len(d.dates) - 1
    run = {'A': PE.run_targets({F: v[1] for F, v in tg['EAR'].items()}, d, cost_bp=0, end=end),
           'B': PE.run_targets({F: v[0] for F, v in tg['EAR'].items()}, d, cost_bp=0, end=end),
           'C': PE.run_targets({F: v[1] for F, v in tg['control'].items()}, d, cost_bp=0, end=end)}
    dates = run['A']['dates']
    yrs = len(dates) / 252
    ff = FA.load_ff('data/factors')

    def net(k, c):
        return run[k]['gross'] - c * 1e-4 * run[k]['traded']
    act = {c: net('A', c) - net('B', c) for c in (10, 20, 30)}
    s = {c: ST.ir_stats(act[c]) for c in act}
    att = FA.attribute(act[10], dates, ff)
    ctl = net('C', 10) - net('B', 10)
    inc = net('A', 10) - net('C', 10)
    sc, si = ST.ir_stats(ctl), ST.ir_stats(inc)
    ac, ai = FA.attribute(ctl, dates, ff), FA.attribute(inc, dates, ff)
    py = ST.per_year(act[10], dates)
    pyc = ST.per_year(ctl, dates)
    tr = {k: run[k]['traded'].sum() / yrs for k in run}
    nb = [int(v[2].sum()) for v in tg['EAR'].values()]
    nt = [int(v[3].sum()) for v in tg['EAR'].values()]
    cov = [v[4] for v in tg['EAR'].values()]
    L = ['# E. EAR overlay: the single predeclared 2016–2023 evaluation', '',
         f'Spec `PREREG.md` (committed before this run). Generated once by `scripts/ear_eval.py`. Window {dates[0]} to '
         f'{dates[-1]} ({yrs:.1f} years), {len(tg["EAR"])} monthly formations, next-open execution, drifting weights, '
         'delisted names liquidated at the last close.', '',
         '## Primary: EAR tilt (A) minus the equal-weight PIT S&P 500 (B)', '',
         '| Cost | Active ann. | Tracking error | **Active IR** [95% CI] | Max active drawdown | Worst year |',
         '|---|---|---|---|---|---|']
    for c in (10, 20, 30):
        worst = min(ST.per_year(act[c], dates).values())
        L.append(f"| {c} bp | {s[c]['ann_mean']:+.4f} | {s[c]['ann_vol']:.4f} | **{s[c]['ir']:+.2f}** "
                 f"[{s[c]['ci'][0]:+.2f}, {s[c]['ci'][1]:+.2f}] | {s[c]['mdd']:+.3f} | {worst:+.4f} |")
    L += ['', f"- **Active FF5 + MOM + ST_Rev alpha (10 bp):** {att['alpha_ann']:+.4f}/yr (t {att['t']:+.2f}), n = "
          f"{att['n']} days; exposures " + ', '.join(f'{k} {v:+.3f}' for k, v in att['betas'].items())
          + f"; R² {att['r2']:.2f}.",
          f"- **Turnover (traded notional / NAV / yr):** A {tr['A']:.2f}, B {tr['B']:.2f}, C {tr['C']:.2f}; the EAR "
          f"overlay adds {tr['A'] - tr['B']:.2f}.",
          f'- **Coverage and concentration:** signal coverage {np.mean(cov):.3f} (min {np.min(cov):.3f}); bottom names '
          f'per formation median {int(np.median(nb))}, top {int(np.median(nt))}; industry weights equal B by '
          'construction; maximum weight 2× B.',
          f"- **Absolute (10 bp):** A {net('A', 10).mean() * 252:+.4f}/yr, B {net('B', 10).mean() * 252:+.4f}/yr.", '',
          '| Year | ' + ' | '.join(str(y) for y in sorted(py)) + ' |', '|---|' + '---|' * len(py),
          '| A − B (10 bp) | ' + ' | '.join(f'{py[y]:+.4f}' for y in sorted(py)) + ' |',
          '| C − B (10 bp) | ' + ' | '.join(f'{pyc.get(y, float("nan")):+.4f}' for y in sorted(py)) + ' |', '',
          '## Matched price-only control (C: same construction on the pre-event window d0−4..d0−2)', '',
          '| Contrast (10 bp) | Ann. | IR [95% CI] | FF alpha (t) |', '|---|---|---|---|',
          f"| C − B (control tilt) | {sc['ann_mean']:+.4f} | {sc['ir']:+.2f} [{sc['ci'][0]:+.2f}, {sc['ci'][1]:+.2f}] | "
          f"{ac['alpha_ann']:+.4f} ({ac['t']:+.2f}) |",
          f"| **A − C (event information beyond the control)** | {si['ann_mean']:+.4f} | **{si['ir']:+.2f}** "
          f"[{si['ci'][0]:+.2f}, {si['ci'][1]:+.2f}] | {ai['alpha_ann']:+.4f} ({ai['t']:+.2f}) |", '']
    ir, t, iir = s[10]['ir'], att['t'], si['ir']
    if ir >= .3 and t >= 1. and s[10]['ann_mean'] >= .003:
        dec = ('CONTINUE (licenses one scoring of the protected 2024-01..2026-09 window + implementation-only paper '
               'trade; not validation)') if iir > 0 else \
            'STOP (primary passes but the event adds nothing over the matched price-only control)'
    else:
        dec = 'STOP (S&P 500 EAR overlay)'
    L += ['## Predeclared decision', '',
          f'Net active IR at 10 bp {ir:+.2f}; active alpha t {t:+.2f}; incremental IR over the control {iir:+.2f} '
          f"; net active {s[10]['ann_mean']:+.4f}/yr (bar 0.0030) → **{dec}**."]
    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
