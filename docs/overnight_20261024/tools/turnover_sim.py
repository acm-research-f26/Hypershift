"""Synthetic turnover and coverage of the frozen changer-avoidance rules (IMPLEMENTABILITY.md). No market data.

  PYTHONPATH=src python docs/overnight_20261024/tools/turnover_sim.py

Stylised S&P 500: N names, fiscal-year ends (assumed mix), 10-K filed 25-60 days and 10-Qs 20-40 days after period
end (large-accelerated-filer deadlines are 60 and 40 days), similarity ranks persistent across a firm's filings with
AR(1) coefficient rho. Uses the frozen functions lazyprices.signals_at / changers / avoid_weights on a business-day
calendar. Prices are flat, so turnover is pure signal turnover: sum over formations of |a_t - a_{t-1}| (one-way
traded notional, x NAV) of the avoidance portfolio; the equal-weight benchmark trades nothing here.
"""
import numpy as np

from hypershift.strategy import lazyprices as LP

FYE_MIX = {12: .80, 9: .07, 6: .07, 3: .06}     # assumption (not measured): share of fiscal-year-end months


def simulate(rho, N=500, years=(2017, 2023), seed=0, months=3):
    rng = np.random.default_rng(seed)
    cal = np.busday_offset('2016-01-01', np.arange(9 * 252), roll='forward')
    cal = cal[cal <= np.datetime64('2023-12-31')]
    fye = rng.choice(list(FYE_MIX), size=N, p=list(FYE_MIX.values()))
    ind = rng.integers(1, 13, size=N)
    rows = []
    for i in range(N):
        z = rng.normal()
        for y in range(2015, 2024):
            for q in range(4):
                m = (fye[i] - 3 * (3 - q) - 1) % 12 + 1          # quarter-end month, q = 3 is the fiscal-year end
                yy = y if m <= fye[i] else y - 1
                end = (np.datetime64(f'{yy}-{m:02d}') + 1).astype('datetime64[D]') - 1
                k = q == 3
                acc = end + int(rng.integers(25, 61) if k else rng.integers(20, 41))
                z = rho * z + np.sqrt(1 - rho ** 2) * rng.normal()
                a = int(np.searchsorted(cal, acc, side='right'))
                if a < len(cal):
                    rows.append((i, a, float(z), '10-K' if k else '10-Q'))
    F = LP.formation_days(cal, np.datetime64(f'{years[0]}-01-01'), np.datetime64(f'{years[1]}-12-31'))
    elig = np.ones(N, bool)
    prev, turn, cov, nch = None, [], {}, []
    for f in F:
        sim, form = LP.signals_at(rows, N, f, cal, months=months)
        ch = LP.changers(sim, form, elig)
        _, a = LP.avoid_weights(elig, ch, ind)
        mon = int(str(cal[f])[5:7])
        cov.setdefault(mon, []).append(np.isfinite(sim).mean())
        nch.append(ch.sum())
        if prev is not None:
            turn.append(np.abs(a - prev).sum())
        prev = a
    yrs = len(F) / 12
    return {'turnover_per_year': float(np.sum(turn) / yrs), 'coverage_by_month': {m: float(np.mean(v)) for m, v in sorted(cov.items())},
            'changers_median': float(np.median(nch)), 'changers_min': int(np.min(nch))}


if __name__ == '__main__':
    for rho in (0., .5, .8):
        r = simulate(rho)
        print(f"rho {rho}: active one-way turnover {r['turnover_per_year']:.2f} x NAV/yr; changers per formation "
              f"median {r['changers_median']:.0f}, min {r['changers_min']}; coverage by formation month "
              + ', '.join(f'{m}:{c:.2f}' for m, c in r['coverage_by_month'].items()))
    for rho in (0., .5):
        r = simulate(rho, months=4)
        print(f"[counterfactual 4-month validity] rho {rho}: turnover {r['turnover_per_year']:.2f} x NAV/yr; coverage "
              + ', '.join(f'{m}:{c:.2f}' for m, c in r['coverage_by_month'].items()))
