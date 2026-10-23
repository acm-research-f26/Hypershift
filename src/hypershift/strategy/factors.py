"""Ken French daily factors (FF5 + Mom + ST_Rev) and HAC factor attribution."""
from pathlib import Path

import numpy as np

FACTORS = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'STREV')
FILES = {'F-F_Research_Data_5_Factors_2x3_daily.csv': None, 'F-F_Momentum_Factor_daily.csv': 'MOM',
         'F-F_ST_Reversal_Factor_daily.csv': 'STREV'}


def _read(path):
    cols, rows = None, {}
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        parts = [p.strip() for p in line.split(',')]
        if cols is None:
            if len(parts) > 1 and parts[0] == '' and parts[1]:
                cols = parts[1:]
            continue
        if not (parts[0].isdigit() and len(parts[0]) == 8):
            if rows:
                break
            continue
        d = np.datetime64(f'{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}')
        rows[d] = [float(x) / 100 for x in parts[1:1 + len(cols)]]
    return cols, rows


def load_ff(directory='data/factors'):
    """Daily factors in decimal units on the dates present in all three files."""
    data = {}
    for fname, rename in FILES.items():
        cols, rows = _read(Path(directory) / fname)
        data[fname] = ([rename] if rename else cols, rows)
    common = sorted(set.intersection(*[set(r) for _, r in data.values()]))
    out = {'dates': np.array(common, dtype='datetime64[D]')}
    for cols, rows in data.values():
        for j, c in enumerate(cols):
            out[c] = np.array([rows[d][j] for d in common])
    return out


def attribute(ret, dates, ff, lags=10, excess=False):
    """OLS of daily ret (minus RF if excess) on FACTORS with Newey-West SEs; alpha annualised x252."""
    ix = {d: i for i, d in enumerate(ff['dates'])}
    keep = [k for k, d in enumerate(dates) if d in ix]
    fi = [ix[dates[k]] for k in keep]
    y = np.asarray(ret, float)[keep] - (ff['RF'][fi] if excess else 0.)
    X = np.column_stack([np.ones(len(fi))] + [ff[f][fi] for f in FACTORS])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return {'alpha_ann': float(b[0] * 252), 't': float(b[0] / np.sqrt(V[0, 0])), 'n': len(fi),
            'betas': {f: float(b[i + 1]) for i, f in enumerate(FACTORS)},
            'r2': float(1 - (e @ e) / max(((y - y.mean()) ** 2).sum(), 1e-18))}
