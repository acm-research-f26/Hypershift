"""Read-only diagnostics of frozen NASDAQ-100 paths; no fitting or selection.

Run from the repository root with .venv/Scripts/python.exe followed by this path.
Writes only audit_results.json beside this script. All intervals are exploratory,
conditional on fitted forecasts, unadjusted for the historical model search.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'runs/nasdaq100_historical_monthly_20261011'
EXTENSION = ROOT / 'runs/nasdaq_topk_pseudohuber_20261011'
REPEATS = 10000
SEED = 20261019


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def path_for(frame, model, k=10):
    p = frame[(frame.model == model) & (frame.seed == 'ensemble') &
              (frame.k == k) & (frame.cost_multiplier == 1)].sort_values('date')
    assert len(p) == 1003 and p.date.is_unique
    return p.set_index('date').net


def hac_covariance(x, residual, lag=20):
    z = x * residual[:, None]
    meat = z.T @ z
    for j in range(1, lag + 1):
        cross = z[j:].T @ z[:-j]
        meat += (1 - j / (lag + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    return bread @ meat @ bread * len(x) / (len(x) - x.shape[1])


def paired_diagnostics(a, b, block):
    assert a.index.equals(b.index)
    a, b = a.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(SEED)
    n = len(a)
    rf = 1.03 ** (1 / 252) - 1
    out = []
    for start in range(0, REPEATS, 200):
        size = min(200, REPEATS - start)
        starts = rng.integers(0, n, (size, int(np.ceil(n / block))))
        ids = ((starts[:, :, None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        aa, bb = a[ids], b[ids]
        active_mean = 252 * (aa - bb).mean(axis=1)
        sharpe = np.sqrt(252) * ((aa.mean(1) - rf) / aa.std(1, ddof=1) -
                                (bb.mean(1) - rf) / bb.std(1, ddof=1))
        relative_growth = np.expm1(252 * (np.log1p(aa) - np.log1p(bb)).mean(1))
        out.append(np.column_stack([active_mean, sharpe, relative_growth]))
    boot = np.concatenate(out)
    observed = [252 * (a - b).mean(),
                np.sqrt(252) * ((a.mean() - rf) / a.std(ddof=1) -
                                (b.mean() - rf) / b.std(ddof=1)),
                np.expm1(252 * (np.log1p(a) - np.log1p(b)).mean())]
    return dict(block_sessions=block, repeats=REPEATS,
                **{name: dict(observed=float(observed[j]),
                              low=float(np.quantile(boot[:, j], .025)),
                              high=float(np.quantile(boot[:, j], .975)))
                   for j, name in enumerate(['annualized_arithmetic_active_return',
                                             'sharpe_difference',
                                             'annualized_relative_wealth_growth'])})


def main():
    # Imports use the existing production loader, but never invoke a training runner.
    import sys
    sys.path.insert(0, str(ROOT))
    from pit_research_engine import load_data, fold

    curves_path = EXTENSION / 'curves.csv'
    frame = pd.read_csv(curves_path, dtype={'seed': str})
    metrics_path = EXTENSION / 'metrics.csv'
    metrics = pd.read_csv(metrics_path, dtype={'seed': str})
    data = load_data(ROOT / 'data/nasdaq_monthly_pit/assembled')
    folds = []
    for year in [2022, 2023, 2024, 2025]:
        train, val, test = fold(data, year)
        first_seen = ~data['mask'][train].any(0) & data['mask'][test].any(0)
        folds.append(dict(year=year, train_dates=len(train), validation_dates=len(val),
                          test_dates=len(test),
                          test_eligible_without_training_membership=[t for t, yes in
                              zip(data['panel'].tickers, first_seen) if yes]))
    primary = metrics[(metrics.seed == 'ensemble') & (metrics.k == 10) &
                      (metrics.cost_multiplier == 1)]
    fields = ['model', 'annualized_return', 'sharpe', 'maximum_drawdown',
              'annualized_volatility', 'turnover_annual']
    comparisons = []
    for model, k, benchmark in [('think_mix', 10, 'QQQ'), ('think_mix', 10, 'SPY'),
                                ('stockmixer', 5, 'QQQ'),
                                ('think_mix_pseudohuber', 5, 'QQQ')]:
        a, b = path_for(frame, model, k), path_for(frame, benchmark)
        active = a.to_numpy() - b.to_numpy()
        x = np.column_stack([np.ones(len(b)), b.to_numpy() - (1.03**(1/252)-1)])
        y = a.to_numpy() - (1.03**(1/252)-1)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        covariance = hac_covariance(x, y - x @ coef)
        alpha_se = np.sqrt(covariance[0, 0]) * 252
        comparisons.append(dict(model=model, k=k, benchmark=benchmark,
            tracking_error_annual=float(active.std(ddof=1)*np.sqrt(252)),
            benchmark_beta=float(coef[1]), hac_lags=20,
            annualized_ols_alpha=float(coef[0]*252),
            alpha_normal_95_low=float(coef[0]*252-1.96*alpha_se),
            alpha_normal_95_high=float(coef[0]*252+1.96*alpha_se),
            intervals=[paired_diagnostics(a, b, block) for block in [5, 20, 60]]))
    records = pd.read_csv(PARENT / 'training_records.csv')
    hashes = {}
    for run in [PARENT, EXTENSION]:
        for filename in ['INPUT_CODE_LOCK.json', 'DEPENDENCY_CODE_LOCK.json']:
            lockfile = run / filename
            if not lockfile.exists():
                continue
            mismatches = [name for name, expected in json.loads(lockfile.read_text()).items()
                          if sha(ROOT / name) != expected]
            hashes[str(lockfile.relative_to(ROOT))] = dict(status='passed' if not mismatches else 'failed',
                                                         mismatches=mismatches)
    result = dict(as_of='2026-10-19 America/Chicago',
        scope='Frozen historical monthly NASDAQ-100 universe; five-session strategies, 2022-2025',
        warning='Exploratory fixed-path diagnostics, not selection-adjusted, not new market evidence. '
                'Circular blocks assume a usable approximately stationary sample; show every declared block length. '
                'Alpha uses one ETF factor, a fixed 3% hurdle and normal HAC intervals, not full factor attribution. '
                'Relative-wealth growth is not the difference of two standalone CAGRs.',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [curves_path, metrics_path]},
        lock_checks=hashes, folds=folds,
        original_model_parameters=records.groupby('model').parameters.first().to_dict(),
        primary_metrics=primary[fields].to_dict(orient='records'),
        think_mix_cost_stress=metrics[(metrics.model=='think_mix') & (metrics.seed=='ensemble') &
             (metrics.k==10)][['cost_multiplier','annualized_return','sharpe']].to_dict(orient='records'),
        comparisons=comparisons)
    result = clean(result)
    (OUT / 'audit_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
