"""Attribute the already-inspected original/new retention ensemble discrepancy."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_ndx_drawdown_research as parent
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from ndx_seed_attribution import attribute_account

ROOT = parent.ROOT
OUT = ROOT/'docs/ndx_seed_attribution_20261029'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    data = parent.load_data(); combined = append_qqq(data, etf_data(data, 'QQQ'))
    source = ROOT/'runs/ndx_retention_replication_20261029'
    rows = []; errors = []; styles = []
    returns = pd.DataFrame(data['returns'])
    qqq = pd.Series(etf_data(data, 'QQQ')['returns'][:, 0])
    beta = np.column_stack([returns[j].rolling(60, min_periods=40).cov(qqq).to_numpy()
                           /qqq.rolling(60, min_periods=40).var().to_numpy() for j in returns])
    vol = returns.rolling(20).std().to_numpy()*np.sqrt(252)
    marks = pd.DataFrame(data['marks'])
    factors = dict(momentum_12minus1=(marks.shift(21)/marks.shift(252)-1).to_numpy(),
                   momentum63=(marks/marks.shift(63)-1).to_numpy(), volatility20=vol,
                   qqq_beta60=beta, log_adv=np.log1p(data['adv']))
    with np.load(source/'TARGETS.npz', allow_pickle=False) as targets, np.load(source/'PATHS.npz', allow_pickle=False) as saved:
        ids = targets['ids']; dates = saved['dates']
        for group in parent.GROUPS:
            key = f'{group}__retention__cap75__b03__c1'; weights = targets[key]
            actual, table, contributions, exposures = attribute_account(combined, ids, weights)
            error = float(np.max(np.abs(actual-saved[key]))); assert error < 1e-12
            table['group'] = group; rows.append(table); errors.append(dict(group=group, max_daily_error=error))
            np.savez_compressed(OUT/f'{group}_attribution.npz', dates=dates,
                tickers=np.asarray(combined['panel'].tickers, dtype=str), contributions=contributions,
                beginning_weights=exposures, returns=actual)
            long = weights[:, :-1]; long /= long.sum(axis=1, keepdims=True)
            for row, i in enumerate(ids):
                value = dict(group=group, date=str(data['panel'].dates[data['entry'][i]].date()))
                for name, factor in factors.items():
                    active = long[row] > 0
                    value[name] = float(long[row, active]@np.nan_to_num(factor[data['origin'][i], active]))
                styles.append(value)
    assets = pd.concat(rows, ignore_index=True); assets.to_csv(OUT/'ASSET_CONTRIBUTIONS.csv', index=False)
    style = pd.DataFrame(styles); style['year'] = pd.to_datetime(style.date).dt.year
    style.to_csv(OUT/'CAUSAL_STYLE_EXPOSURES.csv', index=False)
    (OUT/'AUDIT.json').write_text(json.dumps(dict(status='passed', accounts=errors,
        scope='Dollar-position replay and additive net arithmetic-return contribution; same saved targets/prices; attribution is descriptive and uses inspected test outcomes, not a causal forecast'), indent=2)+'\n')
    print('TOP CONTRIBUTORS')
    for group in ('original_five', 'new_five'):
        for year in (2023, 2025):
            block = assets[(assets.group == group)&(assets.year == year)]
            print(group, year, block.nlargest(6, 'annualized_arithmetic_contribution')[
                ['ticker','annualized_arithmetic_contribution','mean_beginning_weight']].round(4).to_string(index=False))
    print('STYLE EXPOSURES', style.groupby(['group','year'])[list(factors)].mean().round(4).to_string())
    print('max replay error', max(r['max_daily_error'] for r in errors))


if __name__ == '__main__': main()
