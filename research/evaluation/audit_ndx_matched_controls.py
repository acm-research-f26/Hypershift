"""Independent scalar momentum, risk, account and metric checks for controls."""
import json
import math
import statistics
import time

import numpy as np
import pandas as pd

import run_ndx_matched_controls as study
from audit_ndx_drawdown_research import independent_weights, independent_exposure, independent_account
from ndx_matched_controls import etf_data


def momentum_independent(data, ids):
    scores = np.full((len(ids), len(data['panel'].tickers)), np.nan)
    mask = data['mask'][ids].copy()
    for row, i in enumerate(ids):
        t = data['origin'][i]
        for j in range(len(data['panel'].tickers)):
            if t >= 252 and data['tradable'][t-252,j] and data['tradable'][t-21,j]:
                scores[row,j] = data['marks'][t-21,j]/data['marks'][t-252,j]-1
        mask[row] &= np.isfinite(scores[row])
    copied = dict(data); copied['mask'] = data['mask'].copy(); copied['mask'][ids] = mask
    return independent_weights(copied, ids, scores, 'issuer5')


def etf_buyhold_independent(data, ticker, multiplier):
    root = study.ROOT/data['manifest']['root']
    prices = pd.read_csv(root/'etf_adj_close.csv', index_col=0)[ticker].to_numpy()
    volume = pd.read_csv(root/'etf_dollar_volume.csv', index_col=0)[ticker].to_numpy()
    first = data['panel'].dates.get_loc(pd.Timestamp('2022-01-03'))
    fee = lambda dollars, day: multiplier*(.0007+.0003*math.sqrt(abs(dollars)/np.mean(volume[day-20:day])/.01))
    dollars = 1e6/(1+fee(1e6,first)); shares = dollars/prices[first]
    cash = 1e6-dollars*(1+fee(1e6,first))
    nav = cash+shares*prices[first:]
    nav[-1] -= shares*prices[-1]*fee(shares*prices[-1], len(prices)-1)
    return nav


def main():
    started = time.monotonic(); plan = study.verify_plan()
    complete = json.loads((study.RUN/'COMPLETE.json').read_text())
    for relative, expected in complete['output_sha256'].items():
        if study.parent.sha(study.RUN/relative) != expected:
            raise ValueError('Result hash mismatch: '+relative)
    metrics = pd.read_csv(study.RUN/'metrics.csv').set_index('id')
    with np.load(study.RUN/'paths.npz') as saved:
        names = saved['ids'].tolist(); returns = saved['returns'].copy(); dates = saved['dates'].copy()
    if returns.shape != (1003,87) or len(metrics) != 87 or not metrics.index.is_unique:
        raise ValueError('Wrong account inventory')
    for col, key in enumerate(names):
        a = returns[:,col].tolist(); wealth = 1.; high = 1.; dd = 0.
        for value in a:
            wealth *= 1+value; high = max(high,wealth); dd = min(dd,wealth/high-1)
        actual = dict(sharpe=math.sqrt(252)*(math.fsum(a)/len(a)-math.expm1(math.log1p(.03)/252))/statistics.stdev(a),
                      annualized_return=math.expm1(math.fsum(map(math.log1p,a))*252/len(a)), maximum_drawdown=dd)
        for field, value in actual.items():
            np.testing.assert_allclose(value,metrics.loc[key,field],atol=1e-11,rtol=1e-11)
    data = study.parent.load_data(); ids = np.concatenate([study.parent.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    np.testing.assert_array_equal(dates,data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d'))
    scores = study.parent.saved_scores(data,ids); qqq = etf_data(data,'QQQ'); one = np.ones((len(ids),1))
    momentum = momentum_independent(data,ids)
    prepared = {}; production, support = study.prepared_accounts(data,ids,scores)
    for spec in plan['specs']:
        key = spec['id']
        if spec['kind'] in ['neural','qqq_replay']:
            w = independent_weights(data,ids,scores[spec['group']],spec['allocation'])
            ex = independent_exposure(data,ids,w,study.RULES[spec['rule']])
            prepared[key] = (data,w,ex) if spec['kind']=='neural' else (qqq,one,ex)
        elif spec['kind']=='momentum':
            prepared[key] = (data,momentum,independent_exposure(data,ids,momentum,study.RULES[spec['rule']]))
        elif spec['kind']=='qqq_own':
            prepared[key] = (qqq,one,independent_exposure(qqq,ids,one,study.RULES[spec['rule']]))
        if spec['kind'] != 'buy_hold':
            _,w,ex = prepared[key]
            np.testing.assert_allclose(w,production[key][1],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(ex,production[key][2],atol=1e-12,rtol=1e-12)
    largest = 0.; count = 0; parity = 0
    with np.load(study.parent.RUN/'paths.npz') as saved:
        old = dict(zip(saved['ids'].tolist(),saved['returns'].T))
    for spec in plan['specs']:
        for cost in plan['costs']:
            key = spec['id']+f'/cost{cost}'
            nav = etf_buyhold_independent(data,spec['ticker'],cost) if spec['kind']=='buy_hold' else independent_account(*(
                prepared[spec['id']][0], ids, prepared[spec['id']][1], prepared[spec['id']][2]), 'weekly', cost)
            expected = 1e6*np.cumprod(1+returns[:,names.index(key)])
            largest = max(largest,float(np.max(np.abs(nav-expected))))
            np.testing.assert_allclose(nav,expected,atol=1e-5,rtol=1e-11)
            if spec['kind']=='neural' and cost==1:
                np.testing.assert_allclose(returns[:,names.index(key)],old[spec['recipe']+'/'+spec['group']],atol=1e-13,rtol=1e-13)
                parity += 1
            count += 1
        print('Independent account verified:',spec['id'],flush=True)
    previous = study.ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/analysis_paths.npz'
    with np.load(previous) as saved:
        old = dict(zip(saved['ids'].tolist(),saved['returns'].T))
    for ticker in ['SPY','QQQ']:
        np.testing.assert_allclose(returns[:,names.index('buy_hold/'+ticker+'/cost1')],old['original/'+ticker+'/10'],atol=1e-13,rtol=1e-13)
    # Changing outcomes strictly after day t cannot change any signal through t.
    altered = dict(data); altered['marks'] = data['marks'].copy()
    cutoff = int(data['origin'][ids[len(ids)//2]]-21)
    altered['marks'][cutoff+1:] *= 9
    from ndx_matched_controls import momentum_scores
    first = momentum_scores(data,ids); second = momentum_scores(altered,ids)
    np.testing.assert_array_equal(first[data['origin'][ids]-21<=cutoff],second[data['origin'][ids]-21<=cutoff])
    # The frozen candidate and old confirmation gate have not been activated.
    policy = json.loads(study.POLICY.read_text())
    if policy['activation']['activated'] or study.parent.sha(study.ROOT/'latest/config/ndx_confirmation_gate_20261024.json') != policy['original_gate_sha256']:
        raise ValueError('Unexpected prospective activation')
    inference = pd.read_csv(study.RUN/'uncertainty.csv')
    if len(inference)!=24 or not np.isfinite(inference.select_dtypes('number')).all().all():
        raise ValueError('Incomplete inference')
    base = [name.removesuffix('/cost1') for name in names if name.endswith('/cost1')]
    matrix = np.column_stack([returns[:,names.index(name+'/cost1')] for name in base])
    for row in inference.itertuples():
        active = matrix[:,base.index(row.strategy)]-matrix[:,base.index(row.comparator)]
        np.testing.assert_allclose(row.annual_active_mean,active.mean()*252,atol=1e-12)
        if row.simultaneous_active_low > row.annual_active_mean or row.simultaneous_active_high < row.annual_active_mean:
            raise ValueError('Invalid simultaneous interval')
    study.parent.write(study.RUN/'AUDIT.json',dict(status='passed', independent_account_replays=count,
                       independently_recomputed_metrics=len(metrics), parent_neural_base_paths_reproduced=parity,
                       parent_etf_paths_reproduced=2, maximum_nav_difference_dollars=largest,
                       momentum_and_exposure_independently_rebuilt=True, causal_momentum_check=True,
                       matched_exposure_targets_verified=True, prospective_policy_unchanged_and_inactive=True,
                       inference_contrasts_verified=24, source_input_output_locks_verified=True,
                       seconds=time.monotonic()-started, audit_source_sha256=study.parent.sha(study.ROOT/'research/evaluation/audit_ndx_matched_controls.py'),
                       scope='Accounting, timing, controls and preservation audit; not independent market-data validation or prospective performance evidence'))
    print('Passed all 87 independent accounts; max NAV error',largest,flush=True)


if __name__ == '__main__':
    main()
