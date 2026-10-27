"""Replay saved forecasts and independently account in adjusted share units."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, performance, training_groups, target_weights
from run_robust_research import scale_inputs, model_data, ensemble_predictions
from robust_engine import allocation, backtest

OUT = Path('docs/audit/backtests_20261010')
MODELS = ['momentum126','ridge','think','think_mix','think_huber','think_mix_huber']

def share_ledger(data, ids, scores, k, method):
    """Separate share/cash ledger: price units, orders, fills, then daily NAV."""
    marks = data['marks']; n = marks.shape[1]
    units = np.zeros(n); cash = 1e6; previous = np.zeros(n)
    rets = []; contributions = np.zeros(n); counts = np.zeros(n)
    for j, i in enumerate(ids):
        signal, entry, end = [data[key][i] for key in ['origin','entry','exit']]
        ps, pe = marks[signal], marks[entry]
        ps = np.nan_to_num(ps, nan=1.); pe = np.nan_to_num(pe, nan=1.)
        nav_before_fill = cash+units@pe
        signal_nav = cash+units@ps
        if method == 'buy_hold' and j:
            desired = units.copy()
        elif method in ['buy_hold','equal_universe']:
            w = data['mask'][i]/data['mask'][i].sum()
            desired = w*signal_nav/ps
        else:
            w = allocation(scores[j], data['mask'][i], k, method,
                           data['returns'][max(0,signal-59):signal+1], previous)
            previous = w.copy(); desired = w*signal_nav/ps
            counts += w > 0
        cap_units = np.nan_to_num(data['adv'][signal], nan=0.)*.01/ps
        order = np.clip(desired-units, -cap_units, cap_units)*(~data['stale'][entry])
        dollars = order*pe
        def fee(dollars):
            part = abs(dollars)/np.maximum(np.nan_to_num(data['adv'][signal], nan=1.),1.)
            return abs(dollars)@(.0007+.0003*np.sqrt(part/.01))
        charge = fee(dollars)
        buys = np.maximum(dollars,0.); available = cash-np.minimum(dollars,0.).sum()
        need = buys.sum()+charge
        if need > available and buys.sum() > 0:
            dollars -= buys*min((need-available)/buys.sum(),1.)
            charge = fee(dollars); order = dollars/pe
        units += order; cash -= dollars.sum()+charge
        prev_nav = cash+units@pe
        for day in range(entry+1,end+1):
            price = np.nan_to_num(marks[day],nan=1.)
            nav = cash+units@price
            denominator = nav_before_fill if day == entry+1 else prev_nav
            rets.append(nav/denominator-1)
            contributions += units*(price-np.nan_to_num(marks[day-1],nan=1.))/denominator
            prev_nav = nav
    final_dollars = units*np.nan_to_num(marks[end],nan=1.)
    part = abs(final_dollars)/np.maximum(np.nan_to_num(data['adv'][data['origin'][ids[-1]]],nan=1.),1.)
    charge = abs(final_dollars)@(.0007+.0003*np.sqrt(part/.01))
    rets[-1] = (1+rets[-1])*(1-charge/prev_nav)-1
    return np.asarray(rets), contributions, counts

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows, seed_rows, annual, attribution, checks = [], [], [], [], []
    studies = [('original', ['NYSE_recent','NASDAQ_recent'],
                {5:'runs/robust_2022_2025_final',1:'runs/daily_2022_2025_final'}),
               ('top100', ['NYSE_top100_20261010','NASDAQ100_20261010'],
                {5:'runs/largecap_20261010/five_session',1:'runs/largecap_20261010/daily'})]
    for study, datasets, roots in studies:
      for h, rootstr in roots.items():
        root = Path(rootstr)
        for dataset in datasets:
          data = prepare_panel(load_panel(dataset),horizon=h)
          accumulated = {}
          seed_paths = {}
          for year in [2022,2023,2024,2025]:
            folder = root/f'{dataset}_{year}'
            train,val,test = fold_indices(data,year)
            scale = np.load(folder/'preprocessing.npz'); x,mean,std = scale_inputs(data,train)
            for key,value in [('mean',mean),('std',std),('train',train),('validation',val),('test',test)]:
                np.testing.assert_array_equal(scale[key],value)
            assert data['exit'][train[-1]] < data['origin'][val[0]]
            assert data['exit'][val[-1]] < data['origin'][test[0]]
            for kind in MODELS:
                cache = np.load(folder/f'{kind}_predictions.npz'); d = model_data(data,kind)
                np.testing.assert_array_equal(cache['test_ids'],test)
                np.testing.assert_array_equal(cache['eligible'],d['mask'][test])
                np.testing.assert_array_equal(cache['truth'],data['y'][test])
                if kind == 'momentum126':
                    # Direct price ratios, independently of prepare_panel's feature expression.
                    direct = data['marks'][data['origin'][test]]/data['marks'][data['origin'][test]-126]-1
                    np.testing.assert_allclose(cache['test'][0],direct,equal_nan=True,atol=1e-12)
                elif kind == 'ridge':
                    design = x.transpose(0,2,1,3).reshape(len(x),len(data['panel'].tickers),-1)
                    design = np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
                    a=design[train][data['labelmask'][train]]; y=(data['y'][train]*100)[data['labelmask'][train]]
                    penalty=np.eye(a.shape[-1])*10; penalty[-1,-1]=0
                    coef=np.linalg.solve(a.T@a+penalty,a.T@y)
                    np.testing.assert_allclose(coef,np.load(folder/'ridge_coef.npy'),atol=1e-9)
                    np.testing.assert_allclose(cache['test'][0],design[test]@coef,atol=1e-9)
                score=ensemble_predictions(cache['test'],d['mask'][test]) if len(cache['test'])>1 else cache['test'][0]
                chosen=json.loads((folder/f'{kind}_policy.json').read_text())['selected']
                for policy,k,method,suffix in [('selected',chosen['k'],chosen['method'],'daily'),('fixed_k10',10,'equal','equal_k10_daily')]:
                    metric,daily,_=backtest(d,test,score,k,method)
                    saved=pd.read_csv(folder/f'{kind}_{suffix}.csv')
                    np.testing.assert_array_equal(saved.date,daily.date)
                    np.testing.assert_allclose(saved.net,daily.net,atol=2e-12,rtol=1e-10)
                    independent,contrib,count=share_ledger(d,test,score,k,method)
                    np.testing.assert_allclose(independent,daily.net,atol=2e-12,rtol=1e-10)
                    accumulated.setdefault((kind,policy),[]).append(daily)
                    annual.append(dict(study=study,dataset=dataset,horizon=h,year=year,model=kind,policy=policy,k=k,method=method,**performance(daily.net)))
                    if kind in ['ridge','momentum126'] and policy=='fixed_k10':
                        for ticker,pnl,selections in zip(data['panel'].tickers,contrib,count):
                            attribution.append(dict(study=study,dataset=dataset,horizon=h,year=year,model=kind,ticker=ticker,
                                                    arithmetic_return_contribution=pnl,selected_periods=selections))
                for seed,pred in zip([7,19,42] if len(cache['test'])>1 else [0],cache['test']):
                    _,daily,_=backtest(d,test,pred,10,'equal')
                    seed_paths.setdefault((kind,seed),[]).append(daily)
                checks.append(dict(study=study,dataset=dataset,horizon=h,year=year,model=kind,status='saved path and independent share ledger agree'))
            # Perturb ALL prices/volumes after first test signal; no past feature, graph, scale changes.
            panel=load_panel(dataset); cutoff=data['origin'][test[0]]+1
            panel.close[cutoff:] *= np.linspace(.1,10,len(panel.tickers)); panel.volume[cutoff:] *= 100
            other=prepare_panel(panel,horizon=h); prior=data['origin']<cutoff
            for key in ['x','mask','momentum','reversal']:
                np.testing.assert_allclose(data[key][prior],other[key][prior],equal_nan=True)
            _,m,s=scale_inputs(other,train)
            np.testing.assert_array_equal(mean,m); np.testing.assert_array_equal(std,s)
            assert training_groups(data,train)==training_groups(other,train)
          for (kind,policy),pieces in accumulated.items():
              path=pd.concat(pieces,ignore_index=True)
              assert path.date.is_unique and path.date.is_monotonic_increasing
              rows.append(dict(study=study,dataset=dataset,horizon=h,model=kind,policy=policy,**performance(path.net)))
          for (kind,seed),pieces in seed_paths.items():
              seed_rows.append(dict(study=study,dataset=dataset,horizon=h,model=kind,seed=seed,**performance(pd.concat(pieces).net)))
          print(f'Audited {study} h{h} {dataset}',flush=True)
    for filename,values in [('pooled.csv',rows),('annual.csv',annual),('seed_fixed_k10.csv',seed_rows),('attribution.csv',attribution)]:
        pd.DataFrame(values).to_csv(OUT/filename,index=False)
    result=dict(status='passed',replayed_model_folds=len(checks),independent_account_paths=len(checks)*2,
                checks=checks,known_biases=['Current 2026 constituents and market caps applied to 2022-2025 (future universe selection)',
                'Missing delisted issuer histories and partially verified identity','Previously viewed test years and repeated comparisons',
                'Synthetic fold compounding with omitted boundary dates'],
                scope='Audits forecast construction, saved masks/splits, training-only scalers, graph invariance, replay and independent share accounting; does not certify historical data or eliminate universe bias')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
