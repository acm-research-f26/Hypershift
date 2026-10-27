"""Evaluate, independently audit and publish the finite ranking-target experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ndx_alpaca_models import inputs, scale, make
from ndx_ranking_objectives import rank_ic, target_data
from pit_research_engine import load_data, fold, historical_groups, continuous_backtest, etf_backtest
from research_engine import performance, target_weights
from run_ndx_alpaca_training import ROOT, RUN as PARENT, feature_data, sha, write
from run_ndx_ranking import RUN, complete
from run_robust_research import ensemble_predictions, predict

DOCS = ROOT/'docs/ndx_ranking_20261020'
OBJECTIVE_LABELS = dict(raw_mse='Raw-return MSE',residual_mse='QQQ-residual MSE',residual_rank='QQQ-residual MSE + ranking',
                        ridge='Residual ridge',parent_mse_stop='Prior MSE-stopped THINK+mixer',monthly_basket='Monthly basket',SPY='SPY',QQQ='QQQ')


def source_lock(name,files):
    lock = {str(Path(p).resolve()):sha(p) for p in files}
    path = RUN/name
    if path.exists() and json.loads(path.read_text())!=lock:
        raise ValueError('Completed stage source changed: '+name)
    write(path,lock)
    return lock


def contrast_family(keys,financial=False):
    contrasts = []
    objectives = ['raw_mse','residual_mse','residual_rank']
    if financial:
        for view in ['daily_control','intraday_augmented']:
            for kind in objectives:
                for benchmark in ['SPY','QQQ']:
                    contrasts.append((f'{view}/{kind} minus {benchmark}',(view,kind),('control',benchmark)))
    for kind in objectives:
        contrasts.append((f'intraday minus daily: {kind}',('intraday_augmented',kind),('daily_control',kind)))
    for view in ['daily_control','intraday_augmented']:
        contrasts.append((f'{view}: residual MSE minus raw MSE',(view,'residual_mse'),(view,'raw_mse')))
        contrasts.append((f'{view}: ranking minus residual MSE',(view,'residual_rank'),(view,'residual_mse')))
    if any(left not in keys or right not in keys for _,left,right in contrasts):
        raise ValueError('Incomplete contrast family')
    return contrasts


def joint_intervals(paths,contrasts,blocks,repeats=10000,annualizer=1.):
    changes = np.column_stack([np.asarray(paths[a])-np.asarray(paths[b]) for _,a,b in contrasts])
    if not np.isfinite(changes).all():
        raise ValueError('Nonfinite paired inference sample')
    observed = changes.mean(0)*annualizer
    n = len(changes)
    rows = []
    for block in blocks:
        rng = np.random.default_rng(20261020+block)
        sampled = []
        for start in range(0,repeats,100):
            size = min(100,repeats-start)
            starts = rng.integers(0,n,(size,int(np.ceil(n/block))))
            indices = ((starts[...,None]+np.arange(block))%n).reshape(size,-1)[:,:n]
            sampled.append(changes[indices].mean(1)*annualizer)
        sampled = np.concatenate(sampled)
        error = sampled.std(0,ddof=1)
        critical = float(np.quantile(np.max(np.abs((sampled-observed)/np.maximum(error,1e-12)),axis=1),.95))
        for j,(name,left,right) in enumerate(contrasts):
            rows.append(dict(contrast=name,left_view=left[0],left_model=left[1],right_view=right[0],right_model=right[1],
                block_length=block,replicates=repeats,observations=n,estimate=observed[j],
                paired_low=np.quantile(sampled[:,j],.025),paired_high=np.quantile(sampled[:,j],.975),
                simultaneous_low=observed[j]-critical*error[j],simultaneous_high=observed[j]+critical*error[j],
                family_comparisons=len(contrasts),max_statistic_critical=critical,
                interpretation='Exploratory fixed-path circular-block approximation; no correction for prior research or training uncertainty.'))
    return pd.DataFrame(rows)


def ridge_forecasts(data,target,view,ids,plan):
    feature = feature_data()[0] if view=='intraday_augmented' else None
    raw = inputs(data,view,feature)
    predictions = []
    for year in plan['test_years']:
        tr,va,te = fold(data,year)
        x,_,_ = scale(raw,data['mask'],tr)
        design = x.transpose(0,2,1,3).reshape(len(x),len(data['panel'].tickers),-1)
        design = np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
        a = design[tr][data['labelmask'][tr]]
        y = target['residual'][tr][data['labelmask'][tr]]
        penalty = np.eye(a.shape[-1])*10.; penalty[-1,-1]=0.
        coefficients = np.linalg.solve(a.T@a+penalty,a.T@y)
        folder = RUN/'ridge'/view; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'{year}.npz',coefficients=coefficients,test=design[te]@coefficients,validation=design[va]@coefficients)
        predictions.append(design[te]@coefficients)
    return np.concatenate(predictions)


def evaluate():
    plan = complete()
    if (RUN/'EVALUATED.json').exists():
        raise ValueError('Evaluation already complete; use saved outputs')
    tests = json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    if (tests['status']!='passed' or sha(ROOT/'test_ndx_ranking_report.py')!=tests['test_source_sha256'] or
        sha(__file__)!=tests['report_source_sha256']):
        raise ValueError('Evaluation tests must pass before test metrics')
    source_lock('EVALUATION_SOURCE_LOCK.json',[__file__,ROOT/'test_ndx_ranking_report.py'])
    source_lock('PARENT_REFERENCE_LOCK.json',[PARENT/name for name in ['metrics.csv','curves.csv','portfolio_scores.npz','AUDIT.json']])
    data = load_data(); target = target_data(data)
    ids = np.concatenate([fold(data,y)[2] for y in plan['test_years']])
    forecasts,records,raw_archive = {},[],{}
    for view in plan['views']:
        for kind in plan['objectives']:
            available = []
            for seed in plan['seeds']:
                parts = []
                for year in plan['test_years']:
                    folder = RUN/view/str(year)
                    with np.load(folder/f'{kind}_{seed}_predictions.npz') as saved:
                        parts.append(saved['test'].copy())
                    records.append(json.loads((folder/f'{kind}_{seed}_record.json').read_text()))
                prediction = np.concatenate(parts)
                available.append(prediction)
                raw_archive[f'{view}__{kind}__{seed}'] = prediction
            forecasts[view,kind] = available
        forecasts[view,'ridge'] = [ridge_forecasts(data,target,view,ids,plan)]
    metric_rows,curve_rows,ledgers,rank_rows,scores,primary,rank_paths = [],[],[],[],{}, {},{}
    def record(view,kind,seed,mult,metrics,path,ledger=None):
        tags = dict(view=view,model=kind,seed=str(seed),k=10,cost_multiplier=mult)
        metric_rows.append(dict(**tags,**metrics)); curve_rows.append(path.assign(**tags))
        if str(seed)=='ensemble' and mult==1:
            primary[view,kind] = path.net.to_numpy()
            if ledger is not None:
                ledgers.append(ledger.assign(**tags))
    for (view,kind),available in forecasts.items():
        score = ensemble_predictions(available,data['mask'][ids]) if len(available)>1 else available[0]
        scores[view+'__'+kind] = score
        residual_ic = rank_ic(score,target['residual'][ids],data['labelmask'][ids])
        raw_ic = rank_ic(score,target['raw'][ids],data['labelmask'][ids])
        rank_paths[view,kind] = residual_ic[np.isfinite(residual_ic)]
        for j,i in enumerate(ids):
            weights = target_weights(score[j],data['mask'][i],10,'equal')
            selected = weights>0
            fully_observed = bool(data['labelmask'][i,selected].all())
            rank_rows.append(dict(view=view,model=kind,signal_date=str(data['panel'].dates[data['origin'][i]].date()),
                entry_date=str(data['panel'].dates[data['entry'][i]].date()),raw_rankic=raw_ic[j],residual_rankic=residual_ic[j],
                observed_labels=int(data['labelmask'][i].sum()),eligible=int(data['mask'][i].sum()),
                top10_residual_pp=float(weights@target['residual'][i]) if fully_observed else np.nan,
                top10_return_pp=float(weights@target['raw'][i]) if fully_observed else np.nan,
                selected_beta=float(weights@target['beta'][i]),selected_labels_complete=fully_observed))
        for mult in plan['sensitivities']['cost_multipliers']:
            metric,path,ledger = continuous_backtest(data,ids,score,10,mult)
            record(view,kind,'ensemble',mult,metric,path,ledger)
        if len(available)>1:
            for seed,prediction in zip(plan['seeds'],available):
                metric,path,ledger = continuous_backtest(data,ids,prediction,10,1)
                record(view,kind,seed,1,metric,path)
        print('Evaluated',view,kind,flush=True)
    parent_curves = pd.read_csv(PARENT/'curves.csv',dtype={'seed':str})
    parent_metrics = pd.read_csv(PARENT/'metrics.csv',dtype={'seed':str})
    with np.load(PARENT/'portfolio_scores.npz') as saved:
        np.testing.assert_array_equal(ids,saved['ids'])
        for view in plan['views']:
            scores[view+'__parent_mse_stop'] = saved[view+'__think_mix'].copy()
            for mult in plan['sensitivities']['cost_multipliers']:
                choose = lambda frame: frame[(frame.view==view)&(frame.model=='think_mix')&(frame.seed=='ensemble')&
                                            (frame.k==10)&(frame.cost_multiplier==mult)]
                metric = choose(parent_metrics).iloc[0].drop(['view','model','seed','k','cost_multiplier']).to_dict()
                path = choose(parent_curves)[['date','net']].copy()
                record(view,'parent_mse_stop','ensemble',mult,metric,path)
    for mult in plan['sensitivities']['cost_multipliers']:
        for ticker in ['SPY','QQQ']:
            metric,path = etf_backtest(data,ticker,mult)
            record('control',ticker,'ensemble',mult,metric,path)
        zero = np.zeros_like(data['y'][ids])
        metric,path,ledger = continuous_backtest(data,ids,zero,10,mult,basket=True)
        record('control','monthly_basket','ensemble',mult,metric,path,ledger)
    metrics = pd.DataFrame(metric_rows); curves = pd.concat(curve_rows,ignore_index=True)
    ranks = pd.DataFrame(rank_rows)
    metrics.to_csv(RUN/'metrics.csv',index=False); curves.to_csv(RUN/'curves.csv',index=False)
    ranks.to_csv(RUN/'ranking_periods.csv',index=False)
    ranks.groupby(['view','model'])[['raw_rankic','residual_rankic','top10_residual_pp','top10_return_pp','selected_beta']].mean().to_csv(RUN/'ranking_summary.csv')
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'primary_ledgers.csv',index=False)
    pd.DataFrame(records).to_csv(RUN/'training_records.csv',index=False)
    np.savez_compressed(RUN/'scores.npz',ids=ids,**scores)
    np.savez_compressed(RUN/'raw_forecasts.npz',ids=ids,**raw_archive)
    joint_intervals(rank_paths,contrast_family(rank_paths),[1,4,12]).to_csv(RUN/'rank_intervals.csv',index=False)
    joint_intervals(primary,contrast_family(primary,True),[5,20,60],annualizer=252.).to_csv(RUN/'active_intervals.csv',index=False)
    annual,monthly,seeds,exposure = [],[],[],[]
    fields = ['view','model','seed','k','cost_multiplier']
    for key,path in curves.groupby(fields,sort=False):
        tags = dict(zip(fields,key)); dates = pd.to_datetime(path.date)
        for year,part in path.groupby(dates.dt.year):
            annual.append(dict(**tags,year=int(year),**performance(part.net)))
        for month,part in path.groupby(dates.dt.to_period('M').astype(str)):
            monthly.append(dict(**tags,month=month,net_return=float(np.prod(1+part.net)-1)))
    benchmark_cagr = metrics[(metrics.model=='QQQ')&(metrics.cost_multiplier==1)].annualized_return.iloc[0]
    for (view,kind),part in metrics[metrics.seed!='ensemble'].groupby(['view','model']):
        seeds.append(dict(view=view,model=kind,minimum_cagr=part.annualized_return.min(),median_cagr=part.annualized_return.median(),
                          maximum_cagr=part.annualized_return.max(),seeds_beating_qqq=int((part.annualized_return>benchmark_cagr).sum())))
    for (view,kind),path in primary.items():
        for benchmark in ['SPY','QQQ']:
            reference = primary['control',benchmark]
            exposure.append(dict(view=view,model=kind,benchmark=benchmark,
                tracking_error_annual=float(np.std(path-reference,ddof=1)*np.sqrt(252)),
                beta=float(np.cov(path,reference,ddof=1)[0,1]/np.var(reference,ddof=1)),
                mean_active_annual=float(np.mean(path-reference)*252)))
    for name,rows in [('annual',annual),('monthly',monthly),('seed_summary',seeds),('exposure',exposure)]:
        pd.DataFrame(rows).to_csv(RUN/(name+'.csv'),index=False)
    if len(metrics)!=69 or len(next(iter(rank_paths.values())))!=200:
        raise ValueError('Unexpected experiment inventory')
    valid = data['labelmask'][ids]
    write(RUN/'EVALUATED.json',dict(status='completed_pending_independent_audit',neural_fits=120,ridge_fits=8,
        continuous_accounts=len(metrics),complete_ranking_origins=200,beta_supported_test_labels=int(target['beta_supported'][ids][valid].sum()),
        beta_test_labels=int(valid.sum()),history_status='exploratory, already inspected',future_confirmation='reserved, not observed'))


def audit():
    import torch
    from audit_historical_ndx import replay, independent_metrics
    from audit_historical_ndx_ranking import average_ranks, diagnostics
    from report_ndx_alpaca import check_complete as parent_complete
    from training_device import configure_device
    plan = complete(); parent_complete()
    if not (RUN/'EVALUATED.json').exists():
        raise ValueError('Evaluation must precede independent audit')
    for lockname in ['EVALUATION_SOURCE_LOCK.json','PARENT_REFERENCE_LOCK.json']:
        for name,value in json.loads((RUN/lockname).read_text()).items():
            if sha(name)!=value:
                raise ValueError('Evaluation or reference changed')
    audit_lock = source_lock('AUDIT_SOURCE_LOCK.json',[__file__,ROOT/'audit_historical_ndx.py',ROOT/'audit_historical_ndx_ranking.py'])
    data = load_data(); target = target_data(data)
    with np.load(RUN/'targets.npz') as saved:
        for name,value in target.items():
            np.testing.assert_array_equal(saved[name],value)
    # Independent paired OLS, using only information available at each signal.
    q = pd.read_csv(Path(data['manifest']['root'])/'etf_adj_close.csv')['QQQ'].to_numpy()
    qr = np.r_[np.nan,q[1:]/q[:-1]-1]
    rng = np.random.default_rng(20261020)
    beta_checks = []
    for i,node in zip(rng.integers(0,len(data['origin']),80),rng.integers(0,len(data['panel'].tickers),80)):
        day = data['origin'][i]; begin = max(0,day-125)
        a,b = data['returns'][begin:day+1,node],qr[begin:day+1]
        known = np.isfinite(a)&np.isfinite(b); a,b = a[known],b[known]
        expected,supported = 1.,False
        if len(a)>=60 and np.var(b)>1e-14:
            centered = b-b.mean()
            slope = float(centered@(a-a.mean())/(centered@centered))
            expected = float(np.clip(.5+.5*slope,0,2)); supported=True
        np.testing.assert_allclose(target['beta'][i,node],expected,atol=1e-10,rtol=1e-10)
        if bool(target['beta_supported'][i,node])!=supported:
            raise ValueError('Causal beta support mismatch')
        beta_checks.append(dict(signal_date=str(data['panel'].dates[day].date()),security=data['panel'].tickers[node],
                                paired_past_sessions=len(a),expected_beta=expected))
    pd.DataFrame(beta_checks).to_csv(RUN/'independent_beta_samples.csv',index=False)
    ids = np.concatenate([fold(data,y)[2] for y in plan['test_years']])
    forecasts,checked = {},0
    device = configure_device('cuda')
    for view in plan['views']:
        raw = inputs(data,view,feature_data()[0] if view=='intraday_augmented' else None)
        by_kind = {kind:[[] for _ in plan['seeds']] for kind in plan['objectives']}
        ridge_parts = []
        for year in plan['test_years']:
            tr,va,te = fold(data,year); x,mean,std = scale(raw,data['mask'],tr)
            folder = RUN/view/str(year)
            with np.load(folder/'preprocessing.npz') as saved:
                for name,value in dict(train=tr,val=va,test=te,mean=mean,std=std).items():
                    np.testing.assert_array_equal(saved[name],value)
            altered = raw.copy(); altered[max(tr)+1:]=1e10
            _,other_mean,other_std = scale(altered,data['mask'],tr)
            np.testing.assert_array_equal(mean,other_mean); np.testing.assert_array_equal(std,other_std)
            groups = historical_groups(data,tr)
            for kind in plan['objectives']:
                for s,seed in enumerate(plan['seeds']):
                    checkpoint = torch.load(folder/f'{kind}_{seed}.pt',map_location='cpu',weights_only=False)
                    if checkpoint['seed']!=seed or checkpoint['kind']!=kind or checkpoint['groups']!=[sorted(g) for g in groups]:
                        raise ValueError('Checkpoint metadata mismatch')
                    model = make('think_mix',len(data['panel'].tickers),groups).to(device)
                    model.load_state_dict(checkpoint['state_dict'])
                    vp,tp = [predict(model,x,data['mask'],sub,plan['batch']) for sub in [va,te]]
                    with np.load(folder/f'{kind}_{seed}_predictions.npz') as saved:
                        np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0)
                        np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
                    history = json.loads((folder/f'{kind}_{seed}_history.json').read_text())
                    best,chosen = -np.inf,0
                    for item in history:
                        if item['validation_residual_rankic']>best+1e-7:
                            best,chosen = item['validation_residual_rankic'],item['epoch']
                    if checkpoint['epoch']!=chosen:
                        raise ValueError('Earliest best ranking checkpoint not restored')
                    independent = [diagnostics(vp[j,m],target['residual'][i,m])['RankIC']
                                   for j,i in enumerate(va) for m in [data['labelmask'][i]]]
                    np.testing.assert_allclose(np.nan_to_num(independent).mean(),best,atol=1e-12,rtol=0)
                    by_kind[kind][s].append(tp); checked+=1
            design = x.transpose(0,2,1,3).reshape(len(x),len(data['panel'].tickers),-1)
            design = np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
            a = design[tr][data['labelmask'][tr]]; y = target['residual'][tr][data['labelmask'][tr]]
            penalty = np.eye(a.shape[-1])*10.; penalty[-1,-1]=0.
            coefficients = np.linalg.solve(a.T@a+penalty,a.T@y)
            with np.load(RUN/'ridge'/view/f'{year}.npz') as saved:
                np.testing.assert_allclose(coefficients,saved['coefficients'],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(design[te]@coefficients,saved['test'],atol=1e-9,rtol=1e-9)
            ridge_parts.append(design[te]@coefficients)
            print(f'Reproduced {view} {year}: {checked}/120 checkpoints',flush=True)
        for kind,available in by_kind.items():
            forecasts[view,kind] = [np.concatenate(parts) for parts in available]
        forecasts[view,'ridge'] = [np.concatenate(ridge_parts)]
        parent_parts = []
        for seed in plan['seeds']:
            pieces = []
            for year in plan['test_years']:
                with np.load(PARENT/view/str(year)/f'think_mix_{seed}_predictions.npz') as saved:
                    pieces.append(saved['test'].copy())
            parent_parts.append(np.concatenate(pieces))
        forecasts[view,'parent_mse_stop'] = parent_parts
    independent_scores = {}
    ranks = pd.read_csv(RUN/'ranking_periods.csv').set_index(['view','model','signal_date'])
    with np.load(RUN/'scores.npz') as saved:
        np.testing.assert_array_equal(ids,saved['ids'])
        for key,available in forecasts.items():
            if len(available)==1:
                score = available[0]
            else:
                ranked = []
                for p in available:
                    values = np.zeros_like(p,dtype=float)
                    for j,m in enumerate(data['mask'][ids]):
                        values[j,m] = average_ranks(p[j,m])/m.sum()
                    ranked.append(values)
                score = np.mean(ranked,axis=0)
            independent_scores[key] = score
            np.testing.assert_allclose(score,saved['__'.join(key)],atol=1e-12,rtol=0)
            if key[1]=='parent_mse_stop':
                continue
            for j,i in enumerate(ids):
                m = data['labelmask'][i]
                if m.sum()<2:
                    continue
                row = ranks.loc[(*key,str(data['panel'].dates[data['origin'][i]].date()))]
                for label in ['raw','residual']:
                    value = diagnostics(score[j,m],target[label][i,m])['RankIC']
                    np.testing.assert_allclose(np.nan_to_num(value),row[label+'_rankic'],atol=1e-12,rtol=0)
    metrics = pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}).set_index(['view','model','seed','k','cost_multiplier'])
    curves = pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    largest,accounts = 0.,0
    for key,path in curves.groupby(['view','model','seed','k','cost_multiplier'],sort=False):
        view,kind,seed,k,mult = key
        if len(path)!=1003 or not path.date.is_unique or not path.date.is_monotonic_increasing:
            raise ValueError('Account return calendar mismatch')
        for name,value in independent_metrics(path.net).items():
            np.testing.assert_allclose(value,metrics.loc[key,name],atol=1e-10,rtol=1e-10)
        if kind in ['SPY','QQQ']:
            root = Path(data['manifest']['root'])
            price = pd.read_csv(root/'etf_adj_close.csv')[kind].to_numpy()
            volume = pd.read_csv(root/'etf_dollar_volume.csv')[kind].to_numpy()
            first = data['entry'][ids[0]]
            rate = mult*(.0007+.0003*np.sqrt(1e6/volume[first-20:first].mean()/.01))
            nav = 1e6/(1+rate)/price[first]*price[first:]
            nav[-1]*=1-mult*(.0007+.0003*np.sqrt(nav[-1]/volume[-21:-1].mean()/.01))
        else:
            score = (np.zeros_like(data['y'][ids]) if kind=='monthly_basket' else independent_scores[view,kind]
                     if seed=='ensemble' else forecasts[view,kind][plan['seeds'].index(int(seed))])
            nav = replay(data,ids,score,k,mult,basket=kind=='monthly_basket')
        published = 1e6*np.cumprod(1+path.net.to_numpy())
        largest = max(largest,float(abs(nav-published).max()))
        np.testing.assert_allclose(nav,published,atol=1e-5,rtol=1e-10)
        accounts+=1
        if accounts%15==0:
            print(f'Independently replayed {accounts}/69 accounts',flush=True)
    ledger = pd.read_csv(RUN/'primary_ledgers.csv')
    np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    if ledger.cash.min()<-1e-5 or accounts!=69 or checked!=120:
        raise ValueError('Incomplete or invalid account audit')
    for name,value in audit_lock.items():
        if sha(name)!=value:
            raise ValueError('Audit source changed during verification')
    write(RUN/'AUDIT.json',dict(status='passed',new_checkpoints_reproduced=checked,ridge_models_reproduced=8,
        parent_forecast_exports_reconstructed=40,independent_accounts=accounts,ranking_ensembles_reconstructed=len(independent_scores),
        independent_beta_samples=len(beta_checks),largest_nav_difference_dollars=largest,
        future_confirmation='not evaluated; no untouched historical claim'))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def pct(x):
    return f'{x*100:.1f}%'


def label(view,kind):
    prefix = 'Daily' if view=='daily_control' else 'Intraday' if view=='intraday_augmented' else 'Control'
    return prefix+' / '+OBJECTIVE_LABELS[kind]


def publish():
    complete()
    audit_record = json.loads((RUN/'AUDIT.json').read_text())
    if audit_record['status']!='passed':
        raise ValueError('Independent audit required for publication')
    for name in ['EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json']:
        for path,value in json.loads((RUN/name).read_text()).items():
            if sha(path)!=value:
                raise ValueError('Audited source changed')
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    metrics = pd.read_csv(RUN/'metrics.csv',dtype={'seed':str})
    curves = pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    primary = metrics[(metrics.seed=='ensemble')&(metrics.cost_multiplier==1)].set_index(['view','model'])
    paths = curves[(curves.seed=='ensemble')&(curves.cost_multiplier==1)]
    ranks = pd.read_csv(RUN/'ranking_summary.csv').set_index(['view','model'])
    ranking_periods = pd.read_csv(RUN/'ranking_periods.csv')
    annual = pd.read_csv(RUN/'annual.csv',dtype={'seed':str})
    monthly = pd.read_csv(RUN/'monthly.csv',dtype={'seed':str})
    seeds = pd.read_csv(RUN/'seed_summary.csv').set_index(['view','model'])
    exposure = pd.read_csv(RUN/'exposure.csv').set_index(['view','model','benchmark'])
    rank_ci = pd.read_csv(RUN/'rank_intervals.csv'); rank_ci = rank_ci[rank_ci.block_length==4]
    active_ci = pd.read_csv(RUN/'active_intervals.csv'); active_ci = active_ci[active_ci.block_length==20]
    evaluated = json.loads((RUN/'EVALUATED.json').read_text())
    tests = json.loads((RUN/'UNIT_TESTS.json').read_text())
    evaluation_tests = json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    DOCS.mkdir(parents=True,exist_ok=True); figures = DOCS/'figures'; figures.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(raw_mse='#1f77b4',residual_mse='#ff7f0e',residual_rank='#2ca02c',ridge='#999999',parent_mse_stop='#9467bd')
    upper = max(float(np.cumprod(1+path.net).max()) for _,path in paths.groupby(['view','model']))*1.05
    for view in plan['views']:
        fig,ax = plt.subplots(figsize=(12,7))
        for kind in plan['objectives']+['ridge','parent_mse_stop']:
            path = paths[(paths.view==view)&(paths.model==kind)]
            m = primary.loc[view,kind]
            ax.plot(pd.to_datetime(path.date),np.cumprod(1+path.net),color=colors[kind],linewidth=1.8,
                    label=f'{OBJECTIVE_LABELS[kind]}: CAGR {pct(m.annualized_return)}, SR {m.sharpe:.3f}')
        for kind,color,style in [('SPY','black','--'),('QQQ','#333399','--'),('monthly_basket','#888888',':')]:
            path = paths[(paths.view=='control')&(paths.model==kind)]
            m = primary.loc['control',kind]
            ax.plot(pd.to_datetime(path.date),np.cumprod(1+path.net),color=color,linestyle=style,linewidth=2.,
                    label=f'{OBJECTIVE_LABELS[kind]}: CAGR {pct(m.annualized_return)}')
        ax.set(title=label(view,'residual_rank').split(' / ')[0]+' inputs: fixed top 10, exploratory 2022–2025',
               ylabel='Net wealth per initial $1',ylim=(0,upper))
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper left'); fig.tight_layout()
        fig.savefig(figures/(view+'_wealth.png'),dpi=160); plt.close(fig)
    for frame,filename,title,scale_factor in [
        (rank_ci,'ranking_contrasts.png','Exploratory residual RankIC changes',1.),
        (active_ci[active_ci.right_model=='QQQ'],'qqq_active_intervals.png','Exploratory annual active return versus QQQ (percentage points)',100.)]:
        fig,ax = plt.subplots(figsize=(13,7))
        for j,row in enumerate(frame.itertuples(index=False)):
            ax.plot(np.array([row.simultaneous_low,row.simultaneous_high])*scale_factor,[j,j],color='#cccccc',linewidth=5)
            ax.plot(np.array([row.paired_low,row.paired_high])*scale_factor,[j,j],color='#1f77b4',linewidth=2)
            ax.scatter([row.estimate*scale_factor],[j],color='#1f77b4')
        names = frame.contrast.str.replace('intraday_augmented','Intraday').str.replace('daily_control','Daily')
        ax.set(yticks=np.arange(len(frame)),yticklabels=names,title=title,
               xlabel='Blue: paired 95%; gray: simultaneous diagnostic within the declared family')
        ax.axvline(0,color='black',linewidth=1); ax.grid(axis='x',alpha=.2); fig.tight_layout()
        fig.savefig(figures/filename,dpi=160); plt.close(fig)
    rank_rows,return_rows,seed_rows,cost_rows,annual_rows,month_rows = [],[],[],[],[],[]
    base_months = monthly[(monthly.seed=='ensemble')&(monthly.cost_multiplier==1)]
    qq = base_months[base_months.model=='QQQ'].set_index('month').net_return
    sp = base_months[base_months.model=='SPY'].set_index('month').net_return
    for view in plan['views']:
        for kind in plan['objectives']+['ridge','parent_mse_stop']:
            m = primary.loc[view,kind]; e = exposure.loc[view,kind,'QQQ']
            return_rows.append([label(view,kind),pct(m.annualized_return),f'{m.sharpe:.3f}',pct(m.maximum_drawdown),
                                pct(e.tracking_error_annual),f'{e.beta:.2f}'])
            if kind!='parent_mse_stop':
                rank = ranks.loc[view,kind]
                rank_rows.append([label(view,kind),f'{rank.raw_rankic:+.4f}',f'{rank.residual_rankic:+.4f}',
                                  f'{rank.top10_residual_pp:+.3f} pp',f'{rank.selected_beta:.2f}'])
            if kind in plan['objectives']:
                seed = seeds.loc[view,kind]
                seed_rows.append([label(view,kind),pct(seed.minimum_cagr),pct(seed.median_cagr),pct(seed.maximum_cagr),
                                  f'{int(seed.seeds_beating_qqq)}/5'])
                costs = metrics[(metrics.view==view)&(metrics.model==kind)&(metrics.seed=='ensemble')].set_index('cost_multiplier')
                cost_rows.append([label(view,kind)]+[pct(costs.loc[c,'annualized_return']) for c in [1,2,4]]+
                                 [f'{costs.loc[1,"turnover_annual"]:.1f}×'])
                months = base_months[(base_months.view==view)&(base_months.model==kind)].set_index('month').net_return
                month_rows.append([label(view,kind),f'{int((months>qq).sum())}/48',f'{int((months>sp).sum())}/48'])
                ys = annual[(annual.view==view)&(annual.model==kind)&(annual.seed=='ensemble')&
                            (annual.cost_multiplier==1)].set_index('year')
                annual_rows.append([label(view,kind)]+[pct(ys.loc[y,'cumulative_return']) for y in plan['test_years']])
    for kind in ['SPY','QQQ','monthly_basket']:
        m = primary.loc['control',kind]
        return_rows.append([OBJECTIVE_LABELS[kind],pct(m.annualized_return),f'{m.sharpe:.3f}',pct(m.maximum_drawdown),'—','—'])
        ys = annual[(annual.model==kind)&(annual.seed=='ensemble')&(annual.cost_multiplier==1)].set_index('year')
        annual_rows.append([kind]+[pct(ys.loc[y,'cumulative_return']) for y in plan['test_years']])
    rank_contrasts = [[r.contrast,f'{r.estimate:+.4f}',f'[{r.paired_low:+.4f}, {r.paired_high:+.4f}]',
                       f'[{r.simultaneous_low:+.4f}, {r.simultaneous_high:+.4f}]'] for r in rank_ci.itertuples(index=False)]
    active_contrasts = [[r.contrast,f'{r.estimate*100:+.2f} pp',f'[{r.paired_low*100:+.2f}, {r.paired_high*100:+.2f}] pp',
                         f'[{r.simultaneous_low*100:+.2f}, {r.simultaneous_high*100:+.2f}] pp'] for r in active_ci.itertuples(index=False)]
    changes = {}
    for kind in plan['objectives']:
        changes[kind] = float(primary.loc['intraday_augmented',kind]['annualized_return']-primary.loc['daily_control',kind]['annualized_return'])
    positive_etf = active_ci[(active_ci.right_model.isin(['SPY','QQQ']))&(active_ci.simultaneous_low>0)]
    positive_rank = rank_ci[rank_ci.simultaneous_low>0]
    label_counts = evaluated['beta_test_labels']; supported = evaluated['beta_supported_test_labels']
    lines = [
        '# NASDAQ-100: benchmark-residual targets and ranking-aware validation',
        '',
        '**Complete: 120 new neural fits, eight residual-ridge fits and 69 continuous accounts; independent audit passed.**',
        '',
        'This finite six-cell experiment compares daily versus intraday inputs under raw-return MSE, '
        'QQQ-beta-residual MSE, and residual MSE plus a fixed ranking loss. Every neural checkpoint is selected '
        'by prior-year residual RankIC. Existing reviewed membership, the 13-feature THINK+mixer architecture, '
        'five seeds, five-session horizon, next-close execution and equal top-10 policy are held fixed.',
        '',
        '**2022–2025 had already been inspected before this experiment. These are exploratory development '
        'results. No existing historical dates have been relabeled as untouched confirmation.**',
        '',
        f'{len(positive_rank)}/7 ranking contrasts and {len(positive_etf)}/12 model-versus-ETF active-return '
        'contrasts have positive simultaneous diagnostic lower bounds in their respective primary families. '
        'The families are separate, and neither accounts for the earlier research search or training uncertainty.',
        '',
        '## What changed and why',
        '',
        'At each signal close, estimate each stock’s QQQ beta from at most 126 prior daily returns, requiring '
        '60 paired observations. Shrink the OLS slope halfway toward one and clip to [0, 2]; use beta one '
        'when history or benchmark variance is insufficient. The residual label is the next-entry-close to '
        'five-session-exit stock economic return minus that signal-time beta times the matching QQQ return. '
        'Future QQQ returns are labels only; all split boundaries remain maturity-purged. '
        f'Beta history is supported for {supported:,}/{label_counts:,} observed test labels ({supported/label_counts:.2%}); '
        'fallbacks never remove eligible members.',
        '',
        'The ranking arm adds the mean logistic loss on unequal residual-return pairs, with fixed weight one '
        'and a fixed one-percentage-point score temperature. Each date receives equal loss weight. There is '
        'no rank-weight, beta-window, model-width, portfolio-size or feature-grid search. All three arms '
        'select the earliest best validation residual RankIC within 1e-7, with patience six and at most 30 '
        'epochs. Constant predictions have RankIC zero. Five eligible-stock rank forecasts are averaged '
        'for each ensemble; individual seeds remain disclosed.',
        '',
        'Subtracting one common benchmark return would leave cross-sectional label ranks unchanged. '
        'Using different causally estimated stock betas makes this an exposure-adjusted selection target. '
        'The resulting long-only portfolio is still not beta-hedged or constrained to benchmark weights.',
        '',
        'The prior MSE-stopped THINK+mixer rows are frozen references from the completed experiment. '
        'New raw-MSE models use equal date weights and ranking-based stopping; their difference from those '
        'references cannot be attributed solely to a change in checkpoint criterion.',
        '',
        '## Ranking quality',
        '',
        table(['Input / objective','Raw-return RankIC','Residual RankIC','Top-10 residual return / five sessions','Mean selected lagged beta'],rank_rows),
        '',
        'RankIC is a cross-sectional Spearman correlation, averaged across 200 complete forecast origins. '
        'Bucket returns are gross equal-target-weight label diagnostics, require every selected label to be '
        'observed, and are not the realized net account return. The final partial holding period remains in '
        'accounts but has no complete five-session ranking label.',
        '',
        table(['Ranking contrast','Mean RankIC change','Paired 95% interval','Seven-contrast simultaneous diagnostic'],rank_contrasts),
        '',
        '![Ranking contrasts](figures/ranking_contrasts.png)',
        '',
        '## Net portfolio results',
        '',
        table(['Input / objective','CAGR','Sharpe','Maximum drawdown','QQQ tracking error','Realized QQQ beta'],return_rows),
        '',
        'Accounts start at $1 million and carry cash/holdings continuously from January 3, 2022 through '
        'December 31, 2025. Costs are 7 bp plus 3 bp times the square root of participation divided by 1%; '
        'normal orders are capped at 1% of lagged 20-session average dollar volume. Frozen share orders '
        'fill next close, subject to liquidity and cash; unavailable fills cancel. Reviewed daily economic '
        'prices and acquisition settlement are retained. Terminal liquidation and 3% fixed Sharpe hurdle '
        'are inherited approximations; cash earns no interest. These are five-session strategies using '
        'session summaries derived from Alpaca 15-minute bars.',
        '',
        '![Daily inputs](figures/daily_control_wealth.png)',
        '',
        '![Intraday inputs](figures/intraday_augmented_wealth.png)',
        '',
        '## Consistency and cost sensitivity',
        '',
        table(['Input / objective','2022','2023','2024','2025'],annual_rows),
        '',
        table(['Input / objective','Months above QQQ','Months above SPY'],month_rows),
        '',
        table(['Input / objective','Minimum seed CAGR','Median seed CAGR','Maximum seed CAGR','Seeds above QQQ'],seed_rows),
        '',
        'Seeds share the same market path; their range is a sensitivity diagnostic. Ensemble CAGR is not '
        'the average seed CAGR. Annual accounts never reset, and monthly observations are not independent confirmations.',
        '',
        table(['Input / objective','1× cost CAGR','2× cost CAGR','4× cost CAGR','Base annual traded-notional/NAV'],cost_rows),
        '',
        '## Benchmark-relative uncertainty',
        '',
        '![QQQ active-return intervals](figures/qqq_active_intervals.png)',
        '',
        table(['Financial contrast','Annual arithmetic mean change','Paired 95% interval','19-contrast simultaneous diagnostic'],active_contrasts),
        '',
        'All intervals use 10,000 joint circular-block resamples, preserving contemporaneous dependence '
        'across compared strategies. Ranking inference samples whole five-session origins with blocks of '
        '1/4/12 origins; the displayed primary block is four. Financial inference samples whole daily account '
        'rows with 5/20/60-session blocks; the displayed primary block is 20. Arithmetic active means are '
        'not CAGR differences. The two max-statistic families do not provide joint confirmation across ranking '
        'and returns. Block stationarity, annual model changes, past research selection and training uncertainty '
        'limit inference. More bars do not add independent market years.',
        '',
        '## Untouched confirmation',
        '',
        '**Reserved; not started or evaluated.** The earliest permitted future signal is October 21, 2026. '
        'No future price or performance is available in this development panel. The frozen protocol requires '
        'updated historical membership, identities, corporate actions and both input views to pass review '
        'before launch. The fixed 156-node mixer needs an explicit, verified policy for newly admitted '
        'securities; it cannot silently omit them.',
        '',
        'After that gate, one of the six cells must be chosen using only median five-seed prior-year '
        'validation residual RankIC, with ties resolved in protocol order. Choice, data, schema, refit '
        'schedule and account rules must be locked before the first signal. The first formal performance '
        'read is after 252 future sessions, with no interim promotion. Even that first year may be '
        'underpowered. Positive conservative net active-return lower bounds versus both benchmarks and '
        'further confirmation are needed before claiming regular outperformance. No automation, future '
        'job or trading order was created.',
        '',
        '## Verification and limits',
        '',
        f'The training suite passed {tests["tests"]} tests; {evaluation_tests["tests"]} additional inference '
        'tests passed before evaluation. Independent checks reproduced all 120 new checkpoints and '
        'validation/test exports, eight ridge fits, 40 parent forecast exports, 80 causal beta samples '
        f'and all 69 accounts. Maximum NAV discrepancy: ${audit_record["largest_nav_difference_dollars"]:.8f}. '
        'Input/code hashes preserve the completed Alpaca and older studies.',
        '',
        'The prior Alpaca identity and coverage review is reused without changing inputs. Daily adjusted '
        'economic accounts are approximations to dividends, raw-share actions and execution. Dense '
        'ticker-specific mixers retain cold-start limitations. Short effective time history, concentrated '
        'selection, feature missingness and the already-inspected years remain material constraints. '
        'A favorable exploratory curve or conditional interval is not evidence of deployable future skill.',
        '',
        '## Artifacts',
        '',
        '[Protocol and future gates](PROTOCOL.json) · [Audit](AUDIT.json) · [All metrics](metrics.csv) · '
        '[Daily paths](curves.csv) · [Ranking by origin](ranking_periods.csv) · [Ranking summary](ranking_summary.csv) · '
        '[All ranking intervals](rank_intervals.csv) · [All active-return intervals](active_intervals.csv) · '
        '[Seed ranges](seed_summary.csv) · [Annual returns](annual.csv) · [Monthly returns](monthly.csv) · '
        '[Benchmark exposure](exposure.csv) · [Training records](training_records.csv) · [Independent beta checks](independent_beta_samples.csv)',
        '',
        'Full checkpoints, preprocessing, targets, source snapshots and integrity locks are saved under '
        '`runs/ndx_ranking_20261020`. The experiment can be resumed with '
        '`.\\.venv\\Scripts\\python.exe run_ndx_ranking.py --stage all --workers 2`.',
        ''
    ]
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    files = ['PROTOCOL.json','FREEZE.json','AUDIT.json','UNIT_TESTS.json','EVALUATION_UNIT_TESTS.json',
             'metrics.csv','curves.csv','ranking_periods.csv','ranking_summary.csv','rank_intervals.csv',
             'active_intervals.csv','seed_summary.csv','annual.csv','monthly.csv','exposure.csv',
             'training_records.csv','independent_beta_samples.csv','EVALUATED.json',
             'EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','PARENT_REFERENCE_LOCK.json']
    for name in files:
        shutil.copyfile(RUN/name,DOCS/name)
    for name in ['report_ndx_ranking.py','test_ndx_ranking_report.py','audit_historical_ndx.py','audit_historical_ndx_ranking.py']:
        shutil.copyfile(ROOT/name,RUN/'source_snapshot'/name)
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_development',new_neural_fits=120,
        residual_ridge_fits=8,independent_accounts=69,untouched_confirmation='reserved, not evaluated',
        intraday_minus_daily_cagr=changes,positive_rank_family_lower_bounds=len(positive_rank),
        positive_model_etf_family_lower_bounds=len(positive_etf),report=str(DOCS/'REPORT.md'),
        artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    print(json.dumps({k:v for k,v in json.loads((RUN/'FINAL_PUBLICATION.json').read_text()).items() if k!='artifact_hashes'},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['evaluate','audit','publish'])
    args = parser.parse_args()
    globals()[args.stage]()


if __name__=='__main__':
    main()
