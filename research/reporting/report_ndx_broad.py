"""Full disclosure, independent reproduction and evidence gates for broad research."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from ndx_broad_models import prepare,splits,normalize,SharedNetwork,predict_network,stability
from ndx_policy_engine import policy_backtest,policy_replay
from ndx_ranking_objectives import rank_ic
from pit_research_engine import fold,etf_backtest
from research_engine import performance
from report_ndx_ranking import joint_intervals
from run_ndx_alpaca_training import ROOT,sha,write
from run_ndx_broad import RUN,complete,tree_prediction
from run_robust_research import ensemble_predictions,model_data

DOCS=ROOT/'docs/ndx_broad_20261020'
ORIGINAL=ROOT/'runs/nasdaq100_historical_monthly_20261011'
RANKING=ROOT/'runs/ndx_ranking_20261020'


def source_lock(name,files):
    value={str(Path(p).resolve()):sha(p) for p in files}
    path=RUN/name
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Frozen evaluation source/reference changed')
    write(path,value)


def cell_id(view,target,family):
    return '__'.join([view,target,family])


def qqq_adapter(base):
    data=base.copy(); root=Path(base['manifest']['root'])
    q=pd.read_csv(root/'etf_adj_close.csv',index_col=0)['QQQ'].to_numpy(float)
    dv=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0)['QQQ'].to_numpy(float)
    data['panel']=copy.copy(base['panel']); data['panel'].tickers=['QQQ']
    data['marks']=q[:,None]; data['adv']=pd.Series(dv).rolling(20).mean().to_numpy()[:,None]
    data['tradable']=np.isfinite(q[:,None]); data['membership']=np.ones((len(q),1),bool)
    data['mask']=np.ones((len(base['origin']),1),bool); data['events']=[]
    return data


def forecast_maps(data,plan,ranker=ensemble_predictions):
    base=data['base']; ids=np.concatenate([fold(base,y)[2] for y in plan['test_years']])
    mask=base['mask'][ids]; raw_forecasts={}; scores={}; distinct=[]
    for view in plan['views']:
        for target in plan['targets']:
            for family in plan['families']:
                name=cell_id(view,target,family)
                seeds=[plan['seeds'][0]] if family=='ridge' else plan['seeds']; by_seed={}
                for seed in seeds:
                    pieces=[]
                    for year in plan['test_years']:
                        path=RUN/view/str(year)/f'{family}_{target}_{seed}_predictions.npz'
                        with np.load(path) as saved:
                            pieces.append(saved['test'].copy())
                    by_seed[seed]=np.concatenate(pieces)
                raw_forecasts[name]=by_seed
                scores[name]=ranker(list(by_seed.values()),mask)
                distinct.append(dict(model=name,seed_runs=len(seeds),distinct_test_forecast_arrays=len({
                    hashlib.sha256(p.tobytes()).hexdigest() for p in by_seed.values()})))
    selections=json.loads((RUN/'SELECTION_LOCK.json').read_text())['choices']
    selectors={name:np.zeros_like(next(iter(scores.values()))) for name in ['validation_top3','validation_best','equal_all']}
    matched={seed:np.zeros_like(next(iter(scores.values()))) for seed in plan['seeds']}
    start=0
    for choice in selections:
        _,_,test=splits(data,choice['year']); stop=start+len(test); section=slice(start,stop)
        selected=[cell_id(c['view'],c['target'],c['family']) for c in choice['top_three']]
        best=choice['single_best']; best=cell_id(best['view'],best['target'],best['family'])
        selectors['validation_top3'][section]=ranker([scores[n][section] for n in selected],mask[section])
        selectors['validation_best'][section]=ranker([scores[best][section]],mask[section])
        selectors['equal_all'][section]=ranker([scores[n][section] for n in scores],mask[section])
        for seed in plan['seeds']:
            available=[raw_forecasts[n][seed if seed in raw_forecasts[n] else plan['seeds'][0]][section] for n in selected]
            matched[seed][section]=ranker(available,mask[section])
        start=stop
    scores.update(selectors)
    references={}; reference_files=[]
    for name,folder,view in [('reference_original_think_mix',ORIGINAL,None),
                             ('reference_rank_daily',RANKING,'daily_control'),
                             ('reference_rank_intraday',RANKING,'intraday_augmented')]:
        available=[]
        for seed in plan['seeds']:
            pieces=[]
            for year in plan['test_years']:
                path=(folder/str(year)/f'think_mix_{seed}_predictions.npz' if view is None else
                      folder/view/str(year)/f'raw_mse_{seed}_predictions.npz')
                reference_files.append(path)
                with np.load(path) as saved:
                    pieces.append(saved['test'].copy())
            available.append(np.concatenate(pieces))
        references[name]=ranker(available,mask)
    return ids,scores,raw_forecasts,matched,references,pd.DataFrame(distinct),reference_files


def evaluate():
    plan=complete()
    if (RUN/'EVALUATED.json').exists():
        raise ValueError('Completed broad evaluation is immutable')
    tests=json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    if (tests['status']!='passed' or tests['report_sha256']!=sha(__file__) or
        tests['test_sha256']!=sha(ROOT/'test_ndx_broad_report.py')):
        raise ValueError('Evaluation test receipt is missing or stale')
    source_lock('EVALUATION_SOURCE_LOCK.json',[__file__,ROOT/'test_ndx_broad_report.py',ROOT/'report_ndx_ranking.py'])
    data=prepare(); base=data['base']; ids,scores,raw,matched,references,distinct,reference_files=forecast_maps(data,plan)
    source_lock('REFERENCE_LOCK.json',reference_files)
    np.savez_compressed(RUN/'scores.npz',ids=ids,**scores,**references,
                        **{'matched_seed_'+str(s):p for s,p in matched.items()})
    distinct.to_csv(RUN/'seed_distinctness.csv',index=False)
    metrics,curves,ledgers,ranking,primary=[],[],[],[],{}
    def record(model,policy,seed,cost,metric,path,ledger=None):
        tags=dict(model=model,policy=policy,seed=str(seed),cost_multiplier=cost)
        metrics.append(dict(**tags,**metric)); curves.append(path.assign(**tags))
        if str(seed)=='ensemble' and cost==1:
            primary[policy,model]=path.net.to_numpy()
            if ledger is not None:
                ledgers.append(ledger.assign(**tags))
    def exposure(policy):
        return np.ones(len(base['origin'])) if policy=='full' else data['trend_exposure']
    def account(name,prediction,seed,costs,source=base):
        for policy in plan['risk_policies']:
            for cost in costs:
                metric,path,ledger=policy_backtest(source,ids,prediction,10,cost,exposure=exposure(policy))
                record(name,policy,seed,cost,metric,path,ledger)
    test_indices=np.concatenate([splits(data,y)[2] for y in plan['test_years']])
    for name,prediction in scores.items():
        account(name,prediction,'ensemble',plan['cost_multipliers'])
        raw_ic=rank_ic(prediction,data['targets']['raw'][test_indices],data['labelmask'][test_indices])
        residual_ic=rank_ic(prediction,data['targets']['residual'][test_indices],data['labelmask'][test_indices])
        for j,i in enumerate(test_indices):
            ranking.append(dict(model=name,signal_date=str(data['dates'][data['origin'][i]].date()),
                                raw_rankic=raw_ic[j],residual_rankic=residual_ic[j],observed_labels=int(data['labelmask'][i].sum())))
        if name in raw and len(raw[name])>1:
            for seed,p in raw[name].items():
                account(name,p,seed,[1])
        print('Evaluated',name,flush=True)
    for seed,prediction in matched.items():
        account('validation_top3',prediction,seed,[1])
    for name,prediction in references.items():
        account(name,prediction,'ensemble',plan['cost_multipliers'])
    momentum=model_data(base,'momentum126'); account('momentum126',base['momentum'][ids],'ensemble',plan['cost_multipliers'],momentum)
    for cost in plan['cost_multipliers']:
        metric,path,ledger=policy_backtest(base,ids,np.zeros_like(base['y'][ids]),10,cost,basket=True)
        record('monthly_basket','full','ensemble',cost,metric,path,ledger)
        for ticker in ['SPY','QQQ']:
            metric,path=etf_backtest(base,ticker,cost); record(ticker,'full','ensemble',cost,metric,path)
        adapter=qqq_adapter(base)
        metric,path,ledger=policy_backtest(adapter,ids,np.ones((len(ids),1)),1,cost,exposure=data['trend_exposure'])
        record('QQQ_timing','trend_half','ensemble',cost,metric,path,ledger)
    metric_frame=pd.DataFrame(metrics); curve_frame=pd.concat(curves,ignore_index=True)
    metric_frame.to_csv(RUN/'metrics.csv',index=False); curve_frame.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'primary_ledgers.csv',index=False)
    rank_frame=pd.DataFrame(ranking); rank_frame.to_csv(RUN/'ranking_periods.csv',index=False)
    rank_frame.groupby('model')[['raw_rankic','residual_rankic']].mean().to_csv(RUN/'ranking_summary.csv')
    annual,monthly,seeds,exposure_rows=[],[],[],[]
    for key,path in curve_frame.groupby(['model','policy','seed','cost_multiplier'],sort=False):
        tags=dict(zip(['model','policy','seed','cost_multiplier'],key)); date=pd.to_datetime(path.date)
        for year,part in path.groupby(date.dt.year):
            annual.append(dict(**tags,year=int(year),**performance(part.net)))
        for month,part in path.groupby(date.dt.to_period('M').astype(str)):
            monthly.append(dict(**tags,month=month,net_return=float(np.prod(1+part.net)-1)))
    spy=metric_frame[(metric_frame.model=='SPY')&(metric_frame.cost_multiplier==1)].annualized_return.iloc[0]
    for (name,policy),part in metric_frame[metric_frame.seed!='ensemble'].groupby(['model','policy']):
        seeds.append(dict(model=name,policy=policy,minimum_cagr=part.annualized_return.min(),median_cagr=part.annualized_return.median(),
                          maximum_cagr=part.annualized_return.max(),seeds_beating_spy=int((part.annualized_return>spy).sum()),seed_runs=len(part)))
    for (policy,name),path in primary.items():
        reference=primary['full','SPY']; qq=primary['full','QQQ']
        exposure_rows.append(dict(model=name,policy=policy,spy_beta=float(np.cov(path,reference)[0,1]/np.var(reference,ddof=1)),
            qqq_beta=float(np.cov(path,qq)[0,1]/np.var(qq,ddof=1)),spy_tracking_error=float(np.std(path-reference,ddof=1)*np.sqrt(252)),
            qqq_tracking_error=float(np.std(path-qq,ddof=1)*np.sqrt(252)),spy_active_mean=float(np.mean(path-reference)*252)))
    for name,rows in [('annual',annual),('monthly',monthly),('seed_summary',seeds),('exposure',exposure_rows)]:
        pd.DataFrame(rows).to_csv(RUN/(name+'.csv'),index=False)
    cells=list(raw)
    contrasts=[]
    for policy in plan['risk_policies']:
        for benchmark in ['SPY','QQQ']:
            contrasts.append((f'{policy}: primary minus {benchmark}',(policy,'validation_top3'),('full',benchmark)))
        for name in cells:
            contrasts.append((f'{policy}: primary minus {name}',(policy,'validation_top3'),(policy,name)))
    intervals=joint_intervals(primary,contrasts,[5,20,60],annualizer=252.)
    intervals=intervals.rename(columns={'left_view':'left_policy','right_view':'right_policy'})
    intervals.to_csv(RUN/'paired_intervals.csv',index=False)
    annual=pd.DataFrame(annual); monthly=pd.DataFrame(monthly)
    screens=[]
    for policy in plan['risk_policies']:
        def pick(frame,model,cost=1):
            return frame[(frame.model==model)&(frame.policy==(policy if model=='validation_top3' else 'full'))&
                         (frame.seed=='ensemble')&(frame.cost_multiplier==cost)]
        m=pick(metric_frame,'validation_top3').iloc[0]; sp=pick(metric_frame,'SPY').iloc[0]
        a=pick(annual,'validation_top3').set_index('year').cumulative_return
        b=pick(annual,'SPY').set_index('year').cumulative_return
        mm=pick(monthly,'validation_top3').set_index('month').net_return
        sm=pick(monthly,'SPY').set_index('month').net_return
        row=dict(policy=policy,base_cagr_above_spy=bool(m.annualized_return>sp.annualized_return),
            double_cost_cagr_above_spy=bool(pick(metric_frame,'validation_top3',2).annualized_return.iloc[0]>
                                          pick(metric_frame,'SPY',2).annualized_return.iloc[0]),
            years_above_spy=int((a>b).sum()),months_above_spy=int((mm>sm).sum()),
            drawdown_within_five_pp=bool(m.maximum_drawdown>=sp.maximum_drawdown-.05))
        row['historical_screen_passed']=bool(row['base_cagr_above_spy'] and row['double_cost_cagr_above_spy'] and
                    row['years_above_spy']>=3 and row['months_above_spy']>=28 and row['drawdown_within_five_pp'])
        selected=intervals[(intervals.block_length==20)&(intervals.left_policy==policy)]
        row['positive_simultaneous_contrasts']=int((selected.simultaneous_low>0).sum())
        row['comparisons']=len(selected); screens.append(row)
    pd.DataFrame(screens).to_csv(RUN/'historical_screen.csv',index=False)
    if len(metrics)!=216 or len(contrasts)!=28:
        raise ValueError('Unexpected broad portfolio/contrast inventory: '+str((len(metrics),len(contrasts))))
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',selected_fold_models=176,candidate_fits=208,
        portfolio_accounts=len(metrics),contrasts=len(contrasts),test_history='exploratory 2022-2025',
        validation_selection_preceded_metrics=json.loads((RUN/'SELECTION_LOCK.json').read_text())['created_before_new_financial_metrics']))


def audit():
    from audit_historical_ndx import independent_metrics
    from audit_historical_ndx_ranking import average_ranks
    from sklearn.ensemble import HistGradientBoostingRegressor
    from training_device import configure_device
    plan=complete()
    if not (RUN/'EVALUATED.json').exists():
        raise ValueError('Broad evaluation must precede audit')
    for name in ['EVALUATION_SOURCE_LOCK.json','REFERENCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:
                raise ValueError('Evaluation/reference bytes changed')
    source_lock('AUDIT_SOURCE_LOCK.json',[__file__,ROOT/'audit_historical_ndx.py',ROOT/'audit_historical_ndx_ranking.py',ROOT/'ndx_policy_engine.py'])
    data=prepare(); base=data['base']; device=configure_device('cuda')
    reproduced,ridge_refits,tree_refits=0,0,0
    independent_choices=[]
    for year in plan['test_years']:
        tr,va,te=splits(data,year); months=data['dates'][data['origin'][va]].to_period('M').astype(str); cells=[]
        for view in plan['views']:
            x,mean,std=normalize(data['views'][view],data['mask'],tr); folder=RUN/view/str(year)
            with np.load(folder/'preprocessing.npz') as saved:
                for name,value in dict(train=tr,validation=va,test=te,mean=mean,std=std).items():
                    np.testing.assert_array_equal(saved[name],value)
            altered=data['views'][view].copy(); altered[max(tr)+1:]=1e10
            _,new_mean,new_std=normalize(altered,data['mask'],tr)
            np.testing.assert_array_equal(mean,new_mean); np.testing.assert_array_equal(std,new_std)
            for target in plan['targets']:
                observed=data['labelmask'][tr]; y=data['targets'][target][tr][observed]
                weights=np.broadcast_to((100/observed.sum(1))[:,None],observed.shape)[observed]; flat=x[tr][observed]
                for family in plan['families']:
                    seeds=[plan['seeds'][0]] if family=='ridge' else plan['seeds']; validation_predictions=[]
                    for seed in seeds:
                        prefix=folder/f'{family}_{target}_{seed}'
                        history=json.loads(prefix.with_name(prefix.name+'_history.json').read_text())
                        if family=='ridge':
                            a=np.column_stack([flat,np.ones(len(flat))])*np.sqrt(weights[:,None]); b=y*np.sqrt(weights)
                            vd=np.concatenate([x[va],np.ones((*x[va].shape[:2],1))],axis=-1)
                            td=np.concatenate([x[te],np.ones((*x[te].shape[:2],1))],axis=-1)
                            with np.load(prefix.with_suffix('.npz')) as model:
                                for number,penalty in enumerate(plan['ridge_penalties']):
                                    p=np.eye(a.shape[-1])*penalty; p[-1,-1]=0.
                                    coefficient=np.linalg.solve(a.T@a+p,a.T@b)
                                    np.testing.assert_allclose(coefficient,model['all_coefficients'][number],atol=1e-9,rtol=1e-9)
                                    ridge_refits+=1
                                chosen=model['coefficients'].copy()
                            vp,tp=vd@chosen,td@chosen
                            candidates=history['candidates']; choice_key='penalty'; chosen_key='selected_penalty'
                        elif family=='boosted_trees':
                            model=joblib.load(prefix.with_suffix('.joblib'))
                            with threadpool_limits(limits=2):
                                vp=tree_prediction(model,x[va].reshape(-1,x.shape[-1]),history['selected_iteration']).reshape(len(va),-1)
                                tp=tree_prediction(model,x[te].reshape(-1,x.shape[-1]),history['selected_iteration']).reshape(len(te),-1)
                                if seed==plan['seeds'][0]:
                                    cfg={k:v for k,v in plan['tree_parameters'].items() if k!='candidate_iterations'}
                                    fresh=HistGradientBoostingRegressor(**cfg,random_state=seed).fit(flat,y,sample_weight=weights)
                                    independently_refitted=tree_prediction(fresh,x[te].reshape(-1,x.shape[-1]),history['selected_iteration']).reshape(len(te),-1)
                                    np.testing.assert_allclose(independently_refitted,tp,atol=1e-10,rtol=1e-10); tree_refits+=1
                            candidates=history['candidates']; choice_key='iteration'; chosen_key='selected_iteration'
                        else:
                            checkpoint=torch.load(prefix.with_suffix('.pt'),map_location='cpu',weights_only=False)
                            model=SharedNetwork(checkpoint['features'],checkpoint['hidden']).to(device)
                            model.load_state_dict(checkpoint['state_dict'])
                            vp,tp=[predict_network(model,x,data['mask'],sub,device,plan['network_parameters']['batch_dates']) for sub in [va,te]]
                            if checkpoint['epoch']!=history['selected_epoch'] or checkpoint['seed']!=seed:
                                raise ValueError('Shared checkpoint metadata mismatch')
                            candidates=history['history']; choice_key='epoch'; chosen_key='selected_epoch'
                        best,selected=-np.inf,None
                        for candidate in candidates:
                            if candidate['validation_stability']>best+1e-7:
                                best,selected=candidate['validation_stability'],candidate[choice_key]
                        if selected!=history[chosen_key]:
                            raise ValueError('Validation-only tuning rule mismatch')
                        np.testing.assert_allclose(stability(vp,data['targets']['residual'][va],data['labelmask'][va],months),best,atol=1e-12,rtol=0)
                        with np.load(prefix.with_name(prefix.name+'_predictions.npz')) as saved:
                            np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0)
                            np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
                        validation_predictions.append(vp); reproduced+=1
                    ensemble=ensemble_predictions(validation_predictions,data['mask'][va])
                    value=stability(ensemble,data['targets']['residual'][va],data['labelmask'][va],months)
                    cells.append(dict(view=view,target=target,family=family,validation_stability=value,protocol_order=len(cells)))
        ordered=[]
        while cells:
            highest=max(c['validation_stability'] for c in cells)
            first=min((c for c in cells if c['validation_stability']>=highest-1e-7),key=lambda c:c['protocol_order'])
            ordered.append(first); cells.remove(first)
        independent_choices.append(dict(year=year,top_three=ordered[:3],single_best=ordered[0],all_cells_ordered=ordered))
        print(f'Reproduced {year}: {reproduced}/176 fold models; {tree_refits}/16 independent tree refits',flush=True)
    locked=json.loads((RUN/'SELECTION_LOCK.json').read_text())
    if not locked['created_before_new_financial_metrics'] or independent_choices!=locked['choices']:
        raise ValueError('Annual model selections do not reproduce')
    def independent_ranker(available,mask):
        ranked=[]
        for p in available:
            value=np.zeros_like(p,dtype=float)
            for j,m in enumerate(mask):
                value[j,m]=average_ranks(p[j,m])/max(m.sum(),1)
            ranked.append(value)
        return np.mean(ranked,axis=0)
    ids,scores,raw,matched,references,_,_=forecast_maps(data,plan,independent_ranker)
    with np.load(RUN/'scores.npz') as saved:
        np.testing.assert_array_equal(ids,saved['ids'])
        for name,prediction in dict(**scores,**references,**{'matched_seed_'+str(s):p for s,p in matched.items()}).items():
            np.testing.assert_allclose(prediction,saved[name],atol=1e-12,rtol=0)
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}).set_index(['model','policy','seed','cost_multiplier'])
    curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str}); accounts,largest=0,0.
    for key,path in curves.groupby(['model','policy','seed','cost_multiplier'],sort=False):
        name,policy,seed,cost=key
        if len(path)!=1003 or not path.date.is_unique or not path.date.is_monotonic_increasing:
            raise ValueError('Broad account calendar mismatch')
        for field,value in independent_metrics(path.net).items():
            np.testing.assert_allclose(value,metrics.loc[key,field],atol=1e-10,rtol=1e-10)
        exposure=np.ones(len(base['origin'])) if policy=='full' else data['trend_exposure']
        source=base
        if name in ['SPY','QQQ']:
            root=Path(base['manifest']['root']); first=base['entry'][ids[0]]
            price=pd.read_csv(root/'etf_adj_close.csv')[name].to_numpy(); volume=pd.read_csv(root/'etf_dollar_volume.csv')[name].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/volume[first-20:first].mean()/.01))
            nav=1e6/(1+rate)/price[first]*price[first:]
            nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/volume[-21:-1].mean()/.01))
        else:
            if name=='QQQ_timing':
                source=qqq_adapter(base); prediction=np.ones((len(ids),1))
            elif name=='monthly_basket':
                prediction=np.zeros_like(base['y'][ids])
            elif name=='momentum126':
                source=model_data(base,'momentum126'); prediction=base['momentum'][ids]
            elif name in references:
                prediction=references[name]
            elif seed!='ensemble':
                prediction=matched[int(seed)] if name=='validation_top3' else raw[name][int(seed)]
            else:
                prediction=scores[name]
            nav=policy_replay(source,ids,prediction,1 if name=='QQQ_timing' else 10,cost,
                              basket=name=='monthly_basket',exposure=exposure)
        actual=1e6*np.cumprod(1+path.net.to_numpy()); largest=max(largest,float(np.abs(actual-nav).max()))
        np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10)
        accounts+=1
        if accounts%30==0:
            print(f'Independently replayed {accounts}/216 broad accounts',flush=True)
    ledger=pd.read_csv(RUN/'primary_ledgers.csv')
    np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    if ledger.cash.min()<-1e-5 or accounts!=216 or reproduced!=176 or ridge_refits!=48 or tree_refits!=16:
        raise ValueError('Broad audit inventory or cash check failed')
    write(RUN/'AUDIT.json',dict(status='passed',fold_models_reproduced=reproduced,ridge_coefficients_refitted=ridge_refits,
        tree_models_independently_refitted=tree_refits,annual_model_selections_reproduced=4,
        independent_accounts=accounts,largest_nav_difference_dollars=largest,
        untouched_confirmation='not performed; historical development only'))


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def pct(value):
    return f'{value*100:.1f}%'


def pretty(name):
    labels=dict(validation_top3='Validation top-three blend (primary)',validation_best='Validation best single cell',
        equal_all='Equal blend of all twelve cells',reference_original_think_mix='Original five-feature THINK+mixer',
        reference_rank_daily='Daily ranking-stopped THINK+mixer',reference_rank_intraday='Intraday ranking-stopped THINK+mixer',
        monthly_basket='Monthly basket',momentum126='Momentum 126',QQQ_timing='QQQ with the same trend rule')
    return labels.get(name,name.replace('__',' / ').replace('boosted_trees','trees').replace('shared_network','shared net'))


def publish():
    complete(); audit_record=json.loads((RUN/'AUDIT.json').read_text())
    if audit_record['status']!='passed':
        raise ValueError('Broad independent audit has not passed')
    for name in ['EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:
                raise ValueError('Audited source changed')
    DOCS.mkdir(parents=True,exist_ok=True); figures=DOCS/'figures'; figures.mkdir(exist_ok=True)
    plan=json.loads((RUN/'PROTOCOL.json').read_text()); tests=json.loads((RUN/'UNIT_TESTS.json').read_text())
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}); curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    main=metrics[(metrics.seed=='ensemble')&(metrics.cost_multiplier==1)].set_index(['model','policy'])
    annual=pd.read_csv(RUN/'annual.csv',dtype={'seed':str}); monthly=pd.read_csv(RUN/'monthly.csv',dtype={'seed':str})
    seeds=pd.read_csv(RUN/'seed_summary.csv').set_index(['model','policy']); distinct=pd.read_csv(RUN/'seed_distinctness.csv')
    rank=pd.read_csv(RUN/'ranking_summary.csv').set_index('model'); screen=pd.read_csv(RUN/'historical_screen.csv').set_index('policy')
    intervals=pd.read_csv(RUN/'paired_intervals.csv'); primary_ci=intervals[intervals.block_length==20]
    exposure=pd.read_csv(RUN/'exposure.csv').set_index(['model','policy'])
    choices=json.loads((RUN/'SELECTION_LOCK.json').read_text()); features=json.loads((RUN/'FEATURES.json').read_text())
    cells=distinct.model.tolist(); evaluated_models=cells+['validation_top3','validation_best','equal_all']
    upper=max(float(np.cumprod(1+p.net).max()) for _,p in curves[(curves.seed=='ensemble')&(curves.cost_multiplier==1)].groupby(['model','policy']))*1.05
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for policy in plan['risk_policies']:
        fig,ax=plt.subplots(figsize=(13,7))
        for name,color,style in [('validation_top3','black','-'),('validation_best','#777777','-.'),('equal_all','#996633','-'),
            ('reference_original_think_mix','#9467bd','-'),('reference_rank_daily','#1f77b4',':'),('reference_rank_intraday','#ff7f0e',':'),
            ('SPY','#cc0000','--'),('QQQ','#333399','--')]+([('QQQ_timing','#2ca02c','--')] if policy=='trend_half' else []):
            actual_policy='full' if name in ['SPY','QQQ'] else policy
            p=curves[(curves.model==name)&(curves.policy==actual_policy)&(curves.seed=='ensemble')&(curves.cost_multiplier==1)]
            m=main.loc[name,actual_policy]
            ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),color=color,linestyle=style,
                    linewidth=2.5 if name=='validation_top3' else 1.6,label=f'{pretty(name)}: {pct(m.annualized_return)} CAGR')
        ax.set(title=f'Exploratory 2022–2025: {policy} exposure; fixed top 10',ylabel='Net wealth per initial $1',ylim=(0,upper))
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper left'); fig.tight_layout(); fig.savefig(figures/(policy+'_wealth.png'),dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,8)); y=np.arange(len(cells))
    for policy,color,offset in [('full','#1f77b4',-.18),('trend_half','#ff7f0e',.18)]:
        values=[main.loc[name,policy]['annualized_return']*100 for name in cells]
        ax.barh(y+offset,values,height=.34,color=color,label=policy)
    ax.axvline(main.loc['SPY','full']['annualized_return']*100,color='black',linestyle='--',label='SPY')
    ax.set(yticks=y,yticklabels=[pretty(n) for n in cells],xlabel='Net CAGR (%)',title='Every base-model cell and both fixed risk policies')
    ax.legend(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(figures/'all_cells.png',dpi=160); plt.close(fig)
    subset=primary_ci[primary_ci.right_model.isin(['SPY','QQQ'])]
    fig,ax=plt.subplots(figsize=(12,5))
    for j,row in enumerate(subset.itertuples(index=False)):
        ax.plot(np.array([row.simultaneous_low,row.simultaneous_high])*100,[j,j],color='#cccccc',linewidth=5)
        ax.plot(np.array([row.paired_low,row.paired_high])*100,[j,j],color='#1f77b4',linewidth=2)
        ax.scatter(row.estimate*100,j,color='#1f77b4')
    ax.set(yticks=np.arange(len(subset)),yticklabels=subset.contrast,xlabel='Annual arithmetic mean active return (percentage points)',
           title='Primary blend: paired 95% intervals and 28-contrast simultaneous diagnostics')
    ax.axvline(0,color='black'); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(figures/'primary_intervals.png',dpi=160); plt.close(fig)
    performance_rows=[]
    for name in evaluated_models+['reference_original_think_mix','reference_rank_daily','reference_rank_intraday','momentum126','monthly_basket','SPY','QQQ','QQQ_timing']:
        values=[]
        for policy in plan['risk_policies']:
            if (name,policy) in main.index:
                m=main.loc[name,policy]; values.extend([pct(m.annualized_return),f'{m.sharpe:.3f}',pct(m.maximum_drawdown)])
            else:
                values.extend(['—']*3)
        performance_rows.append([pretty(name)]+values)
    screening_rows,seed_rows,cost_rows,annual_rows,monthly_rows=[],[],[],[],[]
    for policy in plan['risk_policies']:
        s=screen.loc[policy]; z=seeds.loc['validation_top3',policy]
        screening_rows.append([policy,str(bool(s.historical_screen_passed)),f'{int(s.years_above_spy)}/4',f'{int(s.months_above_spy)}/48',
                               str(bool(s.drawdown_within_five_pp)),f'{int(z.seeds_beating_spy)}/{int(z.seed_runs)}',
                               f'{int(s.positive_simultaneous_contrasts)}/{int(s.comparisons)}'])
    for name in evaluated_models:
        for policy in plan['risk_policies']:
            m=main.loc[name,policy]
            costs=metrics[(metrics.model==name)&(metrics.policy==policy)&(metrics.seed=='ensemble')].set_index('cost_multiplier')
            cost_rows.append([pretty(name),policy]+[pct(costs.loc[c,'annualized_return']) for c in [1,2,4]]+[f'{m.turnover_annual:.1f}×'])
            if (name,policy) in seeds.index:
                z=seeds.loc[name,policy]
                seed_rows.append([pretty(name),policy,pct(z.minimum_cagr),pct(z.median_cagr),pct(z.maximum_cagr),
                                  f'{int(z.seeds_beating_spy)}/{int(z.seed_runs)}'])
    for name in ['validation_top3','validation_best','equal_all','reference_original_think_mix','QQQ_timing','SPY','QQQ']:
        for policy in plan['risk_policies']:
            if (name,policy) not in main.index:
                continue
            a=annual[(annual.model==name)&(annual.policy==policy)&(annual.seed=='ensemble')&(annual.cost_multiplier==1)].set_index('year')
            annual_rows.append([pretty(name),policy]+[pct(a.loc[y,'cumulative_return']) for y in plan['test_years']])
            p=monthly[(monthly.model==name)&(monthly.policy==policy)&(monthly.seed=='ensemble')&(monthly.cost_multiplier==1)].set_index('month').net_return
            sp=monthly[(monthly.model=='SPY')&(monthly.seed=='ensemble')&(monthly.cost_multiplier==1)].set_index('month').net_return
            monthly_rows.append([pretty(name),policy,f'{int((p>sp).sum())}/48'])
    choice_rows=[]
    for c in choices['choices']:
        choice_rows.append([c['year'],'; '.join(pretty(cell_id(z['view'],z['target'],z['family'])) for z in c['top_three']),
                            '; '.join(f'{z["validation_stability"]:+.4f}' for z in c['top_three'])])
    ci_rows=[[r.contrast,f'{r.estimate*100:+.2f} pp',f'[{r.paired_low*100:+.2f}, {r.paired_high*100:+.2f}] pp',
              f'[{r.simultaneous_low*100:+.2f}, {r.simultaneous_high*100:+.2f}] pp'] for r in primary_ci.itertuples(index=False)]
    full=main.loc['validation_top3','full']; risk=main.loc['validation_top3','trend_half']; spy=main.loc['SPY','full']
    best=(metrics[(metrics.seed=='ensemble')&(metrics.cost_multiplier==1)&metrics.model.isin(evaluated_models)]
          .sort_values('annualized_return',ascending=False).iloc[0])
    lines=[
        '# NASDAQ-100: broader feature, model and risk-policy research',
        '',
        '**Complete: 208 candidate fits, 176 selected fold models and 216 continuous portfolio accounts; independent audit passed.**',
        '',
        f'The validation-selected top-three blend returned **{pct(full.annualized_return)} CAGR** at full exposure and '
        f'**{pct(risk.annualized_return)} CAGR** under the fixed trend rule. SPY returned '
        f'**{pct(spy.annualized_return)} CAGR** over the same 2022–2025 account dates.',
        '',
        f'The highest observed development CAGR among the twelve base cells and three selector policies was '
        f'{pct(best.annualized_return)} for {pretty(best.model)} / {best.policy}. This is a descriptive, '
        'after-the-fact observation. The primary strategy remains the validation-selected top-three rule.',
        '',
        '**All 2022–2025 results remain exploratory. These dates were examined in earlier research. '
        'Passing a historical screen or finding a higher curve does not establish reliable future outperformance.**',
        '',
        '## Prespecified reliability screen',
        '',
        'The primary development screen requires CAGR above SPY at base and double costs, at least three '
        'of four years and 28 of 48 months above SPY, and maximum drawdown no more than five percentage '
        'points worse than SPY. Matched seed blends and simultaneous comparisons against every base '
        'model are additional diagnostics. The screen is not a significance test.',
        '',
        md_table(['Primary risk policy','Historical screen passed','Years above SPY','Months above SPY','Drawdown limit passed',
                  'Matched seeds above SPY','Positive simultaneous contrasts'],screening_rows),
        '',
        '## Features, models and causal selection',
        '',
        f'The inputs contain {features["features"]} channels. Daily features include 1–252-session trends, '
        'QQQ-relative trends, moving-average distances, realized and downside volatility, drawdown, '
        'relative dollar volume, shrunk lagged beta, market context and cross-sectional trend ranks. '
        'The intraday view adds current, five-session and twenty-session summaries of the reviewed '
        'Alpaca bars. Daily controls pad those channels. Missing values are imputed using training '
        'means; no historical member is removed for missing long-history or intraday features.',
        '',
        'The finite family crosses two input views, raw versus QQQ-beta-residual targets and three '
        'model types: ridge, shallow histogram gradient boosting and a shared masked neural network. '
        'The network pools eligible-stock context and uses shared parameters, so it has no ticker-specific '
        'output rows and supports a changing security count. This addresses one architectural cold-start '
        'problem; missing history and universe quality still matter. The tree and network descriptions '
        'are local implementations, not reproductions of an author’s investment results. '
        '[Tree implementation reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); '
        '[shared-set architecture background](https://arxiv.org/abs/1703.06114).',
        '',
        'Training uses daily origins with overlapping five-session labels: 375/627/878/1,000 training '
        'origins in the four folds and 245–248 prior-year validation origins. All label maturities '
        'are purged across boundaries. Daily origin density increases training examples, not independent '
        'market years. Test accounts retain the original 201 forecast origins, of which 200 have '
        'complete five-session labels. Normalization uses only eligible training observations.',
        '',
        'Ridge chooses among three fixed penalties, trees among 50/100/200 stages of a fixed shallow '
        '200-stage fit, and the network uses validation stopping up to 30 epochs. The common criterion '
        'is median monthly validation residual RankIC minus one quarter of its monthly standard '
        'deviation. Tree automatic/random validation splitting is disabled. Family selection reads '
        'saved validation predictions only, and all four annual choices were locked before any new '
        'financial test metric. Every attempted cell is disclosed.',
        '',
        md_table(['Test year','Top three selected using prior-year validation only','Validation stability scores'],choice_rows),
        '',
        '## Net results for every cell and comparison',
        '',
        md_table(['Model / selector','Full CAGR','Full Sharpe','Full drawdown','Trend-rule CAGR','Trend-rule Sharpe','Trend-rule drawdown'],performance_rows),
        '',
        'Full exposure targets 100% stocks. The fixed trend rule targets 100% when signal-close QQQ '
        'is at or above its trailing 200-session mean and 50% plus cash otherwise. Both use equal top '
        'ten targets, frozen signal-close share orders, next-close execution, 1% ADV caps, modeled '
        'fees, continuous cash and reviewed economic-price acquisition settlement. Cash earns no '
        'interest. Exposure applies to every five-session model and the separate QQQ timing control. '
        'The monthly basket is a full-exposure monthly reference. No risk threshold or K was selected '
        'from test outcomes. Accounts start with $1 million; Sharpe retains the fixed 3% hurdle.',
        '',
        '![Full exposure](figures/full_wealth.png)',
        '',
        '![Fixed trend rule](figures/trend_half_wealth.png)',
        '',
        '![Every base-model cell](figures/all_cells.png)',
        '',
        '## Consistency, seeds and costs',
        '',
        md_table(['Model / selector','Policy','2022','2023','2024','2025'],annual_rows),
        '',
        md_table(['Model / selector','Policy','Months above SPY'],monthly_rows),
        '',
        md_table(['Model / selector','Policy','Minimum seed CAGR','Median seed CAGR','Maximum seed CAGR','Seeds above SPY'],seed_rows),
        '',
        'Primary matched-seed blends hold the annual validation-selected cells fixed and use the '
        'matching seed from each selected stochastic model, or the single deterministic ridge. '
        'They diagnose initialization sensitivity and do not refit the selection policy independently. '
        'Different tree seed labels can produce identical predictions. Distinct forecast counts are '
        'disclosed below; even different forecasts share one historical market path.',
        '',
        md_table(['Base cell','Seed runs','Distinct test-forecast arrays'],[[pretty(r.model),r.seed_runs,r.distinct_test_forecast_arrays] for r in distinct.itertuples(index=False)]),
        '',
        md_table(['Model / selector','Policy','1× cost CAGR','2× cost CAGR','4× cost CAGR','Base annual traded-notional/NAV'],cost_rows),
        '',
        '## Paired uncertainty and competing models',
        '',
        '![Primary benchmark intervals](figures/primary_intervals.png)',
        '',
        md_table(['Primary contrast','Mean active return/year','Paired 95% interval','28-contrast simultaneous diagnostic'],ci_rows),
        '',
        'The primary blend is compared with SPY, QQQ and every one of the twelve base cells under '
        'the matching risk policy. ETF benchmark rows remain buy-and-hold. One joint max-statistic '
        'family covers all 28 contrasts. Each interval uses 10,000 whole-day paired circular-block '
        'resamples; the primary block is 20 sessions and 5/60-session sensitivities are retained. '
        'Arithmetic active means differ from CAGR gaps. The intervals condition on fitted paths and '
        'do not account for the full historical research search, training uncertainty or regime change. '
        'Prior published references are descriptive controls outside this comparison family.',
        '',
        '## Verification and remaining work',
        '',
        f'The broad training suite passed {tests["tests"]} tests, including actual future-input '
        'perturbations, permutation equivariance, inactive-NaN masking, normalization and known '
        'risk-account fixtures. Additional evaluation tests verify that changing test predictions '
        'cannot alter annual validation selections. Independent checks reproduced all 176 fold '
        'model exports, refitted all 48 ridge candidate coefficients and 16 tree models, reproduced '
        f'all four annual selections and replayed all 216 accounts. Maximum NAV discrepancy was '
        f'${audit_record["largest_nav_difference_dollars"]:.8f}.',
        '',
        'The original data, models and published studies remain unchanged. Corporate-action and '
        'total-return approximations, missing historical labels, revised vendor histories, market '
        'dependence and repeated examination limit conclusions. The reserved confirmation dates '
        'from October 21, 2026 onward are not evaluated here. Updated point-in-time membership, '
        'corporate actions, input coverage, the selected model/refit policy and execution must be '
        'reviewed and locked before any future run. A model that passes historical screens can '
        'still fail on those unseen dates. No trading orders or recurring jobs were created.',
        '',
        '## Complete artifacts',
        '',
        '[Protocol](PROTOCOL.json) · [Audit](AUDIT.json) · [All metrics](metrics.csv) · [Every daily path](curves.csv) · '
        '[Historical screen](historical_screen.csv) · [Annual validation selections](SELECTION_LOCK.json) · '
        '[Annual returns](annual.csv) · [Monthly returns](monthly.csv) · [Seed summaries](seed_summary.csv) · '
        '[Distinct forecasts](seed_distinctness.csv) · [All confidence intervals](paired_intervals.csv) · '
        '[Ranking summary](ranking_summary.csv) · [Exposure](exposure.csv) · [Feature inventory](FEATURES.json) · '
        '[Fitting environment](ENVIRONMENT.json)',
        '',
        'Checkpoints, tree objects, coefficients, preprocessing, validation/test exports, trial histories '
        'and integrity locks are under `runs/ndx_broad_20261020`. Resume with '
        '`.\\.venv\\Scripts\\python.exe run_ndx_broad.py --stage all`.',
        ''
    ]
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    names=['PROTOCOL.json','AUDIT.json','FREEZE.json','ENVIRONMENT.json','UNIT_TESTS.json','EVALUATION_UNIT_TESTS.json',
        'metrics.csv','curves.csv','annual.csv','monthly.csv','seed_summary.csv','seed_distinctness.csv','paired_intervals.csv',
        'historical_screen.csv','SELECTION_LOCK.json','ranking_summary.csv','exposure.csv','FEATURES.json',
        'EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','REFERENCE_LOCK.json','EVALUATED.json']
    for name in names:
        shutil.copyfile(RUN/name,DOCS/name)
    for name in ['report_ndx_broad.py','test_ndx_broad_report.py','report_ndx_ranking.py','requirements-research-20261020.txt']:
        shutil.copyfile(ROOT/name,RUN/'source_snapshot'/name)
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_research',selected_fold_models=176,
        candidate_fits=208,independent_accounts=216,primary_full_cagr=float(full.annualized_return),
        primary_trend_cagr=float(risk.annualized_return),spy_cagr=float(spy.annualized_return),
        historical_screens=screen.reset_index().to_dict('records'),future_reliability='not established by historical development',
        descriptive_best=dict(model=best.model,policy=best.policy,cagr=float(best.annualized_return)),
        report=str(DOCS/'REPORT.md'),artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    print(json.dumps({k:v for k,v in json.loads((RUN/'FINAL_PUBLICATION.json').read_text()).items() if k!='artifact_hashes'},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['evaluate','audit','publish'])
    args=parser.parse_args(); globals()[args.stage]()


if __name__=='__main__':
    main()
