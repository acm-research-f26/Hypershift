"""Identical portfolios, uncertainty and complementarity for reference transfers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_historical_ndx import independent_metrics
from compare_ndx_strategies_20261020 import bootstrap_sharpe
from ndx_extended_hours import ExtendedModel,predict
from ndx_loss_optimizer import PARENT,prepared_fold
from ndx_policy_engine import policy_replay
from ndx_ranking_objectives import rank_ic
from ndx_reference_models import ROOT,make_reference
from ndx_reference_training import RUN,KINDS,prepare,predictions,inputs
from pit_research_engine import load_data,fold,continuous_backtest,etf_backtest
from research_engine import performance
from report_ndx_ranking import joint_intervals
from run_ndx_alpaca_training import sha,write
from run_ndx_extended_hours_top5 import cagr_samples
from run_ndx_reference_networks import complete
from run_robust_research import ensemble_predictions
from training_device import configure_device,tensor_on

PURE=['baseline',*KINDS]
METHODS=PURE+[k+'_blend' for k in KINDS]
GROUPS=['ten_seed','original_five','new_five']
def account(method,group,k=10,seed='ensemble',cost=1):return f'{method}/{group}/{seed}/{k}/{cost}'
def read(name):return pd.read_csv(RUN/name,dtype={'seed':str})


def group_seeds(plan):return {'ten_seed':plan['seeds'],'original_five':plan['seeds'][:5],'new_five':plan['seeds'][5:]}


def family(group):
    baseline=account('baseline',group);result=[]
    for method in METHODS:
        for ticker in ['SPY','QQQ']:result.append((method+' minus '+ticker,('account',account(method,group)),('account',account(ticker,'benchmark',0))))
        if method!='baseline':result.append((method+' minus baseline',('account',account(method,group)),('account',baseline)))
    for kind in KINDS:result.append((kind+' blend minus standalone',('account',account(kind+'_blend',group)),('account',account(kind,group))))
    assert len(result)==23
    return result


def blend_scores(reference,baseline):return .5*reference+.5*baseline


def fit_folder(kind,year):return (PARENT/'after_hours_premarket' if kind=='baseline' else RUN/kind)/str(year)


def forecasts():
    plan=complete();data=load_data();ids=np.concatenate([fold(data,y)[2] for y in plan['years']]);raw={};scores={};validation={}
    for kind in PURE:
        raw[kind]={};validation[kind]={}
        for seed in plan['seeds']:
            tests=[];vals=[]
            for year in plan['years']:
                with np.load(fit_folder(kind,year)/f'{seed}_predictions.npz') as saved:
                    tests.append(saved['test'].copy());vals.append(saved['validation'].copy())
            raw[kind][seed]=np.concatenate(tests);validation[kind][seed]=np.concatenate(vals)
        for group,seeds in group_seeds(plan).items():scores[kind,group]=ensemble_predictions([raw[kind][s] for s in seeds],data['mask'][ids])
    for kind in KINDS:
        for group in GROUPS:scores[kind+'_blend',group]=blend_scores(scores[kind,group],scores['baseline',group])
    return plan,data,ids,raw,scores,validation


def complementarity(plan,data,ids,raw,scores,validation,ps):
    result=[]
    validation_ids=np.concatenate([fold(data,y)[1] for y in plan['years']])
    for kind in KINDS:
        for group,seeds in group_seeds(plan).items():
            for split,sub,forecasts in [('validation',validation_ids,validation),('development_test',ids,raw)]:
                mask=data['mask'][sub];observed=data['labelmask'][sub]
                a=ensemble_predictions([forecasts[kind][s] for s in seeds],mask)
                b=ensemble_predictions([forecasts['baseline'][s] for s in seeds],mask)
                ea=(np.mean([forecasts[kind][s] for s in seeds],0)-data['y'][sub]*100)[observed]
                eb=(np.mean([forecasts['baseline'][s] for s in seeds],0)-data['y'][sub]*100)[observed]
                correlations=[];overlap=[]
                for ar,br,m in zip(a,b,mask):
                    correlations.append(float(np.corrcoef(ar[m],br[m])[0,1]))
                    ai=np.argsort(-np.where(m,ar,-np.inf),kind='stable')[:10]
                    bi=np.argsort(-np.where(m,br,-np.inf),kind='stable')[:10]
                    overlap.append(len(set(ai)&set(bi))/10)
                row=dict(kind=kind,group=group,split=split,mean_cross_section_rank_correlation=float(np.mean(correlations)),
                         mean_top_ten_overlap=float(np.mean(overlap)),raw_error_correlation=float(np.corrcoef(ea,eb)[0,1]))
                if split=='development_test':
                    ra=ps[account(kind,group)];rb=ps[account('baseline',group)];qqq=ps[account('QQQ','benchmark',0)]
                    row.update(net_return_correlation=float(np.corrcoef(ra,rb)[0,1]),qqq_active_return_correlation=float(np.corrcoef(ra-qqq,rb-qqq)[0,1]))
                result.append(row)
    return pd.DataFrame(result)


def freeze_evaluation():
    import unittest
    tests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName('test_ndx_reference_evaluation'))
    if not tests.wasSuccessful():raise ValueError('Reference evaluation tests failed')
    files=[Path(__file__),ROOT/'test_ndx_reference_evaluation.py',ROOT/'ndx_policy_engine.py',ROOT/'audit_historical_ndx.py',
           ROOT/'report_ndx_ranking.py',ROOT/'compare_ndx_strategies_20261020.py',ROOT/'run_ndx_extended_hours_top5.py']
    lock={str(p.resolve()):sha(p) for p in files};target=RUN/'EVALUATION_SOURCE_LOCK.json'
    if target.exists() and json.loads(target.read_text())!=lock:raise ValueError('Frozen reference evaluation changed')
    write(target,lock)


def evaluate():
    if (RUN/'EVALUATED.json').exists():raise ValueError('Reference evaluation already complete')
    freeze_evaluation();plan,data,ids,raw,scores,validation=forecasts();metrics=[];curves=[];ledgers=[];ranking=[];quality=[]
    def record(method,group,k,seed,cost,m,p,l=None):
        tags=dict(id=account(method,group,k,str(seed),cost),method=method,group=group,k=k,seed=str(seed),cost_multiplier=cost)
        metrics.append(dict(**tags,**m));curves.append(p.assign(**tags))
        if l is not None:ledgers.append(l.assign(**tags))
    for method in METHODS:
        for group in GROUPS:
            score=scores[method,group]
            ranking.append(dict(method=method,group=group,rankic=float(np.nanmean(rank_ic(score,data['y'][ids],data['labelmask'][ids])))))
            for k in [5,10]:
                for cost in [1,2,4]:
                    m,p,l=continuous_backtest(data,ids,score,k,cost);record(method,group,k,'ensemble',cost,m,p,l)
        if method in PURE:
            for seed,score in raw[method].items():
                m,p,l=continuous_backtest(data,ids,score,10,1);record(method,'individual',10,seed,1,m,p,l)
                quality.append(dict(method=method,seed=seed,mse_pp2=float(((score-data['y'][ids]*100)**2)[data['labelmask'][ids]].mean()),
                    rankic=float(np.nanmean(rank_ic(score,data['y'][ids],data['labelmask'][ids])))))
        print('Reference portfolios computed for',method,flush=True)
    for ticker in ['SPY','QQQ']:
        for cost in [1,2,4]:
            m,p=etf_backtest(data,ticker,cost);record(ticker,'benchmark',0,'ensemble',cost,m,p)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True);assert len(mf)==172 and mf.id.is_unique
    mf.to_csv(RUN/'metrics.csv',index=False);cf.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'ledgers.csv',index=False)
    pd.DataFrame(ranking).to_csv(RUN/'ranking.csv',index=False);pd.DataFrame(quality).to_csv(RUN/'prediction_quality.csv',index=False)
    annual=[];monthly=[]
    for key,p in cf.groupby('id',sort=False):
        dates=pd.to_datetime(p.date)
        for year,q in p.groupby(dates.dt.year):annual.append(dict(id=key,year=int(year),**performance(q.net)))
        for month,q in p.groupby(dates.dt.to_period('M').astype(str)):monthly.append(dict(id=key,month=month,net_return=float(np.prod(1+q.net)-1)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False);pd.DataFrame(monthly).to_csv(RUN/'monthly.csv',index=False)
    order=mf.id.tolist();ps={key:p.net.to_numpy() for key,p in cf.groupby('id',sort=False)}
    matrix=np.column_stack([ps[key] for key in order]);intervals=[];differences=[];active=[]
    for block in [5,20,60]:
        sr=bootstrap_sharpe(matrix,block=block);growth=cagr_samples(matrix,block)
        low,high=np.quantile(sr,[.025,.975],axis=0)
        for j,key in enumerate(order):intervals.append(dict(id=key,block_sessions=block,sr_low=low[j],sr_high=high[j]))
        for group in GROUPS:
            pairs=family(group)
            for measure,samples,point_values in [('sharpe_difference',sr,mf.sharpe.to_numpy()),('cagr_difference',growth,mf.annualized_return.to_numpy())]:
                values=np.column_stack([samples[:,order.index(a[1])]-samples[:,order.index(b[1])] for _,a,b in pairs])
                point=np.array([point_values[order.index(a[1])]-point_values[order.index(b[1])] for _,a,b in pairs])
                error=values.std(0,ddof=1);critical=float(np.quantile(np.max(np.abs((values-point)/np.maximum(error,1e-12)),axis=1),.95))
                for j,(name,a,b) in enumerate(pairs):differences.append(dict(group=group,contrast=name,measure=measure,left_id=a[1],right_id=b[1],block_sessions=block,
                    estimate=point[j],paired_low=np.quantile(values[:,j],.025),paired_high=np.quantile(values[:,j],.975),
                    simultaneous_low=point[j]-critical*error[j],simultaneous_high=point[j]+critical*error[j],family_comparisons=23))
    for group in GROUPS:
        frame=joint_intervals({('account',k):v for k,v in ps.items()},family(group),[5,20,60],annualizer=252.);frame['group']=group;active.append(frame)
    pd.DataFrame(intervals).to_csv(RUN/'sharpe_intervals.csv',index=False);pd.DataFrame(differences).to_csv(RUN/'paired_intervals.csv',index=False)
    pd.concat(active,ignore_index=True).to_csv(RUN/'active_mean_intervals.csv',index=False)
    complementarity(plan,data,ids,raw,scores,validation,ps).to_csv(RUN/'complementarity.csv',index=False)
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',accounts=172,primary_group='ten_seed'))


def audit():
    plan,data,ids,raw,scores,validation=forecasts()
    for path,h in json.loads((RUN/'EVALUATION_SOURCE_LOCK.json').read_text()).items():
        if sha(path)!=h:raise ValueError('Reference evaluation source changed')
    device=configure_device('cuda');reproduced=0;stage_checks=0
    for year in plan['years']:
        p=prepare(year)
        for kind in KINDS:
            for seed in plan['seeds']:
                folder=fit_folder(kind,year);saved=torch.load(folder/f'{seed}.pt',map_location='cpu',weights_only=False)
                model=make_reference(kind,len(data['panel'].tickers),p['snapshot']).to(device)
                model.load_state_dict(saved['state_dict'])
                if kind=='prism':model.freeze_representation()
                actual=[predictions(model,p,p[sub],kind,device) for sub in ['validation','test']]
                with np.load(folder/f'{seed}_predictions.npz') as old:
                    for name,values in zip(['validation','test'],actual):np.testing.assert_allclose(values,old[name],atol=1e-5,rtol=1e-6)
                history=json.loads((folder/f'{seed}_history.json').read_text())
                for stage,rows in history.items():
                    best=np.inf;epoch=0
                    for row in rows:
                        if row['validation_loss']<best-1e-7:best=row['validation_loss'];epoch=row['epoch']
                    assert epoch==saved['epochs'][stage];stage_checks+=1
                    if stage=='prediction':
                        mse=float(((actual[0]-data['y'][p['validation']]*100)**2)[data['labelmask'][p['validation']]].mean())
                        np.testing.assert_allclose(mse,best,atol=1e-5,rtol=1e-6)
                if kind=='prism':
                    model.load_state_dict(saved['representation_state']);model.eval();loss=[]
                    with torch.no_grad():
                        for i in p['prevalidation']:
                            x,prior,eligible=inputs(p,i,kind,device)
                            loss.append(float(model.pretrain_loss(x,prior,tensor_on(p['future'][i],device),eligible)))
                    selected=saved['epochs']['representation'];expected=history['representation'][selected-1]['validation_loss']
                    np.testing.assert_allclose(np.mean(loss),expected,atol=1e-5,rtol=1e-6)
                reproduced+=1;del model
            print('Audited saved reference checkpoints',kind,year,flush=True)
    mf=read('metrics.csv').set_index('id');cf=read('curves.csv');parent=pd.read_csv(ROOT/'docs/ndx_extended_hours_top5_20261024/curves.csv',dtype={'seed':str})
    largest=0.;refs=0;accounts=0
    for key,p in cf.groupby('id',sort=False):
        row=mf.loc[key];kind=row.method;group=row.group;k=int(row.k);cost=int(row.cost_multiplier)
        assert len(p)==1003 and p.date.is_unique and p.date.is_monotonic_increasing
        for field,value in independent_metrics(p.net).items():np.testing.assert_allclose(value,row[field],atol=1e-10,rtol=1e-10)
        if kind in ['SPY','QQQ']:
            root=Path(data['manifest']['root']);first=data['entry'][ids[0]];q=pd.read_csv(root/'etf_adj_close.csv')[kind].to_numpy();v=pd.read_csv(root/'etf_dollar_volume.csv')[kind].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/v[first-20:first].mean()/.01));nav=1e6/(1+rate)/q[first]*q[first:]
            nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/v[-21:-1].mean()/.01))
        else:nav=policy_replay(data,ids,raw[kind][int(row.seed)] if group=='individual' else scores[kind,group],k,cost)
        actual=1e6*np.cumprod(1+p.net.to_numpy());largest=max(largest,float(np.abs(nav-actual).max()));np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10);accounts+=1
        match=None
        if kind in ['SPY','QQQ']:match=parent[parent.id==f'{kind}/benchmark/ensemble/0/{cost}']
        elif kind=='baseline':match=parent[parent.id==f'after_hours_premarket/{group}/{row.seed}/{k}/{cost}']
        if match is not None:
            np.testing.assert_array_equal(match.date,p.date);np.testing.assert_allclose(match.net,p.net,atol=1e-14,rtol=0);refs+=1
    ledger=read('ledgers.csv');np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    assert accounts==172 and refs==34 and ledger.cash.min()>-1e-5
    proof=dict(status='passed',reference_checkpoint_exports_reproduced=reproduced,selected_stages_checked=stage_checks,
               independent_portfolio_replays=172,prior_accounts_reproduced=refs,largest_nav_difference_dollars=largest)
    write(RUN/'AUDIT.json',proof)
    table=read('metrics.csv');table=table[(table.seed=='ensemble')&(table.k==10)&(table.cost_multiplier==1)]
    table=table.merge(read('ranking.csv'),on=['method','group']).merge(read('sharpe_intervals.csv').query('block_sessions==20'),on='id')
    full=read('metrics.csv');baseline=full[(full.method=='baseline')&(full.group=='individual')].set_index('seed').annualized_return;seedrows=[]
    for kind in PURE:
        values=full[(full.method==kind)&(full.group=='individual')].set_index('seed').annualized_return
        seedrows.append(dict(method=kind,seeds_above_paired_baseline=int((values>baseline).sum()),seeds_above_spy=int((values>.1085244627962889).sum()),median_seed_cagr=float(values.median())))
    write(RUN/'RESULTS.json',dict(status='completed_audited_exploratory',primary_group='ten_seed',top_ten_base_cost=json.loads(table.to_json(orient='records')),
          seed_summary=seedrows,inference='Previously inspected development history; author-network transfers with declared input/training adaptations; no untouched confirmation or deployment'))
    write(RUN/'PROGRESS.json',dict(stage='completed_audited',accounts=172));print('Reference comparison complete and audited',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','audit'])
    globals()[parser.parse_args().stage]()
