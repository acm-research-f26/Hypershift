"""Evaluate and independently audit the frozen training comparison; no documents."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from audit_historical_ndx import independent_metrics
from compare_ndx_strategies_20261020 import bootstrap_sharpe
from ndx_extended_hours import ExtendedModel,predict
from ndx_loss_optimizer import RUN,ROOT,PARENT,METHODS,paths,checked_record,prepared_fold,candidate_specs,method_spec
from ndx_policy_engine import policy_replay
from ndx_ranking_objectives import rank_ic
from pit_research_engine import load_data,fold,continuous_backtest,etf_backtest
from report_ndx_ranking import joint_intervals
from research_engine import performance
from run_ndx_alpaca_training import sha,write
from run_ndx_extended_hours_top5 import cagr_samples
from run_ndx_loss_optimizer import complete
from run_robust_research import ensemble_predictions
from training_device import configure_device

GROUPS=['ten_seed','original_five','new_five']


def error_diversity(raw_a,raw_b,target,mask):
    """Compare raw return forecasts in common percentage-point units."""
    a=np.mean(raw_a,axis=0);b=np.mean(raw_b,axis=0)
    return float(np.corrcoef((a-target)[mask],(b-target)[mask])[0,1])


def account(method,group,k=10,seed='ensemble',cost=1):return f'{method}/{group}/{seed}/{k}/{cost}'
def read(name):return pd.read_csv(RUN/name,dtype={'seed':str})


def freeze_evaluation():
    files=[Path(__file__),ROOT/'test_ndx_loss_optimizer_evaluation.py',ROOT/'report_ndx_ranking.py',
        ROOT/'compare_ndx_strategies_20261020.py',ROOT/'run_ndx_extended_hours_top5.py',ROOT/'ndx_policy_engine.py',
        ROOT/'audit_historical_ndx.py',ROOT/'ndx_ranking_objectives.py']
    lock={str(p.resolve()):sha(p) for p in files};target=RUN/'EVALUATION_SOURCE_LOCK.json'
    if target.exists() and json.loads(target.read_text())!=lock:raise ValueError('Frozen evaluation changed')
    write(target,lock)


def forecasts():
    plan,selection=complete();data=load_data();ids=np.concatenate([fold(data,y)[2] for y in plan['years']]);raw={};scores={}
    for method in METHODS:
        raw[method]={}
        for seed in plan['seeds']:
            parts=[]
            for year in plan['years']:
                spec=method_spec(method,year,selection)
                with np.load(paths(*spec,year,seed)[1]) as saved:parts.append(saved['test'].copy())
            raw[method][seed]=np.concatenate(parts)
        for group,seeds in [('ten_seed',plan['seeds']),('original_five',plan['seeds'][:5]),('new_five',plan['seeds'][5:])]:
            scores[method,group]=ensemble_predictions([raw[method][seed] for seed in seeds],data['mask'][ids])
    return plan,selection,data,ids,raw,scores


def family(group):
    result=[];base=account('fixed_adamw_mse',group)
    for method in METHODS:
        a=account(method,group)
        for ticker in ['SPY','QQQ']:result.append((method+' minus '+ticker,('account',a),('account',account(ticker,'benchmark',0))))
        if method!='fixed_adamw_mse':result.append((method+' minus fixed MSE',('account',a),('account',base)))
    for opt in ['radam','sgd']:result.append(('tuned '+opt+' minus tuned AdamW',('account',account('tuned_'+opt+'_mse',group)),('account',account('tuned_adamw_mse',group))))
    result.append(('pseudo-Huber minus Huber',('account',account('fixed_adamw_pseudohuber',group)),('account',account('fixed_adamw_huber',group))))
    assert len(result)==20
    return result


def evaluate():
    if (RUN/'EVALUATED.json').exists():raise ValueError('Completed evaluation is immutable')
    import unittest
    tests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName('test_ndx_loss_optimizer_evaluation'))
    if not tests.wasSuccessful():raise ValueError('Evaluation tests failed')
    freeze_evaluation();plan,selection,data,ids,raw,scores=forecasts()
    metrics=[];curves=[];ledgers=[];quality=[];ranking=[]
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
        for seed,score in raw[method].items():
            m,p,l=continuous_backtest(data,ids,score,10,1);record(method,'individual',10,seed,1,m,p,l)
            quality.append(dict(method=method,seed=seed,return_mse_pp2=float(((score-data['y'][ids]*100)**2)[data['labelmask'][ids]].mean()),
                rankic=float(np.nanmean(rank_ic(score,data['y'][ids],data['labelmask'][ids])))))
        print('Evaluated all fixed portfolios for',method,flush=True)
    for ticker in ['SPY','QQQ']:
        for cost in [1,2,4]:
            m,p=etf_backtest(data,ticker,cost);record(ticker,'benchmark',0,'ensemble',cost,m,p)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True)
    assert len(mf)==174 and mf.id.is_unique
    mf.to_csv(RUN/'metrics.csv',index=False);cf.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'ledgers.csv',index=False)
    pd.DataFrame(quality).to_csv(RUN/'prediction_quality.csv',index=False);pd.DataFrame(ranking).to_csv(RUN/'ranking.csv',index=False)
    annual=[];monthly=[]
    for key,p in cf.groupby('id',sort=False):
        dates=pd.to_datetime(p.date)
        for year,q in p.groupby(dates.dt.year):annual.append(dict(id=key,year=int(year),**performance(q.net)))
        for month,q in p.groupby(dates.dt.to_period('M').astype(str)):monthly.append(dict(id=key,month=month,net_return=float(np.prod(1+q.net)-1)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False);pd.DataFrame(monthly).to_csv(RUN/'monthly.csv',index=False)
    order=mf.id.tolist();ps={key:p.net.to_numpy() for key,p in cf.groupby('id',sort=False)}
    matrix=np.column_stack([ps[key] for key in order]);sr_intervals=[];differences=[]
    for block in [5,20,60]:
        samples=bootstrap_sharpe(matrix,block=block);growth=cagr_samples(matrix,block)
        lower,upper=np.quantile(samples,[.025,.975],axis=0)
        for j,key in enumerate(order):sr_intervals.append(dict(id=key,block_sessions=block,sr_low=lower[j],sr_high=upper[j]))
        for group in GROUPS:
            comparisons=family(group)
            for measure,draws,observed in [('sharpe_difference',samples,mf.sharpe.to_numpy()),('cagr_difference',growth,mf.annualized_return.to_numpy())]:
                values=np.column_stack([draws[:,order.index(a[1])]-draws[:,order.index(b[1])] for _,a,b in comparisons])
                point=np.array([observed[order.index(a[1])]-observed[order.index(b[1])] for _,a,b in comparisons])
                error=values.std(0,ddof=1);critical=float(np.quantile(np.max(np.abs((values-point)/np.maximum(error,1e-12)),axis=1),.95))
                for j,(name,a,b) in enumerate(comparisons):differences.append(dict(group=group,contrast=name,measure=measure,left_id=a[1],right_id=b[1],block_sessions=block,
                    estimate=point[j],paired_low=np.quantile(values[:,j],.025),paired_high=np.quantile(values[:,j],.975),
                    simultaneous_low=point[j]-critical*error[j],simultaneous_high=point[j]+critical*error[j],family_comparisons=20))
    pd.DataFrame(sr_intervals).to_csv(RUN/'sharpe_intervals.csv',index=False);pd.DataFrame(differences).to_csv(RUN/'paired_intervals.csv',index=False)
    active=[]
    for group in GROUPS:
        frame=joint_intervals({('account',k):v for k,v in ps.items()},family(group),[5,20,60],annualizer=252.);frame['group']=group;active.append(frame)
    pd.concat(active,ignore_index=True).to_csv(RUN/'active_mean_intervals.csv',index=False)
    # Complementarity diagnostics do not select model weights.
    diversity=[];mask=data['labelmask'][ids]
    group_seeds={'ten_seed':plan['seeds'],'original_five':plan['seeds'][:5],'new_five':plan['seeds'][5:]}
    for method in METHODS[1:]:
        for group in GROUPS:
            a=scores[method,group];b=scores['fixed_adamw_mse',group]
            seeds=group_seeds[group];benchmark=ps[account('QQQ','benchmark',0)]
            diversity.append(dict(method=method,group=group,score_correlation=float(np.corrcoef(a[mask],b[mask])[0,1]),
                raw_return_error_correlation=error_diversity([raw[method][s] for s in seeds],
                    [raw['fixed_adamw_mse'][s] for s in seeds],data['y'][ids]*100,mask),
                net_return_correlation=float(np.corrcoef(ps[account(method,group)],ps[account('fixed_adamw_mse',group)])[0,1]),
                qqq_active_return_correlation=float(np.corrcoef(ps[account(method,group)]-benchmark,
                    ps[account('fixed_adamw_mse',group)]-benchmark)[0,1])))
    pd.DataFrame(diversity).to_csv(RUN/'complementarity.csv',index=False)
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',accounts=174,methods=6,primary_group='ten_seed'))
    print('All declared accounts and uncertainty diagnostics computed',flush=True)


def audit():
    complete()
    for p,h in json.loads((RUN/'EVALUATION_SOURCE_LOCK.json').read_text()).items():
        if sha(p)!=h:raise ValueError('Evaluation source changed')
    plan,selection,data,ids,raw,scores=forecasts();device=configure_device('cuda');reproduced=0
    for year in plan['years']:
        data,x,extra,tr,va,te,groups=prepared_fold(year)
        for loss,opt,lr,y,seed in [s for s in candidate_specs(plan,selection) if s[3]==year]:
            cp,pred,hist,record=paths(loss,opt,lr,y,seed);meta=torch.load(cp,map_location='cpu',weights_only=False)
            model=ExtendedModel(len(data['panel'].tickers),meta['groups'],'after_hours_premarket').to(device);model.load_state_dict(meta['state_dict'])
            vp,tp=[predict(model,x,extra,data['mask'],sub,device,plan['batch']) for sub in [va,te]]
            with np.load(pred) as saved:
                np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0);np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
            best=np.inf;chosen=0
            for item in json.loads(hist.read_text()):
                if item['validation_mse_pp2']<best-1e-7:best=item['validation_mse_pp2'];chosen=item['epoch']
            if chosen!=meta['epoch']:raise ValueError('Selected checkpoint does not match validation history')
            np.testing.assert_allclose(((vp-data['y'][va]*100)**2)[data['labelmask'][va]].mean(),best,atol=1e-6,rtol=1e-6)
            reproduced+=1;del model
        print('Reproduced all candidate checkpoint exports through',year,reproduced,flush=True)
    # Independently recompute the twelve validation-only LR selections.
    import math
    for year in plan['years']:
        for opt,rates in plan['learning_rates'].items():
            values=[]
            for lr in rates:
                losses=[checked_record('mse',opt,lr,year,seed)['validation_mse'] for seed in plan['pilot_seeds']]
                values.append((lr,math.fsum(losses)/len(losses)))
            best=min(v for _,v in values);chosen=min(lr for lr,v in values if v<=best+1e-7)
            if chosen!=selection[str(year)][opt]['lr']:raise ValueError('LR selection used an unexpected criterion')
    mf=read('metrics.csv').set_index('id');cf=read('curves.csv');largest=0.;accounts=0;refs=0
    prior=pd.read_csv(ROOT/'docs/ndx_extended_hours_20261024/curves.csv',dtype={'seed':str})
    prior_five=pd.read_csv(ROOT/'docs/ndx_extended_hours_top5_20261024/curves.csv',dtype={'seed':str})
    for key,p in cf.groupby('id',sort=False):
        row=mf.loc[key];method,group,k,cost=row.method,row.group,int(row.k),int(row.cost_multiplier)
        assert len(p)==1003 and p.date.is_unique and p.date.is_monotonic_increasing
        for field,value in independent_metrics(p.net).items():np.testing.assert_allclose(value,row[field],atol=1e-10,rtol=1e-10)
        if method in ['SPY','QQQ']:
            root=Path(data['manifest']['root']);first=data['entry'][ids[0]];q=pd.read_csv(root/'etf_adj_close.csv')[method].to_numpy();v=pd.read_csv(root/'etf_dollar_volume.csv')[method].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/v[first-20:first].mean()/.01));nav=1e6/(1+rate)/q[first]*q[first:]
            nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/v[-21:-1].mean()/.01))
        else:nav=policy_replay(data,ids,raw[method][int(row.seed)] if group=='individual' else scores[method,group],k,cost)
        actual=1e6*np.cumprod(1+p.net.to_numpy());largest=max(largest,float(np.abs(nav-actual).max()));np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10);accounts+=1
        match=None
        if method in ['SPY','QQQ']:match=prior[(prior['view']==method)&(prior.cost_multiplier==cost)]
        elif method=='fixed_adamw_mse' and k==10:match=prior[(prior['view']=='after_hours_premarket')&(prior.group==group)&(prior.seed==row.seed)&(prior.cost_multiplier==cost)]
        elif method=='fixed_adamw_mse' and k==5:
            source=f'after_hours_premarket/{group}/{row.seed}/5/{cost}';match=prior_five[prior_five.id==source]
        if match is not None:
            np.testing.assert_array_equal(match.date,p.date);np.testing.assert_allclose(match.net,p.net,atol=1e-14,rtol=0);refs+=1
    ledger=read('ledgers.csv');np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    assert accounts==174 and refs==34 and ledger.cash.min()>-1e-5
    # Materialized sample checks independently verify uncertainty for primary controls/loss variants.
    selected=[account(m,'ten_seed') for m in METHODS[:3]]+[account('SPY','benchmark',0)]
    matrix=np.column_stack([cf[cf.id==k].net.to_numpy() for k in selected]);n=len(matrix);rng=np.random.default_rng(20261040);ss=[];gg=[]
    for _ in range(100):
        starts=rng.integers(0,n,(100,int(np.ceil(n/20))));index=((starts[:,:,None]+np.arange(20))%n).reshape(100,-1)[:,:n];sample=matrix[index]
        ss.append(np.sqrt(252)*(sample.mean(1)-((1.03)**(1/252)-1))/sample.std(1,ddof=1));gg.append(np.expm1(np.log1p(sample).sum(1)*252/n))
    np.testing.assert_allclose(np.concatenate(ss),bootstrap_sharpe(matrix,block=20),atol=2e-10,rtol=1e-10)
    np.testing.assert_allclose(np.concatenate(gg),cagr_samples(matrix,20),atol=2e-10,rtol=1e-10)
    proof=dict(status='passed',checkpoint_exports_reproduced=reproduced,independent_accounts=174,prior_accounts_reproduced=34,
        validation_lr_choices_checked=12,materialized_bootstrap_accounts=4,bootstrap_replicates=10000,largest_nav_difference_dollars=largest)
    write(RUN/'AUDIT.json',proof)
    ranking=read('ranking.csv');ci=read('sharpe_intervals.csv');primary=read('metrics.csv')
    primary=primary[(primary.seed=='ensemble')&(primary.k==10)&(primary.cost_multiplier==1)].merge(ranking,on=['method','group'])
    primary=primary.merge(ci[ci.block_sessions==20],on='id')
    seed_rows=[];full=read('metrics.csv')
    for method in METHODS:
        seed=full[(full.method==method)&(full.group=='individual')].set_index('seed').annualized_return
        base=full[(full.method=='fixed_adamw_mse')&(full.group=='individual')].set_index('seed').annualized_return
        seed_rows.append(dict(method=method,median_seed_cagr=float(seed.median()),seeds_above_paired_mse=int((seed>base).sum()),seeds_above_spy=int((seed>.1085244627962889).sum())))
    write(RUN/'RESULTS.json',dict(status='completed_audited_exploratory',primary_group='ten_seed',
        top_ten_base_cost=json.loads(primary.to_json(orient='records')),seed_summary=seed_rows,
        validation_selection=selection,fit_inventory=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())['unique_candidate_fits'],
        inference='Previously inspected history; no untouched confirmation or deployment'))
    write(RUN/'PROGRESS.json',dict(stage='completed_audited',accounts=174));print('Loss/optimizer comparison complete and independently audited',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','audit'])
    globals()[parser.parse_args().stage]()
