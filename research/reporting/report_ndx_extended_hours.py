"""Evaluate, independently audit and disclose the extended-hours information test."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from compare_ndx_strategies_20261020 import bootstrap_sharpe
from ndx_extended_hours import ExtendedModel,prepared,raw_auxiliary,scale_auxiliary,predict,timing
from ndx_policy_engine import policy_replay
from ndx_ranking_objectives import rank_ic
from pit_research_engine import fold,historical_groups,continuous_backtest,etf_backtest
from prepare_ndx_alpaca import CALENDAR,RAW,calendar_frame
from prepare_ndx_extended_hours import OUT as FEATURES,FIELDS
from report_ndx_broad import md_table,pct
from report_ndx_ranking import joint_intervals
from research_engine import performance
from run_ndx_alpaca_training import ROOT,sha,write
from run_ndx_extended_hours import RUN,complete,old_paths
from run_robust_research import ensemble_predictions,scale_inputs,predict as predict_core

DOCS=ROOT/'docs/ndx_extended_hours_20261024'
NAMES=dict(daily='Daily inputs only',after_hours='Daily + after-hours',after_hours_premarket='Daily + after-hours + premarket',
           SPY='SPY',QQQ='QQQ',monthly_basket='Monthly equal-weight index basket')


def source_lock(name,files):
    value={str(Path(p).resolve()):sha(p) for p in files};path=RUN/name
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen evaluation/reference changed')
    write(path,value)


def forecast_maps(data,plan,ranker=ensemble_predictions):
    ids=np.concatenate([fold(data,y)[2] for y in plan['test_years']]);mask=data['mask'][ids];raw={};scores={}
    for view in plan['views']:
        raw[view]={}
        for seed in plan['seeds']:
            pieces=[]
            for year in plan['test_years']:
                with np.load(RUN/view/str(year)/f'{seed}_predictions.npz') as saved:pieces.append(saved['test'].copy())
            raw[view][seed]=np.concatenate(pieces)
        for group,seeds in [('ten_seed',plan['seeds']),('original_five',plan['seeds'][:5]),('new_five',plan['seeds'][5:])]:
            scores[view,group]=ranker([raw[view][seed] for seed in seeds],mask)
    return ids,raw,scores


def evaluate():
    plan=complete()
    if (RUN/'EVALUATED.json').exists():raise ValueError('Completed extended-hours evaluation is immutable')
    receipt=json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    if receipt['status']!='passed' or receipt['report_sha256']!=sha(__file__) or receipt['test_sha256']!=sha(ROOT/'test_ndx_extended_hours_report.py'):
        raise ValueError('Evaluation preflight missing or stale')
    source_lock('EVALUATION_SOURCE_LOCK.json',[__file__,ROOT/'test_ndx_extended_hours_report.py',ROOT/'report_ndx_ranking.py',
          ROOT/'compare_ndx_strategies_20261020.py',ROOT/'report_ndx_broad.py'])
    data,cubes=prepared();ids,raw,scores=forecast_maps(data,plan);cal=calendar_frame(json.loads(CALENDAR.read_text()))
    timing(data,cal)  # Recheck information availability before computing any path.
    metrics,curves,ledgers,rankings,periods,quality,paths=[],[],[],[],[],[],{}
    np.savez_compressed(RUN/'scores.npz',ids=ids,**{v+'__'+g:s for (v,g),s in scores.items()})
    def record(view,group,seed,cost,m,p,l=None):
        tags=dict(view=view,group=group,seed=str(seed),cost_multiplier=cost)
        metrics.append(dict(**tags,**m));curves.append(p.assign(**tags))
        if group=='ten_seed' and cost==1:paths['ten_seed',view]=p.net.to_numpy()
        if group in ['original_five','new_five'] and cost==1:paths[group,view]=p.net.to_numpy()
        if l is not None and group=='ten_seed' and cost==1:ledgers.append(l.assign(**tags))
    for view in plan['views']:
        for group in ['ten_seed','original_five','new_five']:
            score=scores[view,group]
            for cost in plan['cost_multipliers']:
                m,p,l=continuous_backtest(data,ids,score,10,cost);record(view,group,'ensemble',cost,m,p,l)
            ic=rank_ic(score,data['y'][ids],data['labelmask'][ids])
            rankings.append(dict(view=view,group=group,rankic=float(np.nanmean(ic)),observed_origins=int(np.isfinite(ic).sum())))
            for j,i in enumerate(ids):periods.append(dict(view=view,group=group,signal_date=str(data['panel'].dates[data['origin'][i]].date()),rankic=ic[j]))
        for seed,prediction in raw[view].items():
            m,p,l=continuous_backtest(data,ids,prediction,10,1);record(view,'individual',seed,1,m,p,l)
            observed=data['labelmask'][ids]
            quality.append(dict(view=view,seed=seed,return_mse_pp2=float(((prediction-data['y'][ids]*100)**2)[observed].mean()),
                                rankic=float(np.nanmean(rank_ic(prediction,data['y'][ids],observed)))))
        print('Evaluated all seed groups, costs and individual seeds:',view,flush=True)
    for cost in plan['cost_multipliers']:
        for ticker in ['SPY','QQQ']:
            m,p=etf_backtest(data,ticker,cost);record(ticker,'ten_seed','ensemble',cost,m,p)
            if cost==1:
                for group in ['original_five','new_five']:paths[group,ticker]=p.net.to_numpy()
        m,p,l=continuous_backtest(data,ids,np.zeros_like(data['y'][ids]),10,cost,basket=True)
        record('monthly_basket','ten_seed','ensemble',cost,m,p,l)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True)
    mf.to_csv(RUN/'metrics.csv',index=False);cf.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'primary_ledgers.csv',index=False)
    pd.DataFrame(rankings).to_csv(RUN/'ranking.csv',index=False);pd.DataFrame(periods).to_csv(RUN/'ranking_periods.csv',index=False)
    pd.DataFrame(quality).to_csv(RUN/'prediction_quality.csv',index=False)
    annual,monthly,seed_summary=[],[],[]
    for key,p in cf.groupby(['view','group','seed','cost_multiplier'],sort=False):
        tags=dict(zip(['view','group','seed','cost_multiplier'],key));date=pd.to_datetime(p.date)
        for year,part in p.groupby(date.dt.year):annual.append(dict(**tags,year=int(year),**performance(part.net)))
        for month,part in p.groupby(date.dt.to_period('M').astype(str)):monthly.append(dict(**tags,month=month,net_return=float(np.prod(1+part.net)-1)))
    for view,part in mf[mf.group=='individual'].groupby('view'):
        spy=mf[(mf.view=='SPY')&(mf.cost_multiplier==1)].annualized_return.iloc[0]
        qqq=mf[(mf.view=='QQQ')&(mf.cost_multiplier==1)].annualized_return.iloc[0]
        base=mf[(mf.view=='daily')&(mf.group=='individual')].set_index('seed').annualized_return
        selected=part.set_index('seed').annualized_return
        seed_summary.append(dict(view=view,minimum_cagr=selected.min(),median_cagr=selected.median(),maximum_cagr=selected.max(),
            seeds_above_spy=int((selected>spy).sum()),seeds_above_qqq=int((selected>qqq).sum()),
            seeds_above_paired_daily=int((selected>base).sum()),seed_runs=len(selected)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False);pd.DataFrame(monthly).to_csv(RUN/'monthly.csv',index=False)
    pd.DataFrame(seed_summary).to_csv(RUN/'seed_summary.csv',index=False)
    for group in ['ten_seed','original_five','new_five']:
        contrasts=[]
        for view in plan['views']:
            for benchmark in ['SPY','QQQ']:contrasts.append((f'{view} minus {benchmark}',(group,view),(group,benchmark)))
        contrasts.extend([(a+' minus '+b,(group,a),(group,b)) for a,b in [('after_hours','daily'),('after_hours_premarket','daily'),('after_hours_premarket','after_hours')]])
        interval=joint_intervals(paths,contrasts,[5,20,60],annualizer=252.)
        interval['inference_role']='primary nine-contrast family' if group=='ten_seed' else 'subgroup diagnostic, separate nine-contrast family'
        interval.to_csv(RUN/f'{group}_intervals.csv',index=False)
    # Absolute SR intervals use identical daily block draws for all 66 accounts.
    ordering=[];returns=[]
    for key,p in cf.groupby(['view','group','seed','cost_multiplier'],sort=False):ordering.append(key);returns.append(p.net.to_numpy())
    result=[]
    for block in [5,20,60]:
        samples=bootstrap_sharpe(np.column_stack(returns),block=block)
        lo,hi=np.quantile(samples,[.025,.975],axis=0)
        for j,key in enumerate(ordering):result.append(dict(zip(['view','group','seed','cost_multiplier'],key),block_sessions=block,sr_low=lo[j],sr_high=hi[j]))
    pd.DataFrame(result).to_csv(RUN/'sharpe_intervals.csv',index=False)
    if len(metrics)!=66 or len(rankings)!=9:raise ValueError('Unexpected account/rank inventory')
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',fits=120,accounts=66,primary_contrasts=9,
        timing='09:15 data cutoff, 09:20 entry-day decision, entry-day regular-close fill',history='previously inspected development'))


def reference_summary(frame,calendar,date,segment):
    """Scalar reconstruction independent of the vectorized feature builder."""
    day=pd.Timestamp(date);opening=calendar.at[day,'open_utc'];close=calendar.at[day,'close_utc']
    local=lambda clock:pd.Timestamp(str(day.date())+' '+clock,tz='America/New_York').tz_convert('UTC')
    start,end=(close,local('20:00')) if segment=='after_hours' else (local('04:00'),local('09:15'))
    active=[];anchor=None
    for row in frame.itertuples(index=False):
        ts=pd.Timestamp(row.timestamp_utc);finish=pd.Timestamp(row.bar_end_utc)
        if ts.tz_convert('America/New_York').date()!=day.date():continue
        valid=ts.minute%15==0 and ts.second==0 and finish-ts==pd.Timedelta(minutes=15) and row.volume>0 and row.trade_count>0
        if not valid:continue
        if finish==close and ts>=opening:anchor=float(row.close)
        if ts>=start and finish<=end:active.append((ts,finish,row))
    active.sort(key=lambda r:r[0]);count=len(active)
    if not count:return np.array([np.nan]*7+[0.,1.,1.]),0
    duration=(end-start).total_seconds();first,last=active[0],active[-1]
    result=[np.nan]*7+[count/(duration/900),(first[0]-start).total_seconds()/duration,(end-last[1]).total_seconds()/duration]
    if count<4:return np.array(result),count
    bars=[r[2] for r in active];volume=sum(r.volume for r in bars);late=[r[2] for r in active if r[0]>=end-pd.Timedelta(hours=1)]
    price=float(last[2].close);base=anchor if segment=='after_hours' else float(first[2].open)
    result[:7]=[np.nan if base is None else price/base-1,max(r.high for r in bars)/min(r.low for r in bars)-1,
        np.sqrt(sum(np.log(r.close/r.open)**2 for r in bars)),price/(sum(r.vwap*r.volume for r in bars)/volume)-1,
        np.log1p(sum(r.close*r.volume for r in bars)),price/late[0].open-1 if late else np.nan,
        sum(r.volume for r in late)/volume if late else 0.]
    return np.asarray(result),count


def audit():
    from audit_historical_ndx import independent_metrics
    from audit_historical_ndx_ranking import average_ranks
    from training_device import configure_device
    plan=complete()
    if not (RUN/'EVALUATED.json').exists():raise ValueError('Evaluation must precede audit')
    for name in ['EVALUATION_SOURCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Evaluation source changed')
    source_lock('AUDIT_SOURCE_LOCK.json',[__file__,ROOT/'audit_historical_ndx.py',ROOT/'audit_historical_ndx_ranking.py',ROOT/'ndx_policy_engine.py'])
    data,cubes=prepared();cal=calendar_frame(json.loads(CALENDAR.read_text()));decision=timing(data,cal)
    pd.testing.assert_frame_equal(pd.read_csv(RUN/'TIMING.csv'),decision)
    review=json.loads((FEATURES/'REVIEW.json').read_text())
    for name,h in review['source_hashes'].items():
        if sha(RAW/name)!=h:raise ValueError('Raw source changed after feature review')
    summary=pd.read_csv(FEATURES/'summaries.csv.gz');samples=[];cache={}
    for segment in ['after_hours','premarket']:
        part=summary[summary.segment==segment]
        clock='20:00' if segment=='after_hours' else '09:15'
        deadline=pd.to_datetime(part.date+' '+clock).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        if not (pd.to_datetime(part.latest_bar_end,utc=True)<=deadline).all():raise ValueError('Summary includes a post-cutoff bar')
    after_hours_cutoff=pd.to_datetime(data['panel'].dates[data['origin']].strftime('%Y-%m-%d')+' 20:00').tz_localize('America/New_York').tz_convert('UTC')
    if not (after_hours_cutoff<pd.DatetimeIndex(pd.to_datetime(decision.feature_cutoff_utc,utc=True))).all():raise ValueError('After-hours window exceeds the common information cutoff')
    for segment in ['after_hours','premarket']:
        candidates=summary[(summary.segment==segment)&summary.security_id.isin(data['panel'].tickers)]
        # Forty deterministic source/date strata per segment, not chosen by returns.
        chosen=candidates.iloc[np.linspace(0,len(candidates)-1,40,dtype=int)]
        for row in chosen.itertuples(index=False):
            year=int(row.date[:4]);path=RAW/'bars'/row.source_symbol/f'{year}-01-01_{year+1}-01-01.csv.gz'
            if path not in cache:cache[path]=pd.read_csv(path)
            values,count=reference_summary(cache[path],cal,row.date,segment)
            day=data['panel'].dates.get_loc(pd.Timestamp(row.date));node=data['panel'].tickers.index(row.security_id)
            np.testing.assert_allclose(values,cubes[segment][day,node],atol=2e-6,rtol=2e-6,equal_nan=True)
            samples.append(dict(segment=segment,symbol=row.source_symbol,security_id=row.security_id,date=row.date,observed=count))
    pd.DataFrame(samples).to_csv(RUN/'independent_feature_samples.csv',index=False)
    device=configure_device('cuda');reproduced=0;baselines=0;parent_files=[]
    for year in plan['test_years']:
        tr,va,te=fold(data,year);x,mean,std=scale_inputs(data,tr);groups=historical_groups(data,tr)
        for view in plan['views']:
            raw=raw_auxiliary(data,cubes,view);extra,amean,astd=scale_auxiliary(raw,data['mask'],tr)
            altered=raw.copy();altered[max(tr)+1:]=1e10
            _,newmean,newstd=scale_auxiliary(altered,data['mask'],tr)
            np.testing.assert_array_equal(amean,newmean);np.testing.assert_array_equal(astd,newstd)
            folder=RUN/view/str(year)
            with np.load(folder/'preprocessing.npz') as saved:
                for name,value in dict(train=tr,val=va,test=te,mean=mean,std=std,aux_mean=amean,aux_std=astd).items():np.testing.assert_array_equal(saved[name],value)
            for seed in plan['seeds']:
                cp=torch.load(folder/f'{seed}.pt',map_location='cpu',weights_only=False)
                if (cp['view'],cp['seed'],cp['year'])!=(view,seed,year):raise ValueError('Checkpoint metadata mismatch')
                model=ExtendedModel(len(data['panel'].tickers),groups,view).to(device);model.load_state_dict(cp['state_dict'])
                vp,tp=[predict(model,x,extra,data['mask'],sub,device,plan['batch']) for sub in [va,te]]
                with np.load(folder/f'{seed}_predictions.npz') as saved:
                    np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0);np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
                for prediction,sub in [(vp,va),(tp,te)]:
                    core=predict_core(model.core,x,data['mask'],sub,plan['batch']);eligible=data['mask'][sub]
                    difference=prediction-core
                    np.testing.assert_allclose(np.where(eligible,difference,0.).sum(1),0.,atol=3e-5,rtol=0)
                    sd=np.array([max(row[mask].std(),1e-4) for row,mask in zip(core,eligible)])
                    if np.any(np.abs(difference[eligible])>np.broadcast_to(sd[:,None],difference.shape)[eligible]*1.00001+1e-6):raise ValueError('Auxiliary correction exceeded its bound')
                history=json.loads((folder/f'{seed}_history.json').read_text());best=np.inf;chosen=None
                for row in history:
                    if row['validation_mse_pp2']<best-1e-7:best=row['validation_mse_pp2'];chosen=row['epoch']
                if chosen!=cp['epoch']:raise ValueError('Validation checkpoint choice mismatch')
                np.testing.assert_allclose(((vp-data['y'][va]*100)**2)[data['labelmask'][va]].mean(),best,atol=1e-6,rtol=1e-6)
                if view=='daily':
                    previous,previous_cp=old_paths(year,seed);parent_files.extend([previous,previous_cp])
                    with np.load(previous) as saved:
                        np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0);np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
                    baselines+=1
                reproduced+=1
        print('Reproduced extended-hours checkpoint exports',year,reproduced,'/120',flush=True)
    source_lock('PARENT_BASELINE_LOCK.json',parent_files)
    def ranker(available,mask):
        values=[]
        for p in available:
            ranked=np.zeros_like(p,dtype=float)
            for j,m in enumerate(mask):ranked[j,m]=average_ranks(p[j,m])/m.sum()
            values.append(ranked)
        return np.mean(values,axis=0)
    ids,raw,scores=forecast_maps(data,plan,ranker)
    with np.load(RUN/'scores.npz') as saved:
        for (view,group),p in scores.items():np.testing.assert_allclose(p,saved[view+'__'+group],atol=1e-12,rtol=0)
    mf=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}).set_index(['view','group','seed','cost_multiplier'])
    cf=pd.read_csv(RUN/'curves.csv',dtype={'seed':str});accounts=0;largest=0.
    for key,path in cf.groupby(['view','group','seed','cost_multiplier'],sort=False):
        view,group,seed,cost=key
        if len(path)!=1003 or not path.date.is_unique or not path.date.is_monotonic_increasing:raise ValueError('Account calendar mismatch')
        for field,value in independent_metrics(path.net).items():np.testing.assert_allclose(value,mf.loc[key,field],atol=1e-10,rtol=1e-10)
        if view in ['SPY','QQQ']:
            root=Path(data['manifest']['root']);first=data['entry'][ids[0]];q=pd.read_csv(root/'etf_adj_close.csv')[view].to_numpy();v=pd.read_csv(root/'etf_dollar_volume.csv')[view].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/v[first-20:first].mean()/.01));nav=1e6/(1+rate)/q[first]*q[first:]
            nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/v[-21:-1].mean()/.01))
        else:
            p=np.zeros_like(data['y'][ids]) if view=='monthly_basket' else raw[view][int(seed)] if group=='individual' else scores[view,group]
            nav=policy_replay(data,ids,p,10,cost,basket=view=='monthly_basket')
        actual=1e6*np.cumprod(1+path.net.to_numpy());largest=max(largest,float(np.abs(actual-nav).max()));np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10)
        accounts+=1
    ledger=pd.read_csv(RUN/'primary_ledgers.csv');np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    if accounts!=66 or reproduced!=120 or baselines!=40 or ledger.cash.min()<-1e-5:raise ValueError('Audit count/cash failure')
    write(RUN/'AUDIT.json',dict(status='passed',checkpoints_reproduced=120,daily_parent_forecasts_reproduced=40,
        raw_chunk_hashes_verified=1288,independent_feature_samples=len(samples),timing_origins_verified=len(decision),
        feature_summary_cutoffs_verified=len(summary),
        independent_accounts=66,largest_nav_difference_dollars=largest,history='exploratory; no prospective confirmation'))


def publish():
    plan=complete();a=json.loads((RUN/'AUDIT.json').read_text())
    if a['status']!='passed':raise ValueError('Audit must pass')
    for name in ['EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','PARENT_BASELINE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Publication source/reference changed')
    DOCS.mkdir(parents=True,exist_ok=True);figures=DOCS/'figures';figures.mkdir(exist_ok=True)
    mf=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str});cf=pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    rank=pd.read_csv(RUN/'ranking.csv').set_index(['view','group']);sr=pd.read_csv(RUN/'sharpe_intervals.csv',dtype={'seed':str})
    main=mf[(mf.group=='ten_seed')&(mf.cost_multiplier==1)].set_index('view')
    intervals=pd.read_csv(RUN/'ten_seed_intervals.csv');ci=intervals[intervals.block_length==20]
    review=json.loads((FEATURES/'REVIEW.json').read_text());coverage=pd.read_csv(FEATURES/'coverage.csv')
    annual=pd.read_csv(RUN/'annual.csv',dtype={'seed':str});monthly=pd.read_csv(RUN/'monthly.csv',dtype={'seed':str});seed=pd.read_csv(RUN/'seed_summary.csv')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in ['ten_seed','original_five','new_five']:
        fig,ax=plt.subplots(figsize=(12,6))
        for view,color in [('daily','black'),('after_hours','#1f77b4'),('after_hours_premarket','#2ca02c'),('SPY','#cc0000'),('QQQ','#9467bd')]:
            actual='ten_seed' if view in ['SPY','QQQ'] else group
            p=cf[(cf.view==view)&(cf.group==actual)&(cf.cost_multiplier==1)]
            m=mf[(mf.view==view)&(mf.group==actual)&(mf.cost_multiplier==1)].iloc[0]
            ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),color=color,label=NAMES[view]+': '+pct(m.annualized_return))
        ax.set(title=f'Extended-hours information: {group}; exploratory 2022–2025',ylabel='Net wealth per initial $1');ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(figures/(group+'_wealth.png'),dpi=160);plt.close(fig)
    subset=ci[ci.right_model.isin(['daily','after_hours'])];fig,ax=plt.subplots(figsize=(11,4))
    for j,row in enumerate(subset.itertuples(index=False)):
        ax.plot(np.array([row.simultaneous_low,row.simultaneous_high])*100,[j,j],color='#cccccc',linewidth=6)
        ax.plot(np.array([row.paired_low,row.paired_high])*100,[j,j],color='#1f77b4',linewidth=2);ax.scatter(row.estimate*100,j,color='black')
    ax.set(yticks=range(len(subset)),yticklabels=subset.contrast,xlabel='Annual arithmetic active mean, percentage points',title='Information increments: paired 95% and nine-contrast simultaneous diagnostics');ax.axvline(0,color='black');fig.tight_layout();fig.savefig(figures/'information_intervals.png',dpi=160);plt.close(fig)
    rows=[]
    for group in ['ten_seed','original_five','new_five']:
        for view in plan['views']:
            r=mf[(mf.view==view)&(mf.group==group)&(mf.cost_multiplier==1)].iloc[0]
            interval=sr[(sr.view==view)&(sr.group==group)&(sr.cost_multiplier==1)&(sr.block_sessions==20)].iloc[0]
            rows.append([NAMES[view],group,f'{r.sharpe:.3f}',pct(r.annualized_return),f'{rank.loc[view,group].rankic:.4f}',pct(r.maximum_drawdown),f'[{interval.sr_low:.3f}, {interval.sr_high:.3f}]'])
    for view in ['SPY','QQQ','monthly_basket']:
        r=main.loc[view];s=sr[(sr.view==view)&(sr.cost_multiplier==1)&(sr.block_sessions==20)].iloc[0]
        rows.append([NAMES[view],'benchmark',f'{r.sharpe:.3f}',pct(r.annualized_return),'N/A',pct(r.maximum_drawdown),f'[{s.sr_low:.3f}, {s.sr_high:.3f}]'])
    costs=[];consistency=[]
    for view in plan['views']:
        p=mf[(mf.view==view)&(mf.group=='ten_seed')].set_index('cost_multiplier')
        costs.append([NAMES[view]]+[pct(p.loc[c,'annualized_return']) for c in [1,2,4]])
        aview=annual[(annual.view==view)&(annual.group=='ten_seed')&(annual.cost_multiplier==1)].set_index('year').cumulative_return
        spyannual=annual[(annual.view=='SPY')&(annual.cost_multiplier==1)].set_index('year').cumulative_return
        mview=monthly[(monthly.view==view)&(monthly.group=='ten_seed')&(monthly.cost_multiplier==1)].set_index('month').net_return
        spymonthly=monthly[(monthly.view=='SPY')&(monthly.cost_multiplier==1)].set_index('month').net_return
        consistency.append([NAMES[view],f'{int((aview>spyannual).sum())}/4',f'{int((mview>spymonthly).sum())}/48'])
    cirows=[[r.contrast,f'{r.estimate*100:.2f} pp',f'[{r.paired_low*100:.2f}, {r.paired_high*100:.2f}]',f'[{r.simultaneous_low*100:.2f}, {r.simultaneous_high*100:.2f}]'] for r in ci.itertuples(index=False)]
    seedrows=[[NAMES[r.view],pct(r.minimum_cagr),pct(r.median_cagr),pct(r.maximum_cagr),f'{r.seeds_above_spy}/10',f'{r.seeds_above_paired_daily}/10'] for r in seed.itertuples(index=False)]
    base=main.loc['daily'];ah=main.loc['after_hours'];pm=main.loc['after_hours_premarket']
    lines=['# After-hours and premarket information — October 24, 2026','',
        f'The primary ten-seed daily control returned **{pct(base.annualized_return)} CAGR**, daily plus after-hours **{pct(ah.annualized_return)}**, and daily plus after-hours plus premarket **{pct(pm.annualized_return)}**. '
        f'SPY returned **{pct(main.loc["SPY"].annualized_return)}** and QQQ **{pct(main.loc["QQQ"].annualized_return)}**. '
        'These are previously inspected 2022–2025 development results. No claim of reliable future improvement follows from the highest historical row.','',
        md_table(['Information view','Seed group','Sharpe','Net CAGR','Raw-return RankIC','Max drawdown','Pointwise 95% SR interval'],rows),'',
        '## Timing and the matched experiment','',
        'Every view forms its forecast and share order at **09:20 Eastern on the entry session**, using no premarket bar ending after 09:15. '
        'Orders fill at that same session’s regular close, including 13:00 early closes. Thus the evening/premarket move is not credited as a return earned before entry. '
        'The bookkeeping origin remains the preceding daily session for alignment with earlier studies. Share sizing deliberately uses that known prior regular close’s NAV, marks and ADV; '
        'it does not use a future closing price or a synthetic premarket valuation. The timing inventory verifies cutoff < decision < fill for every origin.','',
        'After-hours runs from the preceding regular close through 20:00 ET. Early-close sessions start this window at 13:00. '
        'Premarket uses 04:00–09:15 of the following EXCHANGE session, skipping weekends and holidays. The 16 daily input slots each align their AH window and following PM window. '
        'Only fully elapsed, positive-volume/trade-count, on-grid 15-minute bars enter summaries. No trade-bar forward fill or future-liquidity member deletion is permitted.','',
        'The core remains the original five-input THINK+mixer. A shared 320-to-16-to-1 branch takes 16 windows × 20 extended-hours features, with a zero-start, bounded, centered score correction. '
        'Augmented cores and branches train jointly using the original raw-return MSE, hyperparameters, chronological batches, train-only graph/scaling and prior-year MSE early stopping. '
        'The daily branch is bypassed exactly, preserving the original core without padding its feature-normalization layers. All three models have the same parameter inventory, with unused auxiliary capacity in the daily control. '
        'All 40 daily-control annual seed exports and selected epochs reproduced their prior checkpoints.','',
        'The primary blend contains ten fixed seeds: the original five and the first five from the previously frozen seed-expansion list, without performance-based selection. '
        'Original-five and new-five subgroups are disclosed as diagnostics. No information view, seed, holding count or checkpoint was selected from new test returns. '
        'All 120 fits finished and were hash locked before these test metrics were computed. All strategies target equal top ten, full exposure, five-session holdings and $1 million continuous accounts. '
        'The original modeled fees, 1% ADV caps, cancellations, cash scaling and acquisition/final-liquidation approximations apply. Cash earns zero interest. No extended-hours orders were simulated or sent.','',
        '![Primary ten-seed paths](figures/ten_seed_wealth.png)','',
        '![Original five-seed diagnostic](figures/original_five_wealth.png)','',
        '## Data coverage and limitations','',
        md_table(['Segment','Member sessions with at least four active bars','Mean active-bar coverage'],
                 [[s,pct(review['usable_member_fraction'][s]),pct(review['mean_member_coverage'][s])] for s in ['after_hours','premarket']]),'',
        'Ten features per segment describe return, range, within-bar realized volatility, close relative to VWAP, log dollar notional, last-hour return, late-volume share, coverage, first-trade delay and last-trade age. '
        'At least four active bars are required for numeric summaries. Missing numeric values use training-only means; coverage and activity-age fields expose sparse or absent trading. '
        'Delay and age use the first active bar start and last active bar end: they are 15-minute boundary proxies, not exact trade timestamps. '
        'Observed activity is not certification that the vendor captured every trade. Detailed stock/year coverage is retained.','',
        'AH return anchors to the same-date regular closing bar. PM return is measured within its own session, deliberately omitting the prior-close raw price gap so stock splits/renames cannot create an artificial cross-date return. '
        'Consequently this experiment does not test every possible overnight feature. Same-window price ratios and dollar-notional volume avoid raw split-unit discontinuities. '
        'Issuer-date exclusions are inherited from the reviewed literal-ticker study. Trade aggregates omit quotes, spread, depth and exact publication latency; retrospective revisions remain possible. '
        'The five-minute availability lag is an assumption, not independently verified first-publication timing.','',
        '## Costs, consistency and seed sensitivity','',md_table(['Primary view','1× fee CAGR','2× fee CAGR','4× fee CAGR'],costs),'',
        md_table(['Primary view','Years above SPY','Months above SPY'],consistency),'',
        md_table(['View','Minimum seed CAGR','Median','Maximum','Seeds above SPY','Seeds above their paired daily control'],seedrows),'',
        'The seed accounts share one market history. More seeds and more bars are not additional independent years. A higher CAGR accompanied by unstable seeds, costs or monthly outcomes does not establish a dependable improvement.','',
        '## Paired uncertainty','', '![Information increments](figures/information_intervals.png)','',
        md_table(['Primary ten-seed contrast','Arithmetic active mean/year','Paired 95% interval, pp','Nine-contrast simultaneous diagnostic, pp'],cirows),'',
        'Ten thousand common-date circular-block resamples use 5/20/60-session blocks, with 20 primary. One nine-contrast family covers the three ten-seed views against SPY/QQQ and all three information increments. '
        'Subgroup families are separate diagnostics. Absolute Sharpe intervals use the same fixed 3% annual hurdle and block method as the prior workbook; they are not intervals for benchmark superiority. '
        'All intervals condition on fitted paths and do not account for the complete prior research search, retraining uncertainty, vendor revisions or future regimes. Arithmetic active means differ from CAGR gaps.','',
        '## Verification and artifacts','',
        f'The audit verified all 1,288 raw chunk hashes, reconstructed {a["independent_feature_samples"]} real-data feature samples with a separate scalar implementation, '
        'reproduced all 120 checkpoint exports and 40 old daily baselines, checked train-only normalization and every information/decision/fill timestamp, '
        f'and independently replayed 66 accounts. Maximum NAV difference was ${a["largest_nav_difference_dollars"]:.8f}. '
        'Unit tests cover early closes, DST, sparse trading, future-bar perturbation, split-unit invariance, next-exchange-session alignment, inactive NaNs and exact original-core initialization.','',
        '[Protocol](PROTOCOL.json) · [Input review](INPUT_REVIEW.json) · [Audit](AUDIT.json) · [All metrics](metrics.csv) · [Daily paths](curves.csv) · '
        '[RankIC](ranking.csv) · [Per-seed forecast quality](prediction_quality.csv) · [Monthly returns](monthly.csv) · [Annual returns](annual.csv) · [Seed summary](seed_summary.csv) · '
        '[Primary paired intervals](ten_seed_intervals.csv) · [Original-five diagnostics](original_five_intervals.csv) · [New-five diagnostics](new_five_intervals.csv) · '
        '[Absolute Sharpe intervals](sharpe_intervals.csv) · [Coverage](coverage.csv) · [Decision clock](TIMING.csv) · [Independent feature samples](independent_feature_samples.csv)','',
        'Sources, feature snapshots, checkpoints and integrity locks remain under the versioned data/run directories. Previous reports and checkpoints are unchanged. '
        'No forward 2026 returns, trades or scheduled jobs were introduced.','']
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ['PROTOCOL.json','AUDIT.json','FREEZE.json','ENVIRONMENT.json','UNIT_TESTS.json','EVALUATION_UNIT_TESTS.json','EVALUATED.json',
        'metrics.csv','curves.csv','ranking.csv','ranking_periods.csv','prediction_quality.csv','annual.csv','monthly.csv','seed_summary.csv','sharpe_intervals.csv',
        'ten_seed_intervals.csv','original_five_intervals.csv','new_five_intervals.csv','TIMING.csv','independent_feature_samples.csv',
        'EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','PARENT_BASELINE_LOCK.json']:shutil.copyfile(RUN/name,DOCS/name)
    shutil.copyfile(FEATURES/'REVIEW.json',DOCS/'INPUT_REVIEW.json');shutil.copyfile(FEATURES/'coverage.csv',DOCS/'coverage.csv')
    for name in ['report_ndx_extended_hours.py','test_ndx_extended_hours_report.py']:shutil.copyfile(ROOT/name,RUN/'source_snapshot'/name)
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_research',fits=120,accounts=66,
        primary_cagrs={v:float(main.loc[v].annualized_return) for v in plan['views']},spy_cagr=float(main.loc['SPY'].annualized_return),
        qqq_cagr=float(main.loc['QQQ'].annualized_return),positive_primary_information_lowers=int((subset.simultaneous_low>0).sum()),
        future_reliability='not established by this historical experiment',report=str(DOCS/'REPORT.md'),
        artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    write(RUN/'PROGRESS.json',dict(stage='completed_audited_published',fits=120,accounts=66));print('Published',DOCS/'REPORT.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','audit','publish']);args=parser.parse_args();globals()[args.stage]()
