"""Evaluate, reproduce and publish frozen-core NASDAQ correction experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from ndx_corrections import prepared,partitions,normalized,Correction,core_model
from ndx_policy_engine import policy_backtest,policy_replay
from ndx_ranking_objectives import rank_ic
from pit_research_engine import fold,etf_backtest,historical_groups
from report_ndx_broad import qqq_adapter,md_table,pct
from report_ndx_ranking import joint_intervals
from research_engine import performance
from run_ndx_alpaca_training import ROOT,sha,write
from run_ndx_corrections import RUN,CELLS,complete,adapter_predict,validation_stage
from run_robust_research import ensemble_predictions,scale_inputs,predict,model_data

DOCS=ROOT/'docs/ndx_corrections_20261020'
ORIGINAL=ROOT/'runs/nasdaq100_historical_monthly_20261011'


def source_lock(name,files):
    value={str(Path(p).resolve()):sha(p) for p in files};path=RUN/name
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Locked evaluation source/reference changed')
    write(path,value)


def maps(data,plan,ranker=ensemble_predictions):
    base=data['base'];ids=np.concatenate([fold(base,y)[2] for y in plan['test_years']])
    choices=json.loads((RUN/'H1_SELECTION_LOCK.json').read_text())['choices']
    gates=json.loads((RUN/'VALIDATION_GATES.json').read_text())['gates']
    names=['primary','paired_core']+CELLS+['ungated__'+cell for cell in CELLS]
    pieces={name:[] for name in names};raw={name:{seed:[] for seed in plan['seeds']} for name in ['primary','paired_core']+CELLS}
    originals=[];reference_files=[]
    for choice,gate in zip(choices,gates):
        year=choice['year'];te=fold(base,year)[2];mask=base['mask'][te];by_name={name:[] for name in names}
        for seed in plan['seeds']:
            folder=RUN/str(year)/str(seed)
            with np.load(folder/'core_predictions.npz') as saved:core=saved['test'].copy()
            for name in names:
                if name=='paired_core':candidate=None
                elif name.startswith('ungated__'):candidate=choice['controls'][name[len('ungated__'):]]['candidate']
                else:candidate=gate['results'][name]['deployed_candidate']
                prediction=core
                if candidate is not None:
                    with np.load(folder/f"{candidate['cell']}_{candidate['epoch']}_predictions.npz") as saved:
                        prediction=core+candidate['shrink']*(saved['test']-core)
                by_name[name].append(prediction)
                if name in raw:raw[name][seed].append(prediction)
        for name in names:pieces[name].append(ranker(by_name[name],mask))
    for seed in plan['seeds']:
        available=[]
        for year in plan['test_years']:
            path=ORIGINAL/str(year)/f'think_mix_{seed}_predictions.npz';reference_files.append(path)
            with np.load(path) as saved:available.append(saved['test'].copy())
        originals.append(np.concatenate(available))
    scores={name:np.concatenate(value) for name,value in pieces.items()}
    raw={name:{seed:np.concatenate(value) for seed,value in seeds.items()} for name,seeds in raw.items()}
    reference=ranker(originals,base['mask'][ids])
    return ids,scores,raw,reference,reference_files


def evaluate():
    plan=complete()
    if (RUN/'EVALUATED.json').exists():raise ValueError('Completed correction evaluation is immutable')
    receipt=json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    if receipt['status']!='passed' or receipt['report_sha256']!=sha(__file__) or receipt['test_sha256']!=sha(ROOT/'test_ndx_corrections_report.py'):
        raise ValueError('Evaluation tests missing/stale')
    source_lock('EVALUATION_SOURCE_LOCK.json',[__file__,ROOT/'test_ndx_corrections_report.py',ROOT/'report_ndx_broad.py',ROOT/'report_ndx_ranking.py'])
    data=prepared();base=data['base'];ids,scores,raw,reference,files=maps(data,plan);source_lock('REFERENCE_LOCK.json',files)
    np.savez_compressed(RUN/'scores.npz',ids=ids,**scores,reference_original=reference)
    metrics,curves,ledgers,ranking,paths=[],[],[],[],{}
    def record(name,policy,seed,cost,m,p,l=None):
        tags=dict(model=name,policy=policy,seed=str(seed),cost_multiplier=cost)
        metrics.append(dict(**tags,**m));curves.append(p.assign(**tags))
        if str(seed)=='ensemble' and cost==1:
            paths[policy,name]=p.net.to_numpy()
            if l is not None:ledgers.append(l.assign(**tags))
    def account(name,prediction,seed,costs,source=base):
        for policy in plan['risk_policies']:
            exposure=np.ones(len(base['origin'])) if policy=='full' else data['trend_exposure']
            for cost in costs:
                m,p,l=policy_backtest(source,ids,prediction,10,cost,exposure=exposure);record(name,policy,seed,cost,m,p,l)
    for name,prediction in scores.items():
        account(name,prediction,'ensemble',plan['cost_multipliers'])
        ic=rank_ic(prediction,base['y'][ids],base['labelmask'][ids])
        for j,i in enumerate(ids):ranking.append(dict(model=name,signal_date=str(base['panel'].dates[base['origin'][i]].date()),raw_rankic=ic[j]))
        if name in raw:
            for seed,p in raw[name].items():account(name,p,seed,[1])
        print('Evaluated correction policy',name,flush=True)
    account('reference_original',reference,'ensemble',plan['cost_multipliers'])
    account('momentum126',base['momentum'][ids],'ensemble',plan['cost_multipliers'],model_data(base,'momentum126'))
    for cost in plan['cost_multipliers']:
        m,p,l=policy_backtest(base,ids,np.zeros_like(base['y'][ids]),10,cost,basket=True);record('monthly_basket','full','ensemble',cost,m,p,l)
        for ticker in ['SPY','QQQ']:
            m,p=etf_backtest(base,ticker,cost);record(ticker,'full','ensemble',cost,m,p)
        m,p,l=policy_backtest(qqq_adapter(base),ids,np.ones((len(ids),1)),1,cost,exposure=data['trend_exposure'])
        record('QQQ_timing','trend_half','ensemble',cost,m,p,l)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True)
    mf.to_csv(RUN/'metrics.csv',index=False);cf.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'primary_ledgers.csv',index=False)
    pd.DataFrame(ranking).groupby('model').raw_rankic.mean().to_csv(RUN/'ranking_summary.csv')
    annual,monthly,seeds=[],[],[]
    for key,p in cf.groupby(['model','policy','seed','cost_multiplier'],sort=False):
        tags=dict(zip(['model','policy','seed','cost_multiplier'],key));date=pd.to_datetime(p.date)
        for year,part in p.groupby(date.dt.year):annual.append(dict(**tags,year=int(year),**performance(part.net)))
        for month,part in p.groupby(date.dt.to_period('M').astype(str)):
            monthly.append(dict(**tags,month=month,net_return=float(np.prod(1+part.net)-1)))
    spy=mf[(mf.model=='SPY')&(mf.cost_multiplier==1)].annualized_return.iloc[0]
    for (name,policy),p in mf[mf.seed!='ensemble'].groupby(['model','policy']):
        seeds.append(dict(model=name,policy=policy,minimum_cagr=p.annualized_return.min(),median_cagr=p.annualized_return.median(),
                          maximum_cagr=p.annualized_return.max(),seeds_above_spy=int((p.annualized_return>spy).sum()),seed_runs=len(p)))
    af=pd.DataFrame(annual);mm=pd.DataFrame(monthly)
    af.to_csv(RUN/'annual.csv',index=False);mm.to_csv(RUN/'monthly.csv',index=False);pd.DataFrame(seeds).to_csv(RUN/'seed_summary.csv',index=False)
    contrasts=[]
    for policy in plan['risk_policies']:
        for name in ['SPY','QQQ','paired_core']+CELLS:
            contrasts.append((f'{policy}: primary minus {name}',(policy,'primary'),('full' if name in ['SPY','QQQ'] else policy,name)))
    ci=joint_intervals(paths,contrasts,[5,20,60],annualizer=252.).rename(columns={'left_view':'left_policy','right_view':'right_policy'})
    ci.to_csv(RUN/'paired_intervals.csv',index=False);screens=[]
    for policy in plan['risk_policies']:
        def pick(frame,name,cost=1):
            return frame[(frame.model==name)&(frame.policy==(policy if name=='primary' else 'full'))&(frame.seed=='ensemble')&(frame.cost_multiplier==cost)]
        m=pick(mf,'primary').iloc[0];sp=pick(mf,'SPY').iloc[0]
        a=pick(af,'primary').set_index('year').cumulative_return;b=pick(af,'SPY').set_index('year').cumulative_return
        months=pick(mm,'primary').set_index('month').net_return;spmonths=pick(mm,'SPY').set_index('month').net_return
        row=dict(policy=policy,base_cagr_above_spy=bool(m.annualized_return>sp.annualized_return),
                 double_cost_cagr_above_spy=bool(pick(mf,'primary',2).annualized_return.iloc[0]>pick(mf,'SPY',2).annualized_return.iloc[0]),
                 years_above_spy=int((a>b).sum()),months_above_spy=int((months>spmonths).sum()),drawdown_within_five_pp=bool(m.maximum_drawdown>=sp.maximum_drawdown-.05))
        row['historical_screen_passed']=bool(row['base_cagr_above_spy'] and row['double_cost_cagr_above_spy'] and row['years_above_spy']>=3 and row['months_above_spy']>=28 and row['drawdown_within_five_pp'])
        selected=ci[(ci.block_length==20)&(ci.left_policy==policy)]
        row['positive_simultaneous_contrasts']=int((selected.simultaneous_low>0).sum());row['comparisons']=len(selected);screens.append(row)
    pd.DataFrame(screens).to_csv(RUN/'historical_screen.csv',index=False)
    if len(metrics)!=144 or len(contrasts)!=14:raise ValueError('Unexpected fixed account/contrast inventory')
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',fits=100,checkpoints=260,accounts=144,contrasts=14,
          validation_locked_before_test_metrics=json.loads((RUN/'VALIDATION_GATES.json').read_text())['created_before_test_metrics']))


def independent_proxy(p,c,mask,observed,y):
    def weight(score):
        out=np.zeros_like(score,dtype=float)
        for j,m in enumerate(mask):
            ids=np.where(m)[0];cutoff=sorted(score[j,ids],reverse=True)[min(10,len(ids))-1]
            above=m&(score[j]>cutoff);equal=m&(score[j]==cutoff)
            out[j,above]=1.;out[j,equal]=(min(10,len(ids))-above.sum())/equal.sum();out[j]/=out[j].sum()
        return out
    a,b=weight(p),weight(c);valid=np.all((a==0)|observed,axis=1)&np.all((b==0)|observed,axis=1)
    def net(w):
        turns=[np.abs(w[0]).sum()]+[np.abs(w[j]-w[j-1]).sum() for j in range(1,len(w))]
        return np.sum(w*np.where(observed,y,0),axis=1)-.1*np.asarray(turns)
    delta=(net(a)-net(b))[valid]
    return dict(observations=int(valid.sum()),omitted=int((~valid).sum()),mean=float(np.mean(delta)) if len(delta) else None,
                variance=float(np.var(delta)) if len(delta) else None,wins=int((delta>0).sum()),score=float(np.mean(delta)-.1*np.var(delta)) if len(delta) else -1e30)


def audit():
    from audit_historical_ndx import independent_metrics
    from audit_historical_ndx_ranking import average_ranks
    from training_device import configure_device
    plan=complete()
    if not (RUN/'EVALUATED.json').exists():raise ValueError('Evaluation must precede audit')
    for name in ['EVALUATION_SOURCE_LOCK.json','REFERENCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Evaluation/reference changed')
    source_lock('AUDIT_SOURCE_LOCK.json',[__file__,ROOT/'audit_historical_ndx.py',ROOT/'audit_historical_ndx_ranking.py',ROOT/'ndx_policy_engine.py'])
    data=prepared();base=data['base'];device=configure_device('cuda');reproduced=0
    for year in plan['test_years']:
        tr,va,te,h1,h2=partitions(base,year);x,mean,std=scale_inputs(base,tr);norm=normalized(data,tr);groups=historical_groups(base,tr)
        for seed in plan['seeds']:
            folder=RUN/str(year)/str(seed)
            with np.load(folder/'preprocessing.npz') as saved:
                for name,value in dict(train=tr,validation=va,test=te,h1=h1,h2=h2,mean=mean,std=std).items():np.testing.assert_array_equal(saved[name],value)
                for view,values in norm.items():
                    np.testing.assert_array_equal(saved[view+'_mean'],values[1]);np.testing.assert_array_equal(saved[view+'_std'],values[2])
            checkpoint=torch.load(folder/f'think_mix_{seed}.pt',map_location='cpu',weights_only=False)
            model=core_model(len(base['panel'].tickers),groups).to(device);model.load_state_dict(checkpoint['state_dict'])
            allforecast=predict(model,x,base['mask'],np.arange(len(x)),8).astype('float32')
            with np.load(folder/'core_predictions.npz') as saved:
                np.testing.assert_allclose(allforecast,saved['all'],atol=1e-6,rtol=0)
                np.testing.assert_array_equal(saved['validation'],saved['all'][va]);np.testing.assert_array_equal(saved['test'],saved['all'][te])
            history=json.loads((folder/f'think_mix_{seed}_history.json').read_text());best=np.inf;selected=None
            for record in history:
                if record['validation_mse_pp2']<best-1e-7:best=record['validation_mse_pp2'];selected=record['epoch']
            if selected!=checkpoint['epoch']:raise ValueError('Core H1 stopping choice mismatch')
            np.testing.assert_allclose(((allforecast[h1]-base['y'][h1]*100)**2)[base['labelmask'][h1]].mean(),best,atol=1e-6,rtol=1e-6)
            reproduced+=1
            for cell in CELLS:
                view,_=cell.split('__');features=norm[view][0]
                for epoch in plan['adapter']['checkpoint_epochs']:
                    cp=torch.load(folder/f'{cell}_{epoch}.pt',map_location='cpu',weights_only=False)
                    if cp['core_sha256']!=sha(folder/f'think_mix_{seed}.pt') or cp['epoch']!=epoch:raise ValueError('Adapter frozen-core link mismatch')
                    adapter=Correction(cp['features'],cp['hidden']).to(device);adapter.load_state_dict(cp['state_dict'])
                    with np.load(folder/f'{cell}_{epoch}_predictions.npz') as saved:
                        for name,sub in [('validation',va),('test',te)]:
                            forecast=adapter_predict(adapter,features,allforecast,base['mask'],sub,device)
                            np.testing.assert_allclose(forecast,saved[name],atol=1e-6,rtol=0)
                            delta=forecast-allforecast[sub];active=base['mask'][sub]
                            np.testing.assert_allclose(np.where(active,delta,0).sum(1),0.,atol=1e-5,rtol=0)
                            scale=np.array([max(row[m].std(),1e-4) for row,m in zip(allforecast[sub],active)])
                            if np.any(np.abs(delta[active])>np.broadcast_to(scale[:,None],delta.shape)[active]+1e-6):raise ValueError('Correction bound exceeded')
                    reproduced+=1
        print(f'Reproduced {year}: {reproduced}/260 checkpoints',flush=True)
    # Independent rank implementation plus independent validation proxy.
    def ranker(available,mask):
        return np.mean([ranked(p,mask) for p in available],axis=0)
    def ranked(p,mask):
        out=np.zeros_like(p,dtype=float)
        for j,m in enumerate(mask):out[j,m]=average_ranks(p[j,m])/m.sum()
        return out
    from run_ndx_corrections import validation_predictions
    choices=json.loads((RUN/'H1_SELECTION_LOCK.json').read_text())
    gates=json.loads((RUN/'VALIDATION_GATES.json').read_text())
    if not choices['created_before_h2'] or not choices['created_before_test_metrics'] or not gates['created_before_test_metrics']:raise ValueError('Validation lock ordering failed')
    for choice,gate in zip(choices['choices'],gates['gates']):
        _,va,_,h1,h2=partitions(base,choice['year']);cores,adapters=validation_predictions(choice['year'],plan);mask=base['mask'][va]
        baseline=ranker(cores,mask);statistics=[]
        def forecast(candidate):
            return baseline if candidate is None else ranker([c+candidate['shrink']*(a-c) for c,a in zip(cores,adapters[candidate['cell'],candidate['epoch']])],mask)
        local=np.searchsorted(va,h1)
        for row in choice['candidates']:
            statistic=independent_proxy(forecast(row['candidate'])[local],baseline[local],mask[local],base['labelmask'][h1],base['y'][h1]*100)
            for name in statistic:
                if statistic[name] is None:
                    if row['statistics'][name] is not None:raise ValueError('Missing proxy mismatch')
                else:np.testing.assert_allclose(statistic[name],row['statistics'][name],atol=1e-10,rtol=0)
            statistics.append(dict(candidate=row['candidate'],statistics=statistic))
        def selected(rows):
            best=dict(candidate=None,statistics=dict(score=0.))
            for row in rows:
                if row['statistics']['score']>best['statistics']['score']+1e-7:best=row
            return best['candidate']
        for name in ['primary']+CELLS:
            candidates=statistics if name=='primary' else [r for r in statistics if r['candidate']['cell']==name]
            chosen=selected(candidates);result=gate['results'][name]
            if chosen!=result['h1_candidate']:raise ValueError('Independent H1 selection mismatch')
            local=np.searchsorted(va,h2);statistic=independent_proxy(forecast(chosen)[local],baseline[local],mask[local],base['labelmask'][h2],base['y'][h2]*100)
            for key in statistic:
                np.testing.assert_allclose(statistic[key],result['h2_statistics'][key],atol=1e-10,rtol=0)
            accept=chosen is not None and statistic['observations']>=12 and statistic['mean']>0 and statistic['wins']*2>=statistic['observations']
            if bool(accept)!=result['accepted'] or result['deployed_candidate']!=(chosen if accept else None):raise ValueError('Independent H2 gate mismatch')
    ids,scores,raw,reference,_=maps(data,plan,ranker)
    with np.load(RUN/'scores.npz') as saved:
        for name,p in dict(**scores,reference_original=reference).items():np.testing.assert_allclose(p,saved[name],atol=1e-12,rtol=0)
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}).set_index(['model','policy','seed','cost_multiplier'])
    curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str});accounts=0;largest=0.
    for key,path in curves.groupby(['model','policy','seed','cost_multiplier'],sort=False):
        name,policy,seed,cost=key
        if len(path)!=1003 or not path.date.is_unique or not path.date.is_monotonic_increasing:raise ValueError('Account calendar mismatch')
        for field,value in independent_metrics(path.net).items():np.testing.assert_allclose(value,metrics.loc[key,field],atol=1e-10,rtol=1e-10)
        exposure=np.ones(len(base['origin'])) if policy=='full' else data['trend_exposure'];source=base
        if name in ['SPY','QQQ']:
            root=Path(base['manifest']['root']);first=base['entry'][ids[0]]
            price=pd.read_csv(root/'etf_adj_close.csv')[name].to_numpy();volume=pd.read_csv(root/'etf_dollar_volume.csv')[name].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/volume[first-20:first].mean()/.01));nav=1e6/(1+rate)/price[first]*price[first:]
            nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/volume[-21:-1].mean()/.01))
        else:
            if name=='QQQ_timing':source=qqq_adapter(base);prediction=np.ones((len(ids),1))
            elif name=='monthly_basket':prediction=np.zeros_like(base['y'][ids])
            elif name=='momentum126':source=model_data(base,'momentum126');prediction=base['momentum'][ids]
            elif name=='reference_original':prediction=reference
            elif seed!='ensemble':prediction=raw[name][int(seed)]
            else:prediction=scores[name]
            nav=policy_replay(source,ids,prediction,1 if name=='QQQ_timing' else 10,cost,basket=name=='monthly_basket',exposure=exposure)
        actual=1e6*np.cumprod(1+path.net.to_numpy());largest=max(largest,float(np.abs(actual-nav).max()));np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10)
        accounts+=1
        if accounts%30==0:print(f'Replayed {accounts}/144 accounts independently',flush=True)
    ledger=pd.read_csv(RUN/'primary_ledgers.csv');np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    if accounts!=144 or reproduced!=260 or ledger.cash.min()<-1e-5:raise ValueError('Audit inventory/cash failure')
    write(RUN/'AUDIT.json',dict(status='passed',checkpoints_reproduced=reproduced,fits=100,independent_accounts=accounts,
          largest_nav_difference_dollars=largest,independent_h1_choices=20,independent_h2_gates=20,
          future_confirmation='not performed; previously inspected historical development'))


def label(name):
    return dict(primary='Validation-gated correction (primary)',paired_core='Paired H1-stopped core',reference_original='Original full-validation THINK+mixer').get(name,name.replace('__',' / '))


def publish():
    complete();audit_record=json.loads((RUN/'AUDIT.json').read_text())
    if audit_record['status']!='passed':raise ValueError('Independent audit must pass')
    for name in ['EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Publication source changed')
    DOCS.mkdir(parents=True,exist_ok=True);figures=DOCS/'figures';figures.mkdir(exist_ok=True)
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str});curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    main=metrics[(metrics.seed=='ensemble')&(metrics.cost_multiplier==1)].set_index(['model','policy'])
    screen=pd.read_csv(RUN/'historical_screen.csv');seeds=pd.read_csv(RUN/'seed_summary.csv');ci=pd.read_csv(RUN/'paired_intervals.csv');ci=ci[ci.block_length==20]
    gates=json.loads((RUN/'VALIDATION_GATES.json').read_text())['gates'];gate_rows=[]
    for row in gates:
        for name,result in row['results'].items():
            c=result['h1_candidate'];h2=result['h2_statistics']
            gate_rows.append([row['year'],label(name),'zero / core' if c is None else f"{c['cell']} / epoch {c['epoch']} / shrink {c['shrink']}",
                              h2['observations'],h2['omitted'],f"{h2['mean']:.3f} pp",f"{h2['wins']}/{h2['observations']}",str(result['accepted'])])
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for policy in ['full','trend_half']:
        fig,ax=plt.subplots(figsize=(12,6))
        for name,color,style in [('primary','black','-'),('paired_core','#1f77b4',':'),('reference_original','#9467bd','-.'),('SPY','#cc0000','--'),('QQQ','#333399','--')]:
            actual='full' if name in ['SPY','QQQ'] else policy
            p=curves[(curves.model==name)&(curves.policy==actual)&(curves.seed=='ensemble')&(curves.cost_multiplier==1)]
            ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),color=color,linestyle=style,label=f'{label(name)}: {pct(main.loc[name,actual].annualized_return)} CAGR')
        ax.set(title=f'Exploratory 2022–2025: {policy} exposure',ylabel='Net wealth per initial $1');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/(policy+'_wealth.png'),dpi=160);plt.close(fig)
    selected=ci[ci.right_model.isin(['SPY','QQQ','paired_core'])];fig,ax=plt.subplots(figsize=(12,6))
    for j,r in enumerate(selected.itertuples(index=False)):
        ax.plot(np.array([r.simultaneous_low,r.simultaneous_high])*100,[j,j],color='#ccc',linewidth=6)
        ax.plot(np.array([r.paired_low,r.paired_high])*100,[j,j],color='#1f77b4',linewidth=2);ax.scatter(r.estimate*100,j,color='black')
    ax.set(yticks=range(len(selected)),yticklabels=selected.contrast,xlabel='Annual arithmetic active mean (percentage points)',title='Paired 95% intervals and 14-contrast simultaneous diagnostics');ax.axvline(0,color='black');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(figures/'intervals.png',dpi=160);plt.close(fig)
    performance_rows=[]
    for (name,policy),r in main.iterrows():performance_rows.append([label(name),policy,pct(r.annualized_return),f'{r.sharpe:.3f}',pct(r.maximum_drawdown)])
    screen_rows=[[r.policy,str(r.historical_screen_passed),f'{r.years_above_spy}/4',f'{r.months_above_spy}/48',str(r.drawdown_within_five_pp),f'{r.positive_simultaneous_contrasts}/{r.comparisons}'] for r in screen.itertuples(index=False)]
    seed_rows=[[label(r.model),r.policy,pct(r.minimum_cagr),pct(r.median_cagr),pct(r.maximum_cagr),f'{r.seeds_above_spy}/{r.seed_runs}'] for r in seeds.itertuples(index=False)]
    ci_rows=[[r.contrast,f'{100*r.estimate:.2f} pp',f'[{100*r.paired_low:.2f}, {100*r.paired_high:.2f}]',f'[{100*r.simultaneous_low:.2f}, {100*r.simultaneous_high:.2f}]'] for r in ci.itertuples(index=False)]
    cost_rows=[]
    for (name,policy),p in metrics[(metrics.seed=='ensemble')&metrics.model.isin(['primary','paired_core','reference_original','SPY','QQQ'])].groupby(['model','policy']):
        p=p.set_index('cost_multiplier');cost_rows.append([label(name),policy]+[pct(p.loc[c,'annualized_return']) for c in [1,2,4]])
    full=main.loc['primary','full'];core=main.loc['paired_core','full'];spy=main.loc['SPY','full'];qqq=main.loc['QQQ','full']
    passes=int(screen.historical_screen_passed.sum());accepted=sum(r['results']['primary']['accepted'] for r in gates)
    lines=['# Frozen-core NASDAQ-100 correction study — October 20, 2026','',
        f'The primary full-exposure policy returned **{pct(full.annualized_return)} CAGR**, versus **{pct(core.annualized_return)}** for its paired core, **{pct(spy.annualized_return)}** for SPY and **{pct(qqq.annualized_return)}** for QQQ. '
        f'The second-half validation gate retained a primary correction in {accepted}/4 years. {passes}/2 risk policies passed the predeclared historical reliability screen. These are development results; reliable future superiority is not established.','',
        'All 100 fits, 260 saved checkpoints and 144 continuous accounts are disclosed. Previously inspected 2022–2025 dates remain exploratory. No original checkpoint, prior negative experiment or input dataset was changed.','',
        '## Selection and acceptance','',
        'Twenty five-feature THINK+mixer cores (five seeds, four annual folds) use the original training origins and train-only graphs/normalization. '
        'Each core selects its earliest best MSE checkpoint from January–June of the prior validation year, with H1 labels purged before the first H2 signal. '
        'The core is frozen. Eighty shared per-stock adapters add either 128 intraday-window inputs (16 sessions × eight reviewed summaries) or 65 causal rich features. '
        'Every feature normalizer uses only eligible training inputs. Each adapter has 16 hidden units, a zero-initialized final layer, no ticker parameters and a centered tanh correction scaled by half the signal-time core-score standard deviation. '
        'Zero correction reproduces the paired core. Absolute changes are bounded by one core-score standard deviation.','',
        'The two adapter losses are a local LambdaRank-inspired weighted pairwise objective and a softmax SPY-active-return utility with concentration, variance and turnover penalties. '
        'Utility returns require complete eligible labels; missing outcomes are never synthesized. The losses are training surrogates, not portfolio accounts. '
        '[Microsoft’s LambdaRank overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) motivates ranking-weight changes; this implementation is not a reproduction or an investment-performance claim.','',
        'After all fits, H1 chooses among epochs 5/10/20, shrink factors 0.25/0.5/1 and four feature/loss cells. Zero is always available. '
        'The H1 score is mean paired top-ten return improvement less 0.1 times its variance, after a 0.1-percentage-point L1 target-turnover proxy. '
        'Fractional boundary ties match the final account engine. Origins with missing labels in either selected basket are omitted and counted. '
        'Every annual H1 selection was locked before H2 scoring. H2 can only accept the chosen correction or fall back to the core: at least 12 comparable origins, positive mean paired proxy improvement and wins on at least half those origins. '
        'H2 rejection never triggers a search for another primary candidate. Four adapter-specific gated controls follow the same prespecified rules. '
        'Ungated rows disclose each cell’s H1 choice before H2 rejection; they cannot replace the primary using test results. The proxy omits price drift, settlement and ADV caps, so the final account is the relevant financial measure.','',
        md_table(['Test year','Policy','H1 choice','H2 comparable','Omitted','Mean proxy change','Positive origins','Accepted'],gate_rows),'',
        '## Continuous accounts','',md_table(['Model','Exposure policy','Net CAGR','Sharpe','Maximum drawdown'],performance_rows),'',
        'Historical monthly Nasdaq-100 membership, issuer-aware reviewed Alpaca summaries, next-close frozen orders, $1 million starting cash, 1% ADV order caps and the original modeled fees are retained. '
        'Signals and holding periods remain five sessions. The input uses 15-minute bars aggregated into causal summaries; this is not 15-minute trading. '
        'Full exposure and the fixed half-exposure-below-QQQ-MA200 rule are both disclosed. Cash earns no interest; Sharpe retains the fixed 3% hurdle. '
        'SPY/QQQ are net buy-and-hold references. Original full-validation THINK+mixer is a descriptive historical reference, distinct from the newly H1-stopped core.','',
        '![Full exposure](figures/full_wealth.png)','', '![Trend exposure](figures/trend_half_wealth.png)','',
        md_table(['Primary risk policy','Historical screen passed','Years above SPY','Months above SPY','Drawdown within 5 pp','Positive simultaneous comparisons'],screen_rows),'',
        'The historical screen requires CAGR above SPY at base and double fees, at least 3/4 years and 28/48 months above SPY, and drawdown at most five percentage points worse. '
        'This is a development screen, not a confidence guarantee. Confidence comparisons with competing models are separate and shown below.','',
        md_table(['Model','Policy','1× fee CAGR','2× fee CAGR','4× fee CAGR'],cost_rows),'',
        '## Seed sensitivity and uncertainty','',md_table(['Model','Policy','Minimum seed CAGR','Median seed CAGR','Maximum seed CAGR','Seeds above SPY'],seed_rows),'',
        'Matched-seed paths hold the annual five-seed validation choices and gates fixed, then use each seed’s forecast. They diagnose initialization sensitivity, not five independent market histories or independently selected policies.','',
        '![Intervals](figures/intervals.png)','',md_table(['Primary comparison','Active mean/year','Paired 95% interval, pp','14-contrast simultaneous diagnostic, pp'],ci_rows),'',
        'Ten thousand paired circular-block samples resample whole daily paths together. The primary block is 20 sessions; 5/60-session sensitivities are included. '
        'One family covers both primary risk policies against SPY, QQQ, the paired core and four gated adapters (14 contrasts). Original full-validation reference and ungated diagnostic cells are outside this family. '
        'Annual arithmetic active means differ from CAGR gaps. These intervals condition on fitted paths: they omit full research-search, training, vendor-revision and regime uncertainty. '
        'More bars within the same market period do not supply independent years or justify artificially narrower intervals.','',
        '## Verification and limits','',
        f'Independent checks reproduced all {audit_record["checkpoints_reproduced"]} checkpoints’ validation/test exports, every H1 choice and H2 gate, and replayed {audit_record["independent_accounts"]} continuous accounts. '
        f'Maximum NAV discrepancy was ${audit_record["largest_nav_difference_dollars"]:.8f}. Frozen hashes cover source, data, preprocessing, core/adapter links and exports. '
        'Unit fixtures cover exact zero corrections, inactive NaNs, permutation symmetry, frozen-core gradients, missing-label utility, boundary ties, train-only normalization and chronological purges.','',
        'Acquisition/total-return approximations, missing labels, short effective history and repeated historical inspection still limit inference. '
        'The reserved future dates from October 21, 2026 have not occurred in this study. Updated point-in-time data, membership/issuer support and a fixed model/refit policy must be reviewed before any prospective run. '
        'No trades or scheduled jobs were created. A higher historical CAGR cannot be promoted as reliable future outperformance.','',
        '## Artifacts','',
        '[Protocol](PROTOCOL.json) · [Independent audit](AUDIT.json) · [All metrics](metrics.csv) · [Daily curves](curves.csv) · '
        '[H1 selections](H1_SELECTION_LOCK.json) · [H2 acceptance gates](VALIDATION_GATES.json) · [Historical screens](historical_screen.csv) · '
        '[Annual returns](annual.csv) · [Monthly returns](monthly.csv) · [Seed results](seed_summary.csv) · [All intervals](paired_intervals.csv) · [Ranking](ranking_summary.csv)','',
        'Training objects, every adapter checkpoint, normalization and all prediction exports remain under `runs/ndx_corrections_20261020`. '
        'The complete prior [Alpaca study](../nasdaq_alpaca_20261019/REPORT.md), [ranking study](../ndx_ranking_20261020/REPORT.md) and [broad feature study](../ndx_broad_20261020/REPORT.md) remain available.','']
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    names=['PROTOCOL.json','AUDIT.json','FREEZE.json','ENVIRONMENT.json','UNIT_TESTS.json','EVALUATION_UNIT_TESTS.json','EVALUATED.json',
           'metrics.csv','curves.csv','annual.csv','monthly.csv','seed_summary.csv','historical_screen.csv','paired_intervals.csv','ranking_summary.csv',
           'H1_SELECTION_LOCK.json','VALIDATION_GATES.json','EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','REFERENCE_LOCK.json']
    for name in names:shutil.copyfile(RUN/name,DOCS/name)
    for name in ['report_ndx_corrections.py','test_ndx_corrections_report.py']:shutil.copyfile(ROOT/name,RUN/'source_snapshot'/name)
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_research',fits=100,checkpoints=260,accounts=144,
          primary_full_cagr=float(full.annualized_return),paired_core_cagr=float(core.annualized_return),spy_cagr=float(spy.annualized_return),
          primary_h2_acceptances=accepted,historical_screens=screen.to_dict('records'),future_reliability='not established',
          report=str(DOCS/'REPORT.md'),artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    write(RUN/'PROGRESS.json',dict(stage='completed_audited_published',fits=100,checkpoints=260,accounts=144))
    print('Published:',DOCS/'REPORT.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','audit','publish']);args=parser.parse_args();globals()[args.stage]()
