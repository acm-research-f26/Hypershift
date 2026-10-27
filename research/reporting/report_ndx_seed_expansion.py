"""Seed-count research with broad historical controls and two uncertainty views."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from ndx_policy_engine import policy_backtest,policy_replay
from pit_research_engine import load_data,fold,historical_groups,etf_backtest
from report_ndx_broad import md_table,pct
from report_ndx_ranking import joint_intervals
from research_engine import performance
from robust_models import make_model
from run_ndx_alpaca_training import ROOT,sha,write
from run_ndx_seed_expansion import RUN,complete,bootstrap_draws
from run_robust_research import ensemble_predictions,scale_inputs,predict

DOCS=ROOT/'docs/ndx_seed_expansion_20261020'
ORIGINAL=ROOT/'runs/nasdaq100_historical_monthly_20261011'


def trend_exposure(base):
    q=pd.read_csv(Path(base['manifest']['root'])/'etf_adj_close.csv')['QQQ']
    return np.where(q.to_numpy()>=q.rolling(200).mean().to_numpy(),1.,.5)[base['origin']]


def lock(name,files):
    value={str(Path(p).resolve()):sha(p) for p in files};path=RUN/name
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Seed evaluation lock changed')
    write(path,value)


def forecasts(base,plan,ranker=ensemble_predictions):
    seeds=plan['original_seeds']+plan['new_seeds'];ids=np.concatenate([fold(base,y)[2] for y in plan['test_years']]);raw=[];original_files=[]
    for seed in seeds:
        parts=[]
        for year in plan['test_years']:
            path=ORIGINAL/str(year)/f'think_mix_{seed}_predictions.npz' if seed in plan['original_seeds'] else RUN/str(year)/f'{seed}_predictions.npz'
            if seed in plan['original_seeds']:original_files.append(path)
            with np.load(path) as saved:parts.append(saved['test'].copy())
        raw.append(np.concatenate(parts))
    ranks=np.stack([ranker([p],base['mask'][ids]) for p in raw]);draws=bootstrap_draws(plan)
    scores=dict(all_twenty_five=ranks.mean(0),original_five=ranks[:5].mean(0),new_twenty=ranks[5:].mean(0))
    boot=np.stack([ranks[draw].mean(0) for draw in draws])
    return ids,scores,np.asarray(raw),boot,original_files


def references(calendar):
    files=[('original',ROOT/'docs/nasdaq_monthly_pit/primary_curves.csv'),
           ('alpaca',ROOT/'runs/nasdaq100_alpaca_20261019/curves.csv'),
           ('ranking',ROOT/'runs/ndx_ranking_20261020/curves.csv'),
           ('broad',ROOT/'runs/ndx_broad_20261020/curves.csv'),
           ('correction',ROOT/'runs/ndx_corrections_20261020/curves.csv')]
    available={};metadata=[]
    for study,path in files:
        if study!='original':
            publication=json.loads((path.parent/'FINAL_PUBLICATION.json').read_text())
            if publication['artifact_hashes']['curves.csv']!=sha(path):raise ValueError('Prior published curve hash mismatch: '+study)
        frame=pd.read_csv(path,dtype={'seed':str})
        selected=(frame.seed=='ensemble')&(frame.cost_multiplier==1)
        if 'k' in frame:selected&=frame.k==10
        frame=frame[selected&~frame.model.isin(['SPY','QQQ'])]
        group=[c for c in ['view','model','policy'] if c in frame]
        for key,part in frame.groupby(group,sort=False):
            if not isinstance(key,tuple):key=(key,)
            name=study+' / '+' / '.join(str(k) for k in key)
            if not np.array_equal(part.date.to_numpy(),calendar):raise ValueError('Prior account dates mismatch: '+name)
            available['reference',name]=part.net.to_numpy()
            metadata.append(dict(reference=name,study=study,source=str(path.relative_to(ROOT)),**performance(part.net)))
    return available,pd.DataFrame(metadata),[p for _,p in files]


def seed_time_intervals(boot_paths,paths,blocks=[5,20,60],repeats=10000):
    rows=[]
    for policy in ['full','trend_half']:
        for reference in ['SPY','QQQ','original_five']:
            control=paths['full' if reference in ['SPY','QQQ'] else policy,reference]
            active=boot_paths[policy]-control[None,:];n=active.shape[1]
            estimate=np.mean(paths[policy,'all_twenty_five']-control)*252
            for block in blocks:
                rng=np.random.default_rng(20261020+block);samples=[]
                for _ in range(0,repeats,100):
                    chosen=rng.integers(0,len(active),100)
                    starts=rng.integers(0,n,(100,int(np.ceil(n/block))))
                    dates=((starts[...,None]+np.arange(block))%n).reshape(100,-1)[:,:n]
                    samples.extend(active[chosen[:,None],dates].mean(1)*252)
                rows.append(dict(policy=policy,reference=reference,block_sessions=block,replicates=repeats,seed_draw_accounts=len(active),
                                 estimate=estimate,pointwise_low=float(np.quantile(samples,.025)),pointwise_high=float(np.quantile(samples,.975)),
                                 interpretation='Finite seed-account plus paired time diagnostic; fixed checkpoints, reference and prior choices. Not a full-search interval or future guarantee.'))
    return pd.DataFrame(rows)


def evaluate():
    plan=complete()
    if (RUN/'EVALUATED.json').exists():raise ValueError('Completed seed evaluation is immutable')
    receipt=json.loads((RUN/'EVALUATION_UNIT_TESTS.json').read_text())
    if receipt['status']!='passed' or receipt['report_sha256']!=sha(__file__) or receipt['test_sha256']!=sha(ROOT/'test_ndx_seed_report.py'):raise ValueError('Seed evaluation preflight stale')
    lock('EVALUATION_SOURCE_LOCK.json',[__file__,ROOT/'test_ndx_seed_report.py',ROOT/'report_ndx_broad.py',ROOT/'report_ndx_ranking.py'])
    base=load_data();trend=trend_exposure(base);ids,scores,raw,boot,old_files=forecasts(base,plan)
    np.savez_compressed(RUN/'scores.npz',ids=ids,raw=raw,bootstrap=boot,draws=bootstrap_draws(plan),**scores)
    metrics,curves,ledgers,paths,boot_paths=[],[],[],{},dict(full=[],trend_half=[])
    def record(name,policy,seed,cost,m,p,l=None):
        tags=dict(model=name,policy=policy,seed=str(seed),cost_multiplier=cost)
        metrics.append(dict(**tags,**m));curves.append(p.assign(**tags))
        if str(seed)=='ensemble' and cost==1:
            paths[policy,name]=p.net.to_numpy()
            if l is not None:ledgers.append(l.assign(**tags))
    def account(name,prediction,seed,costs):
        for policy in plan['risk_policies']:
            exposure=np.ones(len(base['origin'])) if policy=='full' else trend
            for cost in costs:
                m,p,l=policy_backtest(base,ids,prediction,10,cost,exposure=exposure);record(name,policy,seed,cost,m,p,l)
                if name=='seed_bootstrap':boot_paths[policy].append(p.net.to_numpy())
    for name,prediction in scores.items():account(name,prediction,'ensemble',plan['cost_multipliers'])
    for seed,prediction in zip(plan['original_seeds']+plan['new_seeds'],raw):account('individual_seed',prediction,seed,[1])
    for j,prediction in enumerate(boot):
        account('seed_bootstrap',prediction,j,[1])
        if (j+1)%16==0:print('Accounted seed-bootstrap portfolios',j+1,'/128',flush=True)
    for cost in plan['cost_multipliers']:
        for ticker in ['SPY','QQQ']:
            m,p=etf_backtest(base,ticker,cost);record(ticker,'full','ensemble',cost,m,p)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True);mf.to_csv(RUN/'metrics.csv',index=False);cf.to_csv(RUN/'curves.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(RUN/'primary_ledgers.csv',index=False)
    calendar=cf[(cf.model=='SPY')&(cf.cost_multiplier==1)].date.to_numpy()
    prior,metadata,files=references(calendar)
    np.testing.assert_allclose(paths['full','original_five'],prior['reference','original / think_mix'],atol=1e-12,rtol=0)
    paths.update(prior);metadata.to_csv(RUN/'references.csv',index=False);lock('REFERENCE_LOCK.json',files+old_files)
    # These source curves have passed their own checkpoint/account audits.
    # Compare against every eligible ensemble path, never select a test winner.
    contrasts=[]
    for policy in plan['risk_policies']:
        for name in ['SPY','QQQ','original_five','new_twenty']:
            contrasts.append((f'{policy}: primary minus {name}',(policy,'all_twenty_five'),('full' if name in ['SPY','QQQ'] else policy,name)))
        for key in prior:contrasts.append((f'{policy}: primary minus {key[1]}',(policy,'all_twenty_five'),key))
    ci=joint_intervals(paths,contrasts,[5,20,60],annualizer=252.).rename(columns={'left_view':'left_policy','right_view':'right_policy'})
    ci.to_csv(RUN/'paired_intervals.csv',index=False)
    boot_paths={k:np.asarray(v) for k,v in boot_paths.items()}
    seed_time_intervals(boot_paths,paths).to_csv(RUN/'seed_time_intervals.csv',index=False)
    annual,monthly=[],[]
    for key,p in cf[cf.seed=='ensemble'].groupby(['model','policy','cost_multiplier'],sort=False):
        tags=dict(zip(['model','policy','cost_multiplier'],key));date=pd.to_datetime(p.date)
        for year,part in p.groupby(date.dt.year):annual.append(dict(**tags,year=int(year),**performance(part.net)))
        for month,part in p.groupby(date.dt.to_period('M').astype(str)):monthly.append(dict(**tags,month=month,net_return=float(np.prod(1+part.net)-1)))
    af=pd.DataFrame(annual);mm=pd.DataFrame(monthly);af.to_csv(RUN/'annual.csv',index=False);mm.to_csv(RUN/'monthly.csv',index=False)
    screens=[]
    for policy in plan['risk_policies']:
        def select(frame,name,cost=1):return frame[(frame.model==name)&(frame.policy==(policy if name=='all_twenty_five' else 'full'))&(frame.cost_multiplier==cost)]
        m=select(mf,'all_twenty_five').iloc[0];sp=select(mf,'SPY').iloc[0]
        a=select(af,'all_twenty_five').set_index('year').cumulative_return;b=select(af,'SPY').set_index('year').cumulative_return
        mo=select(mm,'all_twenty_five').set_index('month').net_return;so=select(mm,'SPY').set_index('month').net_return
        row=dict(policy=policy,base_cagr_above_spy=bool(m.annualized_return>sp.annualized_return),
                 double_cost_cagr_above_spy=bool(select(mf,'all_twenty_five',2).annualized_return.iloc[0]>select(mf,'SPY',2).annualized_return.iloc[0]),
                 years_above_spy=int((a>b).sum()),months_above_spy=int((mo>so).sum()),drawdown_within_five_pp=bool(m.maximum_drawdown>=sp.maximum_drawdown-.05))
        row['historical_screen_passed']=bool(row['base_cagr_above_spy'] and row['double_cost_cagr_above_spy'] and row['years_above_spy']>=3 and row['months_above_spy']>=28 and row['drawdown_within_five_pp'])
        family=ci[(ci.block_length==20)&(ci.left_policy==policy)];row['positive_simultaneous_contrasts']=int((family.simultaneous_low>0).sum());row['comparisons']=len(family)
        screens.append(row)
    pd.DataFrame(screens).to_csv(RUN/'historical_screen.csv',index=False)
    if len(metrics)!=330:raise ValueError('Seed account inventory mismatch')
    write(RUN/'EVALUATED.json',dict(status='complete_pending_audit',new_fits=80,total_seeds=25,accounts=330,
           bootstrap_accounts=256,reference_paths=len(prior),family_comparisons=len(contrasts),history='exploratory'))


def audit():
    from audit_historical_ndx import independent_metrics
    from audit_historical_ndx_ranking import average_ranks
    from training_device import configure_device
    plan=complete();base=load_data();device=configure_device('cuda');reproduced=0
    for name in ['EVALUATION_SOURCE_LOCK.json','REFERENCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Seed evaluation source/reference changed')
    lock('AUDIT_SOURCE_LOCK.json',[__file__,ROOT/'audit_historical_ndx.py',ROOT/'audit_historical_ndx_ranking.py',ROOT/'ndx_policy_engine.py'])
    for year in plan['test_years']:
        tr,va,te=fold(base,year);x,mean,std=scale_inputs(base,tr);groups=historical_groups(base,tr)
        for seed in plan['new_seeds']:
            folder=RUN/str(year)
            with np.load(folder/f'{seed}_preprocessing.npz') as saved:
                for name,value in dict(train=tr,validation=va,test=te,mean=mean,std=std).items():np.testing.assert_array_equal(saved[name],value)
            cp=torch.load(folder/f'think_mix_{seed}.pt',map_location='cpu',weights_only=False);model=make_model('think_mix',len(base['panel'].tickers),groups,causal_membership=True).to(device);model.load_state_dict(cp['state_dict'])
            vp,tp=[predict(model,x,base['mask'],ids,8) for ids in [va,te]]
            with np.load(folder/f'{seed}_predictions.npz') as saved:
                np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0);np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
            history=json.loads((folder/f'think_mix_{seed}_history.json').read_text());best=np.inf;chosen=None
            for row in history:
                if row['validation_mse_pp2']<best-1e-7:best=row['validation_mse_pp2'];chosen=row['epoch']
            if chosen!=cp['epoch']:raise ValueError('Seed checkpoint selection mismatch')
            np.testing.assert_allclose(((vp-base['y'][va]*100)**2)[base['labelmask'][va]].mean(),best,atol=1e-6,rtol=1e-6);reproduced+=1
        print('Reproduced seed expansion',year,reproduced,'/80',flush=True)
    def ranker(available,mask):
        ranked=[]
        for p in available:
            out=np.zeros_like(p,dtype=float)
            for i,m in enumerate(mask):out[i,m]=average_ranks(p[i,m])/m.sum()
            ranked.append(out)
        return np.mean(ranked,axis=0)
    ids,scores,raw,boot,_=forecasts(base,plan,ranker)
    with np.load(RUN/'scores.npz') as saved:
        for name,p in dict(**scores,raw=raw,bootstrap=boot,draws=bootstrap_draws(plan)).items():np.testing.assert_allclose(saved[name],p,atol=1e-12,rtol=0)
    trend=trend_exposure(base);mf=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str}).set_index(['model','policy','seed','cost_multiplier'])
    cf=pd.read_csv(RUN/'curves.csv',dtype={'seed':str});largest=0.;accounts=0
    seed_position={seed:j for j,seed in enumerate(plan['original_seeds']+plan['new_seeds'])}
    for key,path in cf.groupby(['model','policy','seed','cost_multiplier'],sort=False):
        name,policy,seed,cost=key
        if len(path)!=1003 or not path.date.is_unique or not path.date.is_monotonic_increasing:raise ValueError('Seed calendar mismatch')
        for field,value in independent_metrics(path.net).items():np.testing.assert_allclose(value,mf.loc[key,field],atol=1e-10,rtol=1e-10)
        if name in ['SPY','QQQ']:
            root=Path(base['manifest']['root']);first=base['entry'][ids[0]];q=pd.read_csv(root/'etf_adj_close.csv')[name].to_numpy();v=pd.read_csv(root/'etf_dollar_volume.csv')[name].to_numpy()
            rate=cost*(.0007+.0003*np.sqrt(1e6/v[first-20:first].mean()/.01));nav=1e6/(1+rate)/q[first]*q[first:];nav[-1]*=1-cost*(.0007+.0003*np.sqrt(nav[-1]/v[-21:-1].mean()/.01))
        else:
            prediction=boot[int(seed)] if name=='seed_bootstrap' else raw[seed_position[int(seed)]] if name=='individual_seed' else scores[name]
            nav=policy_replay(base,ids,prediction,10,cost,exposure=np.ones(len(base['origin'])) if policy=='full' else trend)
        actual=1e6*np.cumprod(1+path.net.to_numpy());largest=max(largest,float(np.abs(nav-actual).max()));np.testing.assert_allclose(nav,actual,atol=1e-5,rtol=1e-10);accounts+=1
        if accounts%50==0:print('Independently replayed seed accounts',accounts,'/330',flush=True)
    prior,metadata,files=references(cf[(cf.model=='SPY')&(cf.cost_multiplier==1)].date.to_numpy())
    saved=pd.read_csv(RUN/'references.csv').set_index('reference');metadata=metadata.set_index('reference')
    for name in metadata.index:
        for field in ['annualized_return','sharpe','maximum_drawdown']:np.testing.assert_allclose(saved.loc[name,field],metadata.loc[name,field],atol=1e-12,rtol=0)
        for field,value in independent_metrics(prior['reference',name]).items():
            np.testing.assert_allclose(saved.loc[name,field],value,atol=1e-10,rtol=1e-10)
    ledger=pd.read_csv(RUN/'primary_ledgers.csv');np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,atol=1e-7,rtol=1e-12)
    if accounts!=330 or reproduced!=80 or ledger.cash.min()<-1e-5:raise ValueError('Seed audit inventory/cash failure')
    write(RUN/'AUDIT.json',dict(status='passed',new_checkpoints_reproduced=80,original_seed_forecast_exports_verified=20,
          independent_accounts=330,prior_reference_paths_verified=len(prior),largest_nav_difference_dollars=largest,
          prior_reference_validation='Exact saved bytes, dates and independent metrics checked; previous full account/checkpoint audits retained'))


def publish():
    complete();audit_record=json.loads((RUN/'AUDIT.json').read_text())
    if audit_record['status']!='passed':raise ValueError('Seed audit not passed')
    for name in ['EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','REFERENCE_LOCK.json']:
        for p,h in json.loads((RUN/name).read_text()).items():
            if sha(p)!=h:raise ValueError('Seed publication source/reference changed')
    DOCS.mkdir(parents=True,exist_ok=True);figures=DOCS/'figures';figures.mkdir(exist_ok=True)
    mf=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str});cf=pd.read_csv(RUN/'curves.csv',dtype={'seed':str});plan=json.loads((RUN/'PROTOCOL.json').read_text())
    main=mf[(mf.seed=='ensemble')&(mf.cost_multiplier==1)].set_index(['model','policy']);screen=pd.read_csv(RUN/'historical_screen.csv')
    ci=pd.read_csv(RUN/'paired_intervals.csv');ci20=ci[ci.block_length==20];seedtime=pd.read_csv(RUN/'seed_time_intervals.csv');refs=pd.read_csv(RUN/'references.csv')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for policy in ['full','trend_half']:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for name,color in [('all_twenty_five','black'),('original_five','#9467bd'),('new_twenty','#1f77b4'),('SPY','#cc0000'),('QQQ','#333399')]:
            actual='full' if name in ['SPY','QQQ'] else policy;p=cf[(cf.model==name)&(cf.policy==actual)&(cf.seed=='ensemble')&(cf.cost_multiplier==1)]
            axes[0].plot(pd.to_datetime(p.date),np.cumprod(1+p.net),color=color,label=f'{name}: {pct(main.loc[name,actual].annualized_return)}')
        boot=mf[(mf.model=='seed_bootstrap')&(mf.policy==policy)];axes[1].hist(boot.annualized_return*100,bins=18,color='#1f77b4',alpha=.8)
        axes[1].axvline(main.loc['all_twenty_five',policy].annualized_return*100,color='black',label='25-seed primary');axes[1].axvline(main.loc['SPY','full'].annualized_return*100,color='#cc0000',linestyle='--',label='SPY')
        axes[0].set(title='Net wealth, exploratory 2022–2025',ylabel='Wealth per initial $1');axes[1].set(title='128 initialization resamples, same market path',xlabel='Net CAGR (%)',ylabel='Resampled portfolios')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(policy+' exposure; no seed selection');fig.tight_layout();fig.savefig(figures/(policy+'.png'),dpi=160);plt.close(fig)
    rows=[[name,policy,pct(r.annualized_return),f'{r.sharpe:.3f}',pct(r.maximum_drawdown)] for (name,policy),r in main.iterrows()]
    seedrows=[]
    for policy in ['full','trend_half']:
        for group,seeds in [('original_five',plan['original_seeds']),('new_twenty',plan['new_seeds'])]:
            p=mf[(mf.model=='individual_seed')&(mf.policy==policy)&mf.seed.isin([str(s) for s in seeds])]
            seedrows.append([group,policy,pct(p.annualized_return.min()),pct(p.annualized_return.median()),pct(p.annualized_return.max()),f"{int((p.annualized_return>main.loc['SPY','full'].annualized_return).sum())}/{len(p)}"])
        p=mf[(mf.model=='seed_bootstrap')&(mf.policy==policy)]
        seedrows.append(['128 resampled 25-seed blends',policy]+[pct(p.annualized_return.quantile(q)) for q in [.025,.5,.975]]+[f"{int((p.annualized_return>main.loc['SPY','full'].annualized_return).sum())}/128"])
    keyci=ci20[ci20.right_model.isin(['SPY','QQQ','original_five','new_twenty'])]
    cirows=[[r.contrast,f'{r.estimate*100:.2f} pp',f'[{r.paired_low*100:.2f}, {r.paired_high*100:.2f}]',f'[{r.simultaneous_low*100:.2f}, {r.simultaneous_high*100:.2f}]'] for r in keyci.itertuples(index=False)]
    strows=[[r.policy,r.reference,f'{r.pointwise_low*100:.2f} to {r.pointwise_high*100:.2f} pp'] for r in seedtime[seedtime.block_sessions==20].itertuples(index=False)]
    screenrows=[[r.policy,str(r.historical_screen_passed),f'{r.years_above_spy}/4',f'{r.months_above_spy}/48',str(r.drawdown_within_five_pp),f'{r.positive_simultaneous_contrasts}/{r.comparisons}'] for r in screen.itertuples(index=False)]
    costrows=[]
    for (name,policy),p in mf[mf.seed=='ensemble'].groupby(['model','policy']):
        p=p.set_index('cost_multiplier');costrows.append([name,policy]+[pct(p.loc[c,'annualized_return']) for c in [1,2,4]])
    full=main.loc['all_twenty_five','full'];old=main.loc['original_five','full'];new=main.loc['new_twenty','full'];spy=main.loc['SPY','full'];qqq=main.loc['QQQ','full']
    lines=['# NASDAQ-100 seed-count expansion — October 20, 2026','',
        f'The fixed 25-seed primary returned **{pct(full.annualized_return)} CAGR**, versus **{pct(old.annualized_return)}** for the original five, **{pct(new.annualized_return)}** for the twenty new seeds, **{pct(spy.annualized_return)}** for SPY and **{pct(qqq.annualized_return)}** for QQQ. '
        f'{int(screen.historical_screen_passed.sum())}/2 exposure policies passed the historical reliability screen. Reliable superiority over SPY and competing models remains unestablished unless supported by independent future evidence; these previously inspected years are exploratory.','',
        'The new seeds were generated and locked before any new fits or metrics. Every new seed is retained. Eighty new THINK+mixer fits use the exact original five-feature architecture, train-only normalization/graph, raw-return MSE and prior-year validation-MSE early stopping. '
        'Each validation year chooses the earliest best checkpoint within 30 epochs, with patience six. No seed or ensemble weight is selected by test return. '
        'The primary is the equal eligible-rank blend of all 25 seeds; original five and new twenty are paired controls. This isolates initialization/ensemble-size sensitivity. '
        'The preceding [Alpaca comparison](../nasdaq_alpaca_20261019/REPORT.md) and [bounded intraday correction study](../ndx_corrections_20261020/REPORT.md) test the added intraday information. '
        'This particular core uses daily features; it does not imply that 15-minute features improved performance.','',
        '## Accounts and consistency','',md_table(['Ensemble / benchmark','Policy','Net CAGR','Sharpe','Maximum drawdown'],rows),'',
        'All accounts use historical monthly membership, the reviewed economic-price panel, five-session signals, frozen next-close orders, $1 million initial cash, 1% ADV caps and modeled fees. '
        'Cash earns no interest; Sharpe uses the fixed 3% hurdle. The fixed trend policy holds half stock exposure below the signal-time QQQ MA200. '
        'The primary full-exposure policy was declared before metrics; exposure is not selected from test results.','',
        '![Full exposure](figures/full.png)','', '![Fixed trend exposure](figures/trend_half.png)','',
        md_table(['Policy','Historical screen passed','Years above SPY','Months above SPY','Drawdown within 5 pp','Positive simultaneous contrasts'],screenrows),'',
        'The historical screen requires base- and double-fee CAGR above SPY, at least 3/4 years and 28/48 months above SPY, and drawdown at most five percentage points worse. '
        'Passing would remain development evidence and would not replace the comparison intervals or future confirmation.','',
        md_table(['Ensemble / benchmark','Policy','1× fee CAGR','2× fee CAGR','4× fee CAGR'],costrows),'',
        '## Initialization sensitivity','',md_table(['Seed group / resample','Policy','Minimum / 2.5th percentile','Median','Maximum / 97.5th percentile','Paths above SPY'],seedrows),'',
        'Individual-seed rows report minimum/median/maximum. Resampled-blend rows report 2.5/50/97.5 percentiles from 128 seed-resampled accounts. '
        'Each draw samples 25 seed positions with replacement, keeps the same draw across annual folds/policies, recomputes the rank blend and runs the full account. '
        'These are initialization diagnostics conditional on the same historical market path, data, validation rules and checkpoints. They are not 128 independent investment histories or a future-return confidence interval.','',
        '## Paired market-path and model comparisons','',md_table(['Primary contrast','Active mean/year','Paired 95% interval, pp','Broad-family simultaneous diagnostic, pp'],cirows),'',
        f'The broad family includes {len(refs)} prior base-cost ensemble reference paths plus SPY, QQQ and both seed-count controls for each primary exposure policy. '
        f'There are {int(ci20.family_comparisons.iloc[0])} joint contrasts. Every available original/Alpaca/ranking equal-top-ten ensemble path and every broad/correction ensemble policy path is retained, including unsuccessful and ungated diagnostic policies. '
        'Other exposure policies remain explicitly labeled. No best reference or test year is selected. Previously retired 2026-selected universes are excluded. '
        'Exact reference names, source paths and metrics are in the reference inventory; the full interval table includes every contrast. '
        'Ten thousand whole-day paired circular-block samples use 5/20/60-session blocks; 20 is primary. These fitted-path intervals omit the full prior research search, data revision and training/selection uncertainty. '
        'Repeated seed or time resampling cannot create additional independent years.','',
        md_table(['Primary policy','Reference','Pointwise joint seed/time 95% diagnostic, primary block'],strows),'',
        'The additional diagnostic first draws one of the 128 seed-resampled accounts, then resamples paired daily blocks against a fixed reference (10,000 replicates). '
        'It allows a finite approximation to initialization variation and market-path sampling together, but is pointwise, conditions on fixed checkpoints/choices and does not cover the full historical search. '
        'It must not be substituted for a corrected superiority claim. Arithmetic active-return means differ from CAGR gaps.','',
        '## Verification and remaining evidence','',
        f'All 80 new checkpoints and validation/test exports were reproduced. Twenty original annual forecast exports were hash checked and their independent rank aggregation verified. '
        f'All 330 new accounts, including 256 seed-bootstrap accounts, were independently replayed; the maximum NAV discrepancy was ${audit_record["largest_nav_difference_dollars"]:.8f}. '
        'Every prior reference’s bytes, exact calendar and independent return metrics were verified against its preserved account output; its earlier checkpoint/account audit remains the provenance. '
        'The original studies and model checkpoints remain unchanged. Tests verify seed uniqueness, fixed resampling, weighted rank equivalence and combined uncertainty behavior.','',
        'The usable test history is four already-inspected calendar years. Corporate-action/total-return approximations, missing labels, revised vendor histories and changing regimes limit inference. '
        'More initialization seeds can stabilize an ensemble; they cannot prove a durable economic edge. Future dates reserved from October 21, 2026 were not evaluated. '
        'A reviewed updated point-in-time dataset and fixed prospective model/refit/execution policy are still required before those dates can provide confirmation. No orders or recurring jobs were created.','',
        '## Artifacts','',
        '[Protocol and every seed](PROTOCOL.json) · [Audit](AUDIT.json) · [All metrics](metrics.csv) · [Every daily path](curves.csv) · [References](references.csv) · '
        '[All paired intervals](paired_intervals.csv) · [Seed/time diagnostics](seed_time_intervals.csv) · [Historical screens](historical_screen.csv) · '
        '[Annual returns](annual.csv) · [Monthly returns](monthly.csv)','',
        'Checkpoints, preprocessing, bootstrap draws, prediction exports and integrity locks are under `runs/ndx_seed_expansion_20261020`.','']
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ['PROTOCOL.json','AUDIT.json','FREEZE.json','ENVIRONMENT.json','UNIT_TESTS.json','EVALUATION_UNIT_TESTS.json','EVALUATED.json',
                 'metrics.csv','curves.csv','references.csv','paired_intervals.csv','seed_time_intervals.csv','historical_screen.csv','annual.csv','monthly.csv',
                 'EVALUATION_SOURCE_LOCK.json','AUDIT_SOURCE_LOCK.json','REFERENCE_LOCK.json']:shutil.copyfile(RUN/name,DOCS/name)
    for name in ['report_ndx_seed_expansion.py','test_ndx_seed_report.py']:shutil.copyfile(ROOT/name,RUN/'source_snapshot'/name)
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_research',new_fits=80,accounts=330,primary_full_cagr=float(full.annualized_return),
          original_five_cagr=float(old.annualized_return),new_twenty_cagr=float(new.annualized_return),spy_cagr=float(spy.annualized_return),
          historical_screens=screen.to_dict('records'),future_reliability='not established',report=str(DOCS/'REPORT.md'),
          artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    write(RUN/'PROGRESS.json',dict(stage='completed_audited_published',new_fits=80,accounts=330));print('Published',DOCS/'REPORT.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','audit','publish']);args=parser.parse_args();globals()[args.stage]()
