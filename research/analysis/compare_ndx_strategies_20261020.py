"""Read-only consolidation of active historical strategies and Sharpe intervals."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pit_research_engine import load_data,fold

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
RF=.03
LABELS=dict(think='THINK',think_mix='THINK + mixer (original five seeds)',think_huber='THINK + Huber',
    think_mix_huber='THINK + mixer + Huber',think_mix_pseudohuber='THINK + mixer + pseudo-Huber',
    mlp='MLP',stockmixer='StockMixer',ridge='Ridge',momentum126='126-session momentum',reversal5='Five-session reversal',
    monthly_basket='Monthly equal-weight Nasdaq-100 basket',SPY='SPY',QQQ='QQQ',QQQ_timing='QQQ with fixed trend exposure',
    validation_top3='Validation top-three blend',validation_best='Validation best single model',equal_all='Equal blend of all 12 models',
    paired_core='Paired core (H1 validation stopping)',primary='Validation-gated correction',
    all_twenty_five='THINK + mixer (25-seed ensemble)',new_twenty='THINK + mixer (20 new seeds)')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def performance(returns):
    a=np.asarray(returns,float);wealth=np.r_[1.,np.cumprod(1+a)]
    return dict(sharpe=float(np.sqrt(252)*(a.mean()-((1+RF)**(1/252)-1))/a.std(ddof=1)),
                cagr=float(wealth[-1]**(252/len(a))-1),max_drawdown=float(np.min(wealth/np.maximum.accumulate(wealth)-1)))


def bootstrap_sharpe(returns,block=20,repeats=10000,seed=None):
    """Exact circular-block moments; identical date draws for every strategy."""
    values=np.asarray(returns,float)
    if values.ndim==1:values=values[:,None]
    n,m=values.shape
    if n<2 or not np.isfinite(values).all() or block<1 or block>n:raise ValueError('Invalid bootstrap input')
    doubled=np.concatenate([values,values],axis=0)
    prefix=np.vstack([np.zeros((1,m)),np.cumsum(doubled,axis=0)])
    prefix2=np.vstack([np.zeros((1,m)),np.cumsum(doubled*doubled,axis=0)])
    full,remainder=divmod(n,block);starts=np.arange(n)
    sums=prefix[starts+block]-prefix[starts];squares=prefix2[starts+block]-prefix2[starts]
    tail=prefix[starts+remainder]-prefix[starts];tail2=prefix2[starts+remainder]-prefix2[starts]
    rng=np.random.default_rng(20261020+block if seed is None else seed);samples=[]
    for first in range(0,repeats,100):
        count=min(100,repeats-first);draw=rng.integers(0,n,(count,full+bool(remainder)))
        total=sums[draw[:,:full]].sum(1);square=squares[draw[:,:full]].sum(1)
        if remainder:total+=tail[draw[:,-1]];square+=tail2[draw[:,-1]]
        variance=(square-total*total/n)/(n-1)
        sd=np.sqrt(np.maximum(variance,0.))
        if np.any(sd<=1e-14):raise ValueError('Undefined resampled Sharpe')
        samples.append(np.sqrt(252)*(total/n-((1+RF)**(1/252)-1))/sd)
    return np.concatenate(samples)


def raw_rankic(score,target,observed):
    result=[]
    for p,y,mask in zip(score,target,observed):
        if mask.sum()<2:continue
        a,b=rankdata(p[mask]),rankdata(y[mask]);a-=a.mean();b-=b.mean()
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        if norm>1e-14:result.append(float(a@b/norm))
    return float(np.mean(result)) if result else None


def model_label(model):
    if model in LABELS:return LABELS[model]
    if '__' in model:
        pieces=model.split('__')
        if len(pieces)==3 and pieces[0]!='ungated':
            view,target,family=pieces
            return f"{dict(ridge='Ridge',boosted_trees='Boosted trees',shared_network='Shared neural net')[family]} ({view}, {target})"
        ungated=pieces[0]=='ungated'
        if ungated:pieces=pieces[1:]
        view,objective=pieces
        return f"{dict(intraday_window='Intraday-window',rich_features='Rich-feature')[view]} correction / {dict(top_rank='ranking',net_utility='utility')[objective]} ({'ungated' if ungated else 'gated'})"
    return model


def assemble():
    rows=[];paths=[];sources={};calendar=None;rankchecks=[]
    def read(path):
        nonlocal sources
        path=ROOT/path;sources[str(path.relative_to(ROOT))]=sha(path)
        return pd.read_csv(path,dtype={'seed':str})
    def add(study,key,model,policy,k,row,path,rankic,section='Models',label=None):
        nonlocal calendar
        dates=path.date.to_numpy()
        if calendar is None:calendar=dates
        if len(dates)!=1003 or not np.array_equal(calendar,dates):raise ValueError('Nonmatching dates: '+key)
        actual=performance(path.net.to_numpy())
        for field,saved in [('sharpe','sharpe'),('cagr','annualized_return'),('max_drawdown','maximum_drawdown')]:
            np.testing.assert_allclose(actual[field],row[saved],atol=1e-10,rtol=1e-10)
        rows.append(dict(id=key,study=study,strategy=label or model_label(model),policy=policy,k=int(k),
                         section=section,rankic=rankic,**actual))
        paths.append(path.net.to_numpy())
    original='runs/nasdaq_topk_pseudohuber_20261011'
    m,c,r=[read(original+'/'+name) for name in ['metrics.csv','curves.csv','ranking.csv']]
    ranks=dict(zip(r.model,r.RankIC))
    for _,row in m[(m.seed=='ensemble')&(m.cost_multiplier==1)].iterrows():
        model=row.model;k=int(row.k)
        if model in ['SPY','QQQ','monthly_basket'] and k!=10:continue
        path=c[(c.model==model)&(c.seed=='ensemble')&(c.k==k)&(c.cost_multiplier==1)]
        add('Original daily',f'original/{model}/{k}',model,'full',k,row,path,ranks.get(model),
            'Models' if k==10 else 'Portfolio variants',label=None if k==10 else f'{model_label(model)} / top {k}')
    # Correlation of the underlying stock forecast is unchanged by allocation.
    m=read('runs/nasdaq_long_short_20261011/metrics.csv');c=read('runs/nasdaq_long_short_20261011/curves.csv')
    selected=(m.seed=='ensemble')&(m.cost_multiplier==1)&((np.isclose(m.borrow_rate,.03))|np.isclose(m.borrow_rate,0))&(m.policy!='long100')
    policylabels=dict(long75_cash25='75% long + 25% cash',long50_cash50='50% long + 50% cash',
                      long75_short25='75% long / 25% short',long50_short50='50% long / 50% short')
    for _,row in m[selected].iterrows():
        add('Original allocations',f'signed/{row.model}/{int(row.k)}/{row.policy}',row.model,row.policy,row.k,row,
            c[c.path_id==row.path_id],ranks[row.model],'Portfolio variants',
            f'{model_label(row.model)} / top {int(row.k)} / {policylabels[row.policy]}')
    run='runs/nasdaq100_alpaca_20261019';m,c,r=[read(run+'/'+name) for name in ['metrics.csv','curves.csv','ranking.csv']]
    ranks_a={(v,model):ic for v,model,ic in zip(r.view,r.model,r.RankIC)}
    for _,row in m[(m.seed=='ensemble')&(m.cost_multiplier==1)&(m.view!='control')].iterrows():
        model=row.model;view=row['view'];k=int(row.k)
        if model in ['momentum126','reversal5']:continue
        viewname='Daily padded' if view=='daily_control' else 'Alpaca intraday'
        label=f'{model_label(model).replace(" (original five seeds)","")} / {viewname}'
        path=c[(c.model==model)&(c.view==view)&(c.seed=='ensemble')&(c.k==k)&(c.cost_multiplier==1)]
        add('Alpaca matched features',f'alpaca/{view}/{model}/{k}',model,'full',k,row,path,ranks_a[view,model],
            'Models' if k==10 else 'Portfolio variants',label=label if k==10 else label+f' / top {k}')
    run='runs/ndx_ranking_20261020';m,c,r=[read(run+'/'+name) for name in ['metrics.csv','curves.csv','ranking_summary.csv']]
    ranks_r={(v,model):ic for v,model,ic in zip(r.view,r.model,r.raw_rankic)}
    for _,row in m[(m.seed=='ensemble')&(m.cost_multiplier==1)&m.model.isin(['raw_mse','residual_mse','residual_rank','ridge'])].iterrows():
        view=row['view'];model=row.model
        objective=dict(raw_mse='Raw MSE / RankIC stopping',residual_mse='QQQ-residual MSE',residual_rank='QQQ-residual + ranking',ridge='QQQ-residual ridge')[model]
        label=objective+' / '+('daily padded' if view=='daily_control' else 'Alpaca intraday')
        path=c[(c.model==model)&(c.view==view)&(c.seed=='ensemble')&(c.cost_multiplier==1)]
        add('Ranking objectives',f'ranking/{view}/{model}',model,'full',10,row,path,ranks_r[view,model],label=label)
    for study,run in [('Broad features','ndx_broad_20261020'),('Frozen-core corrections','ndx_corrections_20261020')]:
        folder='runs/'+run;m,c,r=[read(folder+'/'+name) for name in ['metrics.csv','curves.csv','ranking_summary.csv']]
        ranked=dict(zip(r.model,r.raw_rankic))
        for _,row in m[(m.seed=='ensemble')&(m.cost_multiplier==1)].iterrows():
            model=row.model;policy=row.policy
            if model in ['SPY','QQQ','monthly_basket','momentum126']:continue
            if model.startswith('reference'):
                # Reuse each existing logical strategy once under each policy.
                if study!='Broad features' or policy!='trend_half':continue
                alias=dict(reference_original_think_mix=('original','think_mix',ranks['think_mix']),
                           reference_rank_daily=('ranking','raw_mse',ranks_r['daily_control','raw_mse']),
                           reference_rank_intraday=('ranking','raw_mse',ranks_r['intraday_augmented','raw_mse']))[model]
                label=dict(reference_original_think_mix='Original THINK + mixer',reference_rank_daily='Daily raw MSE / RankIC stopping',reference_rank_intraday='Intraday raw MSE / RankIC stopping')[model]
                ic=alias[2]
            else:
                label=model_label(model);ic=ranked.get(model)
            if study=='Frozen-core corrections' and model=='QQQ_timing':continue
            path=c[(c.model==model)&(c.policy==policy)&(c.seed=='ensemble')&(c.cost_multiplier==1)]
            add(study,f'{run}/{model}/{policy}',model,policy,10,row,path,ic,
                'Models' if policy=='full' else 'Defensive',label=label)
    # Compute missing seed-expansion RankIC directly from its audited forecasts.
    base=load_data();ids=np.concatenate([fold(base,year)[2] for year in [2022,2023,2024,2025]])
    folder='runs/ndx_seed_expansion_20261020';m,c=[read(folder+'/'+name) for name in ['metrics.csv','curves.csv']]
    file=ROOT/folder/'scores.npz';sources[str(file.relative_to(ROOT))]=sha(file)
    with np.load(file) as score:
        np.testing.assert_array_equal(ids,score['ids'])
        seedic={key:raw_rankic(score[key],base['y'][ids],base['labelmask'][ids]) for key in ['original_five','new_twenty','all_twenty_five']}
    np.testing.assert_allclose(seedic['original_five'],ranks['think_mix'],atol=1e-12,rtol=0)
    for _,row in m[(m.seed=='ensemble')&(m.cost_multiplier==1)&m.model.isin(['new_twenty','all_twenty_five'])].iterrows():
        path=c[(c.model==row.model)&(c.policy==row.policy)&(c.seed=='ensemble')&(c.cost_multiplier==1)]
        add('Seed expansion',f'seeds/{row.model}/{row.policy}',row.model,row.policy,10,row,path,seedic[row.model],
            'Models' if row.policy=='full' else 'Defensive')
    manifest=ROOT/'data/nasdaq_monthly_pit/assembled/manifest.json'
    sources[str(manifest.relative_to(ROOT))]=sha(manifest)
    for field,value in json.loads(manifest.read_text())['hashes'].items():
        path=manifest.parent/field
        if sha(path)!=value:raise ValueError('Reviewed input changed')
        sources[str(path.relative_to(ROOT))]=value
    return pd.DataFrame(rows),np.column_stack(paths),calendar,sources


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows,paths,calendar,sources=assemble()
    if rows.id.duplicated().any():raise ValueError('Duplicate strategy IDs')
    np.savez_compressed(OUT/'analysis_paths.npz',returns=paths,dates=np.asarray(calendar,dtype=str),ids=rows.id.to_numpy(dtype=str))
    sensitivity=[]
    for block in [5,20,60]:
        samples=bootstrap_sharpe(paths,block=block)
        low,high=np.quantile(samples,[.025,.975],axis=0)
        for j,row in rows.iterrows():sensitivity.append(dict(id=row.id,block_sessions=block,sr_low=float(low[j]),sr_high=float(high[j])))
        if block==20:rows['sr_low']=low;rows['sr_high']=high
        print('Computed 10,000 common-date Sharpe resamples, block',block,'strategies',len(rows),flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/'sharpe_block_sensitivity.csv',index=False)
    # Baselines first, then study order. Preserve every declared unsuccessful cell.
    order={v:i for i,v in enumerate(['Original daily','Alpaca matched features','Ranking objectives','Broad features','Frozen-core corrections','Seed expansion'])}
    rows['study_order']=rows.study.map(order).fillna(99)
    rows['baseline_order']=rows.strategy.map({'SPY':0,'QQQ':1}).fillna(2)
    rows=rows.sort_values(['baseline_order','study_order','id'],kind='stable').drop(columns=['baseline_order','study_order'])
    rows.to_csv(OUT/'all_strategies.csv',index=False)
    output=[]
    for section in ['Models','Defensive','Portfolio variants']:
        frame=rows[rows.section==section]
        if section!='Models':frame=pd.concat([rows[rows.strategy.isin(['SPY','QQQ'])],frame],ignore_index=True)
        output.append(dict(name=section,rows=json.loads(frame.to_json(orient='records'))))
    payload=dict(sheets=output,period='2022-01-03 to 2025-12-31',observations=1003,
                 methodology=[
                    ['Period','January 3, 2022–December 31, 2025. All strategies use the same 1,003 daily observations.'],
                    ['Financial metrics','Net returns at each study’s base modeled trading costs. Ensemble forecasts; no seed selected by test performance.'],
                    ['Sharpe ratio','Annualized daily SR = sqrt(252) × [mean(daily return) − ((1.03)^(1/252)−1)] / sample SD(daily return).'],
                    ['95% Sharpe interval','Pointwise 2.5th/97.5th percentiles from 10,000 circular moving-block resamples. Primary block: 20 sessions.'],
                    ['Dependence','The same date-block draws apply to every strategy. Stocks, bars and seeds are not treated as independent observations.'],
                    ['Sharpe annualization','Conventional annualized daily ratio, matching existing reports. It is not a separately autocorrelation-adjusted annual-holding-period Sharpe.'],
                    ['RankIC','Mean per-origin cross-sectional Spearman correlation of ensemble scores with realized five-session STOCK returns, among observed eligible labels.'],
                    ['RankIC comparability','Raw-return RankIC throughout, including models trained on QQQ residuals. Approximately 200 complete forecast origins.'],
                    ['RankIC for overlays','RankIC describes the underlying stock forecast. Changing exposure or portfolio size does not change that forecast’s RankIC.'],
                    ['N/A','SPY, QQQ, the monthly equal-weight basket and single-ETF trend timing do not provide stock rankings. Their RankIC is not applicable.'],
                    ['Defensive rule','Trend-half policies target full stock exposure when signal-time QQQ is at/above MA200, half exposure plus cash otherwise.'],
                    ['Portfolio variants','Top-five/top-twenty holdings and original cash/short allocations. Short accounts include 3% annual borrow, with borrow availability assumed.'],
                    ['Excluded repetitions','Reference aliases, individual seeds, fee stresses and seed-bootstrap accounts are omitted. Distinct declared selection policies remain, even when their realized paths coincide.'],
                    ['Historical scope','Active historical monthly Nasdaq-100 only. Retired 2026-selected survivor universes and deferred NYSE studies are excluded.'],
                    ['Interpretation','Previously inspected historical development. Intervals condition on fitted paths and omit full research-search, refit, data-revision and future-regime uncertainty.'],
                    ['Comparison warning','A strategy’s SR interval is not an interval for its advantage over SPY/QQQ. Overlap or zero inclusion alone is not a paired superiority test.'],
                    ['Sensitivities','Five- and 60-session block intervals are included in the accompanying sensitivity CSV. No block chosen for favorable bounds.'],
                    ['Source studies','Audited original monthly/top-k/long-short, matched Alpaca, ranking, broad-feature, frozen-core correction and seed-expansion studies.'],
                    ['Block-bootstrap reference','Politis and Romano, A Circular Block-Resampling Procedure for Stationary Data: https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data'],
                    ['Sharpe reference','Andrew W. Lo, The Statistics of Sharpe Ratios (2002): https://alo.mit.edu/publications/page/18/']])
    (OUT/'workbook_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# NASDAQ-100 strategy comparison','',payload['period']+'. Net returns at base modeled costs. SR uses a fixed 3% annual hurdle.','',
           '95% SR intervals: 10,000 common-date circular-block samples, 20-session primary blocks. These are pointwise fitted-path intervals, not corrected superiority claims.','']
    for section in output:
        lines.extend(['## '+section['name'],'','| Strategy | Study | Sharpe | CAGR | RankIC | Max drawdown | SR 95% CI |',
                      '| --- | --- | ---: | ---: | ---: | ---: | --- |'])
        for r in section['rows']:
            lines.append(f"| {r['strategy']} | {r['study']} | {r['sharpe']:.3f} | {r['cagr']*100:.2f}% | {'N/A' if r['rankic'] is None else format(r['rankic'],'.4f')} | {r['max_drawdown']*100:.2f}% | [{r['sr_low']:.3f}, {r['sr_high']:.3f}] |")
        lines.append('')
    lines.extend(['## Definitions and limitations','']+[f'- **{k}:** {v}' for k,v in payload['methodology']])
    (OUT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write=dict(status='computed_pending_workbook',created_at_utc=datetime.now(timezone.utc).isoformat(),strategies=len(rows),
               sections=rows.section.value_counts().to_dict(),daily_observations=len(calendar),source_hashes=sources,
               analysis_source_sha256=sha(__file__),method='10000 circular-block percentile Sharpe replicates, 5/20/60 sessions; 3 percent annual hurdle')
    (OUT/'ANALYSIS.json').write_text(json.dumps(write,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in write.items() if k!='source_hashes'},indent=2),flush=True)


if __name__=='__main__':main()
