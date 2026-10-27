"""Fixed saved-score selection, initialization and as-of price-style diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold,continuous_backtest
from research_engine import target_weights
from run_robust_research import ensemble_predictions
from rishi_mechanism_power import stationary_draws
from rishi_mechanism_batch1 import stationary_indices,paired_contrast
from rishi_raw_replay import persistent_mean,SEEDS,PARENT,REFERENCE

DOC=Path('docs/rishi_mechanism_20261027');SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
CANDIDATES=['B','D','E','Hlabel','E_Hlabel','REV5','MOM126','ridge5','coverage6']
MEASURES=['rankic','top5_minus_ew','top10_minus_ew','decile_spread']


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<2 or a[valid].std()==0 or b[valid].std()==0:return np.nan
    return float(np.corrcoef(a[valid],b[valid])[0,1])


def tail_weights(score,eligible,fraction=.1):
    score=np.asarray(score,float);eligible=np.asarray(eligible,bool)
    if not 0<fraction<=1 or not eligible.any() or not np.isfinite(score[eligible]).all():raise ValueError('Finite eligible tail required')
    mass=float(eligible.sum()*fraction);remaining=mass;result=np.zeros(len(score))
    for value in np.unique(score[eligible])[::-1]:
        tied=eligible&(score==value);take=min(remaining,float(tied.sum()))
        result[tied]=take/tied.sum()/mass;remaining-=take
        if remaining<=1e-12:break
    np.testing.assert_allclose(result.sum(),1,atol=1e-12)
    return result


def observed_label_return(weights,labels,observed):
    valid=np.asarray(observed,bool)&np.isfinite(labels);mass=float(np.asarray(weights)[valid].sum())
    return (float(np.asarray(weights)[valid]@np.asarray(labels)[valid]/mass) if mass>0 else np.nan),mass


def stratified_draws(years,reps,block,seed):
    years=np.asarray(years);result=np.empty((reps,len(years)),dtype=np.int32);rng=np.random.default_rng(seed)
    for year in np.unique(years):
        positions=np.flatnonzero(years==year)
        result[:,positions]=positions[stationary_draws(len(positions),reps,len(positions),block,rng)]
    return result


def active_ir(left,right):
    difference=np.asarray(left,float)-np.asarray(right,float);sd=difference.std(axis=-1,ddof=1)
    return np.divide(difference.mean(axis=-1)*np.sqrt(252),sd,out=np.full(np.shape(sd),np.nan),where=sd>1e-12)


def covariance_terms(b,d,y,h):
    b,d,y,h=[np.asarray(a,float) for a in (b,d,y,h)]
    cov=lambda a,z:float(np.mean((a-a.mean())*(z-z.mean())))
    eb=b-y;ed=d-y
    return dict(forecast_correlation=corr(b,d),error_correlation=corr(eb,ed),
        innovation_correlation=corr(b-h,d-h),forecast_covariance=cov(b,d),b_target_covariance=cov(b,y),
        d_target_covariance=cov(d,y),target_variance=cov(y,y),error_covariance=cov(eb,ed),
        common_baseline_error_invariance_max=float(max(np.max(np.abs(((b-h)-(y-h))-eb)),np.max(np.abs(((d-h)-(y-h))-ed)))))


def asof_style(data,day,qqq_returns):
    n=data['returns'].shape[1];result=np.full((n,4),np.nan)
    start=max(0,day-251);q=qqq_returns[start:day+1]
    for j in range(n):
        r=data['returns'][start:day+1,j];valid=np.isfinite(r)&np.isfinite(q)&data['tradable'][start:day+1,j]
        if valid.sum()>=126 and q[valid].var()>0:result[j,0]=np.cov(r[valid],q[valid],ddof=1)[0,1]/q[valid].var(ddof=1)
        if day>=126 and data['tradable'][day,j] and data['tradable'][day-126,j] and data['marks'][day-126,j]>0:
            result[j,1]=data['marks'][day,j]/data['marks'][day-126,j]-1
        r=data['returns'][max(0,day-19):day+1,j];valid=np.isfinite(r)&data['tradable'][max(0,day-19):day+1,j]
        if valid.sum()>=15:result[j,2]=r[valid].std(ddof=1)*np.sqrt(252)
        adv=data['adv'][day,j]
        if np.isfinite(adv) and adv>0:result[j,3]=np.log10(adv)
    return result


def origin_metrics(score,eligible,labels,observed):
    observed=observed&eligible&np.isfinite(labels)
    ew=observed_label_return(eligible.astype(float)/eligible.sum(),labels,observed)[0]
    rho=corr(pd.Series(score[observed]).rank(),pd.Series(labels[observed]).rank())
    result=dict(rankic=rho,eligible_ew_label=ew,observed_stocks=int(observed.sum()))
    for k in (5,10):
        value,mass=observed_label_return(target_weights(score,eligible,k,'equal'),labels,observed)
        result[f'top{k}_minus_ew']=value-ew;result[f'top{k}_observed_weight']=mass
    top,tm=observed_label_return(tail_weights(score,eligible),labels,observed)
    bottom,bm=observed_label_return(tail_weights(-score,eligible),labels,observed)
    result.update(decile_spread=top-bottom,top_tail_observed_weight=tm,bottom_tail_observed_weight=bm)
    return result


def main():
    out=DOC/'execution_batch10_selection';large=Path('results/RISHI_MECHANISM_SELECTION_20261027')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();scores={};ids=None
    for candidate in CANDIDATES:
        with np.load(SOURCE/f'scores_{candidate}_ten_seed.npz') as saved:
            if ids is None:ids=saved['ids'].copy()
            else:np.testing.assert_array_equal(ids,saved['ids'])
            scores[candidate]=saved['scores'].copy()
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy();rows=[]
    for candidate in CANDIDATES:
        for row,i in enumerate(ids):
            rows.append(dict(candidate=candidate,origin_id=int(i),year=int(years[row]),
                **origin_metrics(scores[candidate][row],data['mask'][i],data['y'][i],data['labelmask'][i])))
    per=pd.DataFrame(rows);per.to_csv(out/'PER_ORIGIN.csv',index=False)
    per.groupby('candidate',sort=False)[MEASURES].mean().to_csv(out/'SELECTION_MEANS.csv')
    contrasts=[]
    for block in [4,1,13]:
        draws=stratified_draws(years,10000,block,27102034)
        for right in ['B','D','Hlabel','E_Hlabel','REV5']:
            for measure in MEASURES:
                left=per[per.candidate=='E'][measure].to_numpy();other=per[per.candidate==right][measure].to_numpy()
                difference=left-other;bootstrap=np.nanmean(difference[draws],axis=1);low,high=np.quantile(bootstrap,[.025,.975])
                contrasts.append(dict(right=right,measure=measure,block_origins=block,
                    difference=float(np.nanmean(difference)),low=float(low),high=float(high),valid_origins=int(np.isfinite(difference).sum())))
    pd.DataFrame(contrasts).to_csv(out/'PAIRED_SELECTION.csv',index=False)
    print('Completed per-origin paired selection',flush=True)
    daily=pd.read_csv(SOURCE/'ACCOUNT_CURVES.csv')
    with np.load('results/RISHI_MECHANISM_RISK_20261027/PATHS_PORTABLE.npz',allow_pickle=False) as saved:
        j=np.flatnonzero(saved['ids']=='EW/benchmark/eligible_weekly/full/modeled1')[0]
        ew=saved['returns'][:,j].copy();dates=saved['dates'].tolist()
    active=[]
    for block in [21,5,63]:
        draws=stationary_indices(len(ew),10000,block,27102035)
        for candidate in CANDIDATES:
            for k in (5,10):
                sub=daily[daily.id==f'{candidate}/ten_seed/{k}']
                if sub.date.tolist()!=dates:raise ValueError('Daily active dates')
                path=sub.net.to_numpy();boot=active_ir(path[draws],ew[draws]);low,high=np.nanquantile(boot,[.025,.975])
                active.append(dict(candidate=candidate,k=k,block=block,active_ir=float(active_ir(path,ew)),
                    active_ir_low=float(low),active_ir_high=float(high),**paired_contrast(path,ew,draws,252)))
    pd.DataFrame(active).to_csv(out/'ACTIVE_VS_EW.csv',index=False)
    print('Completed daily active IR',flush=True)
    raw={family:{seed:[] for seed in SEEDS} for family in ['B','D']};raw_h=[]
    for year in [2022,2023,2024,2025]:
        tr,_,te=fold(data,year);np.testing.assert_array_equal(te,ids[years==year])
        means,_=persistent_mean(data['y'][tr]*100,data['labelmask'][tr]);raw_h.append(np.broadcast_to(means,(len(te),len(means))))
        for family in ['B','D']:
            root=PARENT if family=='B' else REFERENCE/'dynamic_restricted'
            for seed in SEEDS:
                with np.load(root/str(year)/f'{seed}_predictions.npz') as saved:raw[family][seed].append(saved['test'].copy())
    raw={f:{seed:np.concatenate(parts) for seed,parts in family.items()} for f,family in raw.items()}
    raw_h=np.concatenate(raw_h);complement=[]
    for group,seeds in [('original_five',SEEDS[:5]),('new_five',SEEDS[5:]),('ten_seed',SEEDS)]:
        b=np.mean([raw['B'][s] for s in seeds],axis=0);d=np.mean([raw['D'][s] for s in seeds],axis=0)
        for year in [2022,2023,2024,2025,'pooled']:
            use=np.ones(len(ids),bool) if year=='pooled' else years==year
            observed=data['labelmask'][ids[use]]
            value=covariance_terms(b[use][observed],d[use][observed],(data['y'][ids[use]]*100)[observed],raw_h[use][observed])
            np.testing.assert_allclose(value['error_covariance'],value['forecast_covariance']-value['b_target_covariance']-value['d_target_covariance']+value['target_variance'],rtol=1e-10,atol=1e-10)
            complement.append(dict(group=group,year=year,observed_stock_origins=int(observed.sum()),**value))
    pd.DataFrame(complement).to_csv(out/'RAW_COMPLEMENTARITY.csv',index=False)
    bundle_plan=json.loads((DOC/'SEED_BUNDLES.json').read_text());bundles=[]
    for bundle in bundle_plan['bundles']:
        eligibility=data['mask'][ids]
        b=ensemble_predictions([raw['B'][s] for s in bundle['seeds']],eligibility)
        d=ensemble_predictions([raw['D'][s] for s in bundle['seeds']],eligibility)
        for candidate,score in [('B',b),('D',d),('E',.5*b+.5*d)]:
            for k in (5,10):
                metrics,curve,ledger=continuous_backtest(data,ids,score,k)
                bundles.append(dict(bundle=bundle['id'],candidate=candidate,k=k,**metrics))
                curve.to_csv(large/f"{bundle['id']}_{candidate}_{k}.csv",index=False)
        print('Completed fixed '+bundle['id'],flush=True)
    pd.DataFrame(bundles).to_csv(out/'SEED_BUNDLE_ACCOUNTS.csv',index=False)
    root=Path(data['manifest']['root']);qqq=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).to_numpy()
    style=[]
    for row,i in enumerate(ids):
        values=asof_style(data,int(data['origin'][i]),qqq);eligible=data['mask'][i];preference=scores['Hlabel'][row]
        for j,name in enumerate(['beta252','momentum126','volatility20','log10adv20']):
            good=eligible&np.isfinite(values[:,j])
            style.append(dict(origin_id=int(i),year=int(years[row]),feature=name,pairs=int(good.sum()),
                spearman=corr(pd.Series(preference[good]).rank(),pd.Series(values[good,j]).rank())))
    style=pd.DataFrame(style);style.to_csv(out/'STYLE_PER_ORIGIN.csv',index=False)
    style.groupby(['year','feature']).agg(mean_spearman=('spearman','mean'),mean_pairs=('pairs','mean'),min_pairs=('pairs','min')).to_csv(out/'STYLE_ANNUAL.csv')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),DOC/'SELECTION_EXECUTION_PROTOCOL.md',DOC/'SEED_BUNDLES.json']}
    (out/'RESULTS.json').write_text(json.dumps(dict(origins=len(ids),per_origin_rows=len(per),selection_contrasts=len(contrasts),
        daily_active_rows=len(active),raw_covariance_rows=len(complement),seed_bundle_accounts=len(bundles),style_rows=len(style),source_hashes=hashes,
        limits='Exploratory saved exports; date versus initialization uncertainty separate; no dated sectors or neural training predictions supplied'),indent=2)+'\n')


if __name__=='__main__':main()
