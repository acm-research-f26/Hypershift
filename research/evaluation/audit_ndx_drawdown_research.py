"""Independent risk formulas and unit-account replay for drawdown research."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_ndx_drawdown_research as study
from ndx_drawdown_engine import allocations as production_allocations

ROOT, RUN = study.ROOT, study.RUN
ALIASES = {'GOOG':'ALPHABET','GOOGL':'ALPHABET','FOX':'FOX','FOXA':'FOX',
           'LBTYA':'LIBERTY_GLOBAL','LBTYK':'LIBERTY_GLOBAL'}


def independent_weights(data, ids, scores, allocation):
    result = np.zeros_like(scores)
    names = data['panel'].tickers
    for row,i in enumerate(ids):
        eligible = np.flatnonzero(data['mask'][i])
        if allocation in ['equal5','equal10']:
            k = 5 if allocation == 'equal5' else 10
            cutoff = sorted(scores[row,eligible],reverse=True)[min(k,len(eligible))-1]
            above = [j for j in eligible if scores[row,j] > cutoff]
            tied = [j for j in eligible if scores[row,j] == cutoff]
            result[row,above] = 1/k
            result[row,tied] = (k-len(above))/len(tied)/k
            continue
        grouped = {}
        for j in eligible:
            grouped.setdefault(ALIASES.get(names[j],names[j]),[]).append(j)
        maximum = {key:max(scores[row,j] for j in values) for key,values in grouped.items()}
        k = 8 if allocation == 'issuer8_soft' else 5
        cutoff = sorted(maximum.values(),reverse=True)[min(k,len(maximum))-1]
        n_above = sum(v > cutoff for v in maximum.values())
        n_tied = sum(v == cutoff for v in maximum.values())
        for key,values in grouped.items():
            v = maximum[key]
            budget = 1/k if v > cutoff else (k-n_above)/n_tied/k if v == cutoff else 0.
            best = [j for j in values if scores[row,j] == v]
            result[row,best] = budget/len(best)
        if allocation not in ['issuer_invvol5','issuer8_soft']:
            continue
        raw = result[row].copy()
        active = np.flatnonzero(raw > 0)
        if allocation == 'issuer_invvol5':
            for j in active:
                history = data['returns'][max(0,data['origin'][i]-19):data['origin'][i]+1,j]
                finite = history[np.isfinite(history)]
                vol = float(finite.std(ddof=1)*np.sqrt(252)) if len(finite) >= 15 else .30
                raw[j] /= max(vol,.10)
            cap = .30
        else:
            raw[active] *= np.exp(4*(scores[row,active]-scores[row,active].max()))
            cap = .25
        # Bisection solves sum(min(cap, scale*raw))=1 independently of waterfill.
        lo,hi = 0.,1/max(raw[active].min(),1e-12)
        cap = max(cap,1/len(active))
        for _ in range(90):
            mid = (lo+hi)/2
            if np.minimum(cap,mid*raw[active]).sum() > 1:hi=mid
            else:lo=mid
        result[row] = np.minimum(cap,((lo+hi)/2)*raw)
    return result


def independent_exposure(data,ids,weights,config):
    q = pd.read_csv(ROOT/data['manifest']['root']/'etf_adj_close.csv',index_col=0)['QQQ'].to_numpy()
    returns = np.r_[np.nan,q[1:]/q[:-1]-1]
    schedule = {int(data['origin'][i]):row for row,i in enumerate(ids)}
    w = np.zeros(weights.shape[1]); out = np.ones(len(q))
    for day in range(len(q)):
        if day in schedule:w=weights[schedule[day]]
        exposure = config.get('constant',1.)
        if 'ma' in config:
            length=config['ma'];moving=np.mean(q[day-length+1:day+1]) if day>=length-1 else np.nan
            if q[day]<moving:exposure=min(exposure,config['off'])
        if config.get('multitrend'):
            healthy = [q[day]>=np.mean(q[day-k+1:day+1]) if day>=k-1 else False for k in [50,100,200]]
            exposure=min(exposure,.25+.75*np.mean(healthy))
        if 'vol' in config:
            length=config['window']
            if config['vol']=='basket':
                vol=.30
                if day>=length-1 and w.any():
                    active=np.flatnonzero(w)
                    observations=np.array([sum(float(np.nan_to_num(data['returns'][t,j],nan=0.))*w[j] for j in active) for t in range(day-length+1,day+1)])
                    vol=max(float(observations.std(ddof=1)*np.sqrt(252)),.05)
            else:
                vol=float(np.std(returns[day-length+1:day+1],ddof=1)*np.sqrt(252)) if day>=length else np.nan
            exposure=min(exposure,config['target']/max(vol,.05))
        if 'market_dd' in config and day>=199:
            drawdown=q[day]/np.max(q[max(0,day-251):day+1])-1
            if drawdown < -config['market_dd']:exposure=min(exposure,config['off'])
        out[day]=np.clip(exposure,0,1) if np.isfinite(exposure) else .5
    return out


def independent_account(data,ids,weights,exposure,cadence,cost):
    dates=data['panel'].dates;quotes=data['marks'];count=len(data['panel'].tickers)
    shares=np.zeros(count);bank=previous=1e6;values=[];last_exposure=None;base=np.zeros(count)
    schedule={int(data['origin'][i]):row for row,i in enumerate(ids)}
    actions={str(e['date']):e for e in data['events']}
    first=int(data['entry'][ids[0]]);last=len(dates)-1
    def fees(dollars,adv):
        participation=np.abs(dollars)/np.maximum(np.nan_to_num(adv,nan=1.),1.)
        return float(np.sum(np.abs(dollars)*(.0007+.0003*np.sqrt(participation/.01)))*cost)
    for day in range(first,last+1):
        origin=day-1;quantities=None
        weekly=origin in schedule
        if weekly:base=weights[schedule[origin]].copy()
        changed=last_exposure is None or abs(exposure[origin]-last_exposure)>=.10-1e-12
        if day!=last and (weekly or (cadence=='daily' and changed)):
            last_exposure=exposure[origin]
            target=np.zeros(count);cap=np.zeros(count)
            for j in range(count):
                if np.isfinite(quotes[origin,j]) and quotes[origin,j]>0:
                    target[j]=base[j]*last_exposure*previous/quotes[origin,j]
                    cap[j]=.01*np.nan_to_num(data['adv'][origin,j])/quotes[origin,j]
            quantities=np.maximum(np.minimum(target-shares,cap),-cap)
        action=actions.get(str(dates[day].date()))
        if action:
            j=data['panel'].tickers.index(action['security_id']);value=shares[j]*quotes[day,j]
            stock=value*action['stock_fraction_at_completion']
            bank+=value-fees(np.array([stock]),np.array([data['adv'][day-1,j]]));shares[j]=0.
        if quantities is not None:
            quantities=np.where(data['tradable'][day],quantities,0.)
            quantities=np.maximum(quantities,-shares)
            traded=np.nan_to_num(quantities*quotes[day]);sell=np.minimum(traded,0.);buy=np.maximum(traded,0.)
            available=bank-sell.sum()-fees(sell,data['adv'][origin])
            need=buy.sum()+fees(buy,data['adv'][origin])
            if need>available and need>0:
                quantities[traded>0]*=max(0.,available/need)
                traded=np.nan_to_num(quantities*quotes[day])
            bank-=traded.sum()+fees(traded,data['adv'][origin]);shares+=quantities
        if day==last:
            dollars=np.nan_to_num(shares*quotes[day]);bank+=dollars.sum()-fees(dollars,data['adv'][day-1]);shares[:]=0.
        previous=float(bank+np.nan_to_num(shares*quotes[day]).sum())
        if bank < -1e-5 or previous<=0:raise ValueError('Invalid independent cash account')
        values.append(previous)
    return np.array(values)


def main():
    started=time.monotonic()
    plan=json.loads((RUN/'PROTOCOL.json').read_text());complete=json.loads((RUN/'COMPLETE.json').read_text())
    for section in ['source_sha256','input_sha256']:
        for path,h in plan[section].items():
            if study.sha(ROOT/path)!=h:raise ValueError('Frozen input changed: '+path)
    for path,h in complete['output_sha256'].items():
        if study.sha(RUN/path)!=h:raise ValueError('Completed batch changed: '+path)
    with np.load(RUN/'paths.npz') as f:
        path_ids=f['ids'].tolist();returns=f['returns'].copy();dates=pd.DatetimeIndex(f['dates'])
    metrics=pd.read_csv(RUN/'metrics.csv').set_index('id')
    for column,key in enumerate(path_ids):
        a=returns[:,column];wealth=np.r_[1,np.cumprod(1+a)]
        checks=dict(sharpe=np.sqrt(252)*(a.mean()-(1.03**(1/252)-1))/a.std(ddof=1),
                    annualized_return=wealth[-1]**(252/len(a))-1,
                    maximum_drawdown=np.min(wealth/np.maximum.accumulate(wealth)-1))
        for name,value in checks.items():np.testing.assert_allclose(value,metrics.loc[key,name],atol=1e-11,rtol=1e-11)
    print(f'All {len(metrics)} account metrics independently recalculated; source/input locks valid',flush=True)
    data=study.load_data();ids=np.concatenate([study.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    scores=study.saved_scores(data,ids)
    configs={c['id']:c for c in plan['recipes']}
    stress=pd.read_csv(RUN/'cost_stress.csv')
    targets={}
    for group in study.GROUPS:
        for allocation in sorted({c['allocation'] for c in configs.values()}):
            w=independent_weights(data,ids,scores[group],allocation)
            np.testing.assert_allclose(w,production_allocations(data,ids,scores[group],allocation),atol=1e-12,rtol=1e-12)
            targets[group,allocation]=w
    print('All allocation recipes independently reconstructed for all three seed groups',flush=True)
    largest=0.;accounts=0
    for recipe in complete['shortlist']:
        config=configs[recipe]
        for group in study.GROUPS:
            weights=targets[group,config['allocation']]
            exposure=independent_exposure(data,ids,weights,config)
            for cost in [1,2,4]:
                nav=independent_account(data,ids,weights,exposure,config['cadence'],cost)
                expected=stress[(stress.recipe==recipe)&(stress.group==group)&(stress.cost_multiplier==cost)].iloc[0]
                ret=nav/np.r_[1e6,nav[:-1]]-1
                np.testing.assert_allclose(nav[-1],expected.terminal_nav,atol=1e-5,rtol=1e-11)
                np.testing.assert_allclose(np.sqrt(252)*(ret.mean()-(1.03**(1/252)-1))/ret.std(ddof=1),expected.sharpe,atol=1e-10)
                if cost==1:
                    original=1e6*np.cumprod(1+returns[:,path_ids.index(recipe+'/'+group)])
                    largest=max(largest,float(np.abs(nav-original).max()))
                    np.testing.assert_allclose(nav,original,atol=1e-5,rtol=1e-11)
                accounts+=1
        print('Independent close-fill accounts verified:',recipe,flush=True)
    # Retain an additional real-data audit of the best daily-risk configuration.
    daily=metrics[metrics.recipe.str.endswith('_daily') & (metrics.group=='ten_seed')].sort_values('sharpe',ascending=False).iloc[0]
    config=configs[daily.recipe];w=targets['ten_seed',config['allocation']]
    ex=independent_exposure(data,ids,w,config)
    nav=independent_account(data,ids,w,ex,'daily',1)
    expected=1e6*np.cumprod(1+returns[:,path_ids.index(daily['id'] if 'id' in daily else daily.name)])
    np.testing.assert_allclose(nav,expected,atol=1e-5,rtol=1e-11)
    largest=max(largest,float(np.abs(nav-expected).max()));accounts+=1
    output=dict(status='passed',audited_account_metrics=len(metrics),
                allocation_seed_group_combinations=len(targets),independent_replays=accounts,
                maximum_nav_difference_dollars=largest,extra_daily_recipe=daily.recipe,
                source_input_and_output_locks_verified=True,seconds=time.monotonic()-started,
                audit_source_sha256=study.sha(ROOT/'research/evaluation/audit_ndx_drawdown_research.py'),
                scope='Accounting and causal implementation checks; no independent future-performance confirmation')
    study.write(RUN/'AUDIT.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
