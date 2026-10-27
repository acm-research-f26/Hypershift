"""Resumable train-only fit and finite historical proxy study."""
import argparse
import csv
import gzip
import json
from dataclasses import replace,asdict
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.strategy_search.storage import atomic,read,sha,arrays,utc,clean
from experiments.strategy_search.statistics import risk,scales,bootstrap
from experiments.longer_horizon.config import proxy_settings
from .config import STUDY,PRIOR,SCOPES,CATALOG,GATES,COSTS,freeze
from .data import load_market,Quotes,guard_dates
from .signals import features,fit_recipe,candidates
from .replay import run,verify,PATH_COLUMNS
from .audit import fixed_stream
from experiments.strategy_validation.config import Execution


def csv_gz(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        with gzip.open(path,'wt',encoding='utf-8',newline='') as f:f.write('empty\n')
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()})


def daily(m,p,capital):
    rows=[]
    for sid in m['meta']['session_ids']:
        ix=np.flatnonzero(m['sessions']==sid);a,b=ix[0],ix[-1]
        initial=capital if a==0 else p[a-1,1]
        rows.append([sid,p[b,1]/initial-1,p[b,1],p[ix,7].sum(),p[ix,8].sum(),p[ix,9].sum()])
    return np.asarray(rows)


def summarize(m,r,s,ids=None):
    ids=m['evaluate_ids'] if ids is None else ids;p=r['path'];d=daily(m,p,s.capital)
    selected=np.isin(m['sessions'],ids);ds=d[np.isin(d[:,0],ids)];ix=np.flatnonzero(selected)
    initial=s.capital if ix[0]==0 else p[ix[0]-1,1];metrics=risk(ds[:,1],np.r_[initial,p[ix,:2].ravel()])
    trips=[x for x in r['roundtrips'] if x['session'] in ids and x['completed_roundtrip']]
    orders=[o for o in r['orders'] if m['sessions'][o['minute']] in ids]
    notional=sum(o['intended_notional'] for o in orders)
    quoted=sum(o['intended_notional'] for o in orders if o['quote_valid'] is True)
    metrics.update(active_sessions=int(sum(bool((p[m['sessions']==sid,7]>0).any()) for sid in ids)),
        roundtrips=len(trips),turnover_dollars=float(p[ix,7].sum()),fees_dollars=float(p[ix,8].sum()),borrow_dollars=float(p[ix,9].sum()),
        intended_notional=notional,quote_coverage=quoted/notional if notional and any(o['quote_valid'] is not None for o in orders) else None,
        queried_fraction=sum(o['query_complete'] for o in orders)/len(orders) if orders else None,
        risk_failure_minutes=int((p[ix,10:12]>1e-10).any(1).sum()),overnight_failure_sessions=int((p[ix,12]>0).sum()),
        max_abs_net=float(abs(p[ix,4]).max()),filled_shares=sum(abs(o['filled_shares']) for o in orders),
        intended_shares=sum(abs(o['intended_shares']) for o in orders))
    metrics['net_bps_per_traded_dollar']=(p[ix[-1],1]-initial)/metrics['turnover_dollars']*10000 if metrics['turnover_dollars'] else None
    return metrics,d


def training_spreads(m,protocol,output):
    """Past-only full spread estimates, with guarded acquisition if absent."""
    cache=Quotes(m,output,protocol);ids=m['meta']['session_ids'][:60];wanted=[]
    # One fixed middle-of-session snapshot every third training session, all
    # instruments. This is independent of signals/returns and gives20 dates.
    for sid in ids[::3]:
        ix=np.flatnonzero(m['sessions']==sid);i=int(ix[min(150,len(ix)-1)])
        wanted.extend((i,j) for j in range(len(m['symbols'])))
    cache.fetch(wanted)
    values=[[] for _ in m['symbols']]
    for i,j in wanted:
        q=cache.get(i,[j]).get(j)
        if q is not None and 0<=int(m['starts'][i])-int(q[4])<=5e9 and 0<q[0]<=q[1]:
            values[j].append((q[1]-q[0])/((q[0]+q[1])/2)*10000)
    median=np.array([np.median(v) if len(v)>=5 else np.nan for v in values])
    # Missing estimate means no economic gate can approve an entry; never
    # replace missing quotes with a favorable zero spread.
    atomic(Path(output)/'training_spreads.json',dict(training_ids=ids,snapshot_rule='session[::3],minute150,all261',
        spread_bps=median,observations=[len(v) for v in values],missing=[m['symbols'][j] for j in np.flatnonzero(~np.isfinite(median))],
        query_manifest_sha256=sha(cache.root/'manifest.json'),future_exit_spreads_used=False))
    return median


def save_trial(m,stream,config,gate,scenario,settings,spread,output,quote=None,stress='normal'):
    output=Path(output);key='__'.join((m['scope'],config['id'],gate,scenario));root=output/'trials'/key
    receipt=root/'receipt.json'
    if receipt.exists():
        old=read(receipt)
        if old['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('Changed frozen protocol')
        for name,h in old['files'].items():
            if sha(root/name)!=h:raise ValueError('Changed trial output')
        return old
    if shutil.disk_usage(output).free<10_000_000_000:raise RuntimeError('Disk reserve below10GB; resumable stop')
    root.mkdir(parents=True,exist_ok=True);atomic(root/'status.json',dict(status='running',started_UTC=utc(),stream=len(stream)))
    started=time.perf_counter()
    def simulate(scale):return run(m,stream,settings,scale,spread,quote,stress)
    # Query closure includes gate lookups, all intended attempts and required
    # retry paths for BOTH volatility-shadow and scaled books. Negative results
    # are queried results, not presumed missing liquidity.
    passes=0
    while True:
        shadow=simulate(np.ones(len(m['meta']['session_ids'])))
        try:shadow_proof=verify(m,shadow,settings)
        except Exception as error:
            arrays(root/'failed_shadow.npz',equity=shadow['path'],holdings=shadow['holdings'])
            for name in ('fills','orders','decisions','roundtrips'):csv_gz(root/('failed_shadow_'+name+'.csv.gz'),shadow[name])
            atomic(root/'status.json',dict(status='interrupted_verification_failure',updated_UTC=utc(),error=str(error)))
            raise
        shadow_daily=daily(m,shadow['path'],settings.capital)
        scale=scales(shadow_daily);result=simulate(scale)
        try:proof=verify(m,result,settings)
        except Exception as error:
            arrays(root/'failed_scaled.npz',equity=result['path'],holdings=result['holdings'])
            for name in ('fills','orders','decisions','roundtrips'):csv_gz(root/('failed_scaled_'+name+'.csv.gz'),result[name])
            atomic(root/'status.json',dict(status='interrupted_verification_failure',updated_UTC=utc(),error=str(error)))
            raise
        if quote is None:break
        requests=set(map(tuple,shadow['requests']))|set(map(tuple,result['requests']))
        missing=[(i,j) for i,j in requests if j not in quote.get(i,[j])]
        if not missing:break
        quote.fetch(missing);passes+=1
        # This is a correctness failure receipt, not a partially closed result.
        if passes>len(m['sessions']):raise RuntimeError('Quote request closure failed to converge within finite minute universe')
    metrics,d=summarize(m,result,settings)
    metrics['accounting_valid']=all(v[k] for v in (proof,shadow_proof) for k in ('accounting_reconciled','fill_risk_valid','open_close_risk_valid','no_overnight','no_action_failures'))
    blocks={label:summarize(m,result,settings,p['session_ids'])[0] for label,p in read(output/'protocol.json')['binding']['partitions'][m['scope']].items() if label!='warmup'}
    metrics['uncertainty']=bootstrap(d[np.isin(d[:,0],m['evaluate_ids']),1])
    arrays(root/'minute_tapes.npz',equity=result['path'],holdings=result['holdings'],scale=scale,shadow_path=shadow['path'],
        shadow_holdings=shadow['holdings'],daily=d,shadow_daily=shadow_daily,starts=m['starts'],sessions=m['sessions'])
    for name in ('fills','orders','decisions','roundtrips'):
        csv_gz(root/(name+'.csv.gz'),result[name]);csv_gz(root/('shadow_'+name+'.csv.gz'),shadow[name])
    csv_gz(root/'daily.csv.gz',[dict(session=int(v[0]),daily_return=v[1],nav=v[2],turnover=v[3],fees=v[4],borrow=v[5]) for v in d])
    files={p.name:sha(p) for p in root.iterdir() if p.name not in ('receipt.json','status.json')}
    record=dict(key=key,scope=m['scope'],config=config,gate=gate,scenario=scenario,mode='quotes' if quote else 'minute_open_proxy',
        stream_candidates=len(stream),metrics=metrics,blocks=blocks,verification=proof,shadow_verification=shadow_proof,
        protocol_sha256=sha(output/'protocol.json'),files=files,root=str(root),path_columns=PATH_COLUMNS,symbols=m['symbols'],
        settings=asdict(settings),query_closure_passes=passes,seconds=time.perf_counter()-started,completed_UTC=utc())
    atomic(receipt,record);atomic(root/'status.json',dict(status='complete',completed_UTC=utc()))
    print('TRIAL',key,'return',metrics['total_return'],'SR',metrics['Sharpe'],'trips',metrics['roundtrips'],'valid',metrics['accounting_valid'],flush=True)
    return record


def select(rows,output):
    candidates_out=[];diagnostics=[]
    for c in CATALOG:
        dev=[r for r in rows if r['scope'] in SCOPES[:2] and r['config']['id']==c['id'] and r['gate']=='net_edge' and r['scenario']=='cost1']
        if len(dev)!=2:raise ValueError('Both development cohorts must finish before selection')
        blocks=[b for r in dev for b in r['blocks'].values()]
        defined=len(blocks)==5 and all(b['Sharpe'] is not None for b in blocks)
        median=float(np.median([b['Sharpe'] for b in blocks])) if defined else None
        activity=len(blocks)==5 and min(b['roundtrips'] for b in blocks)>=10 and sum(r['metrics']['roundtrips'] for r in dev)>=100 and sum(r['metrics']['active_sessions'] for r in dev)>=40
        valid=all(r['metrics']['accounting_valid'] for r in dev)
        candidates_out.append(dict(id=c['id'],design=c['design'],median_Sharpe=median,activity=bool(activity),valid=valid,
            active_sessions=sum(r['metrics']['active_sessions'] for r in dev),roundtrips=sum(r['metrics']['roundtrips'] for r in dev),
            block_roundtrips=[b['roundtrips'] for b in blocks],
            development_qualified=bool(valid and activity and median is not None and median>=1 and all(r['metrics']['total_return']>0 for r in dev))))
    for design in ('D01','D03','D04'):
        possible=[r for r in candidates_out if r['design']==design and r['valid'] and r['activity'] and r['median_Sharpe'] is not None]
        if possible:diagnostics.append(max(possible,key=lambda r:(r['median_Sharpe'],r['id']))['id'])
    result=dict(frozen_UTC=utc(),candidates=candidates_out,diagnostics=diagnostics,shortlist=[r['id'] for r in candidates_out if r['development_qualified']],
        only_development_used=True,later_exploratory=True,rule='One eligible diagnostic per design by frozen activity/risk and median1bp; no positivity required for diagnostic')
    dest=Path(output)/'selection.json'
    if dest.exists():
        old=read(dest);result['frozen_UTC']=old['frozen_UTC']
        if old!=result:raise ValueError('Changed shortlist')
    else:atomic(dest,result)
    return result


def proxy_study(output,protocol):
    output=Path(output);rows=[];fits={};spread=None
    fit_pickle=output/'fits'/'models.pkl'
    # Local trusted serialized parameters only; never accept external pickle.
    import pickle
    if fit_pickle.exists():
        audit=read(output/'fits'/'manifest.json')
        if sha(fit_pickle)!=audit['sha256']:raise ValueError('Changed fitted parameters')
        with fit_pickle.open('rb') as f:fits=pickle.load(f)
        spread=np.array(read(output/'training_spreads.json')['spread_bps'],float)
    for scope in SCOPES:
        if scope==SCOPES[2]:select(rows,output)
        m=load_market(scope,protocol,output);f=features(m)
        if scope==SCOPES[0] and not fits:
            spread=training_spreads(m,protocol,output);dest=output/'fits';dest.mkdir(parents=True,exist_ok=True)
            for c in CATALOG:
                fits[c['id']]=fit_recipe(m,f,c,dest,spread)
                print('FIT',c['id'],fits[c['id']]['valid'],flush=True)
            with fit_pickle.open('wb') as file:pickle.dump(fits,file)
            atomic(dest/'manifest.json',dict(sha256=sha(fit_pickle),protocol_sha256=sha(output/'protocol.json'),train_ids=m['meta']['session_ids'][:60]))
        if scope==SCOPES[0] and not (output/'engineering'/'train_streams.json').exists():
            train_stream_audit(m,f,fits,spread,output)
        for c in CATALOG:
            gates=GATES if c['id'] in ('D01','D03','D04') else ('net_edge',)
            for gate in gates:
                stream=candidates(m,f,fits[c['id']],c,gate)
                csv_gz(output/'candidates'/(scope+'__'+c['id']+'__'+gate+'.csv.gz'),stream)
                for cost in COSTS:
                    r=save_trial(m,stream,c,gate,'cost'+format(cost,'g'),proxy_settings(cost),spread,output)
                    rows.append(r)
                    atomic(output/'progress.json',dict(stage='proxy_study',updated_UTC=utc(),completed_proxy_trials=len(rows),budget=315,last_trial=r['key']))
        del m,f
    atomic(output/'proxy_results.json',dict(rows=rows,trials=len(rows),protocol_sha256=sha(output/'protocol.json')))
    csv_gz(output/'proxy_ledger.csv.gz',[dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],**r['metrics']) for r in rows])
    return rows


def train_stream_audit(m,f,fits,spread,output):
    """Validate actual fixed streams on train60 before any development run."""
    train=m['meta']['session_ids'][:60];stop=int(np.flatnonzero(m['sessions']==train[-1])[-1])+1
    small={k:(v[:stop] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(m['sessions']) else v) for k,v in m.items()}
    small['meta']=dict(m['meta'],session_ids=train);small['evaluate_ids']=train
    proofs=[];root=Path(output)/'engineering';root.mkdir(parents=True,exist_ok=True)
    for c in [x for x in CATALOG if x['id'] in ('D01','D03','D04')]:
        stream=candidates(small,f,fits[c['id']],c,'no_gate')
        for stress in ('normal','long_first','short_first','hedge_unavailable'):
            r=run(small,stream,Execution(),np.ones(60),spread,stress=stress)
            key=c['id']+'_'+stress;arrays(root/(key+'.npz'),equity=r['path'],holdings=r['holdings'])
            for name in ('fills','orders','decisions','roundtrips'):csv_gz(root/(key+'_'+name+'.csv.gz'),r[name])
            proof=verify(small,r,Execution());proofs.append(dict(id=c['id'],stress=stress,verification=proof,filled_children=len(r['fills'])))
            print('TRAIN_EXECUTION_AUDIT',key,'reconciled',proof['accounting_reconciled'],'fill_risk',proof['fill_risk_valid'],'mark_risk',proof['open_close_risk_valid'],flush=True)
    atomic(root/'train_streams.json',dict(completed_UTC=utc(),training_only=True,not_performance_selection=True,
        training_ids=train,proofs=proofs,all_accounting_reconciled=all(p['verification']['accounting_reconciled'] for p in proofs)))


def quote_study(output,protocol):
    import pickle
    output=Path(output);selection=read(output/'selection.json');source=output/'fits'/'models.pkl'
    if sha(source)!=read(output/'fits'/'manifest.json')['sha256']:raise ValueError('Changed fit')
    with source.open('rb') as file:fits=pickle.load(file)
    spread=np.asarray(read(output/'training_spreads.json')['spread_bps'],float);rows=[]
    settings=Execution()
    scenarios=[('primary',settings,'normal'),('commission_free',replace(settings,commission_free=True),'normal'),
        ('adverse_0p5',replace(settings,adverse_bps=.5),'normal'),('long_first',settings,'long_first'),
        ('short_first',settings,'short_first'),('hedge_unavailable',settings,'hedge_unavailable'),
        ('age60_sensitivity',replace(settings,quote_max_age_seconds=60),'normal')]
    for scope in SCOPES:
        if not selection['diagnostics']:break
        m=load_market(scope,protocol,output);f=features(m);quotes=Quotes(m,output,protocol)
        for identifier in selection['diagnostics']:
            c=next(x for x in CATALOG if x['id']==identifier);stream=candidates(m,f,fits[identifier],c)
            for scenario,setting,stress in scenarios:
                r=save_trial(m,stream,c,'net_edge',scenario,setting,spread,output,quotes,stress)
                rows.append(r);atomic(output/'progress.json',dict(stage='quote_study',updated_UTC=utc(),completed_quote_trials=len(rows),last_trial=r['key']))
        del m,f,quotes
    atomic(output/'quote_results.json',dict(rows=rows,trials=len(rows),diagnostics=selection['diagnostics'],
        empty_set_is_not_coverage=not selection['diagnostics']))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','engineering','proxies','quotes']);p.add_argument('--output',default=str(STUDY/'decision_strategy_execution_2026-10-27_v7'))
    a=p.parse_args();output=Path(a.output)
    if a.command=='engineering':print(fixed_stream(output/'engineering'));return
    protocol=freeze(output)
    if a.command=='freeze':print('FROZEN',sha(output/'protocol.json'));return
    if a.command=='proxies':proxy_study(output,protocol)
    else:quote_study(output,protocol)


if __name__=='__main__':main()
