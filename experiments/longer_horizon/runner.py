"""Bounded screen, frozen development selection, and complete quote replay."""
import argparse
from dataclasses import asdict
import csv
import hashlib
from pathlib import Path
import time
import numpy as np

from experiments.strategy_search.storage import read,atomic,arrays,sha,identity
from experiments.strategy_search.statistics import scales
from experiments.strategy_validation.analysis import normalized_summary,dollar_daily
from experiments.closing_followup.runner import verify
from .config import freeze,SCOPES,CATALOG,FEES,proxy_settings,quote_scenarios
from .data import stock_market,QuoteCache
from .signals import fit,sampled_features,build
from .engine import replay,PATH_COLUMNS,ORDER_COLUMNS


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def simulate(m,base,settings,quotes):
    shadow,shadow_orders,shadow_failures=replay(m,base,np.ones(len(m['meta']['session_ids'])),settings,quotes)
    shadow_daily=dollar_daily(shadow,m['sessions'],settings.capital)
    shadow_proof=verify(m,shadow,shadow_orders,settings,quotes)
    scale=scales(shadow_daily)
    del shadow
    path,orders,failures=replay(m,base,scale,settings,quotes)
    proof=verify(m,path,orders,settings,quotes)
    return dict(path=path,orders=orders,failures=failures,scale=scale,shadow_daily=shadow_daily,
        shadow_orders=shadow_orders,shadow_failures=shadow_failures,verification=proof,shadow_verification=shadow_proof)


def record(m,base,config,name,mode,settings,output,market_digest,simulation=None,cache=None):
    output=Path(output)
    binding=dict(protocol_sha256=sha(output/'protocol.json'),scope=m['scope'],config=config,scenario=name,
        mode=mode,settings=asdict(settings),base_sha256=digest(base),market_sha256=market_digest,
        quote_manifest_sha256=sha(cache.root/'manifest.json') if cache is not None else None)
    binding=__import__('json').loads(__import__('json').dumps(binding))
    key=identity(binding)[:24];dest=output/'trials'/(key+'.json');archive=dest.with_suffix('.npz')
    if dest.exists():
        d=read(dest)
        if d['binding']!=binding or sha(archive)!=d['archive_sha256']:raise ValueError('Changed trial archive')
        return d
    began=time.perf_counter()
    s=simulation if simulation is not None else simulate(m,base,settings,cache.values if cache is not None else None)
    if cache is not None and (cache.missing(s['orders']) or cache.missing(s['shadow_orders'])):
        raise ValueError('An intended quote request has never been queried')
    path,orders=s['path'],s['orders'];metrics,daily=normalized_summary(m,path,orders,s['failures'],settings.capital)
    mask=np.isin(m['sessions'],m['evaluate_ids']);ix=np.flatnonzero(mask)
    metrics.update(active_sessions=sum(bool((path[m['sessions']==sid,7]>0).any()) for sid in m['evaluate_ids']),
        maximum_net_weight=float(abs(path[mask,4]).max()),overnight_failures=int((path[mask,16]>1e-8).sum()),
        evaluation_constraint_failures=int((path[mask,14]>1e-7).sum()),coverage_passed=metrics['quote_coverage']>=.99,
        shadow_failures=s['shadow_failures'],query_completeness=1. if cache is not None else None)
    initial=settings.capital if ix[0]==0 else path[ix[0]-1,1]
    metrics['net_bps_per_traded_dollar']=float((path[ix[-1],1]-initial)/metrics['turnover_dollars']*10000) if metrics['turnover_dollars'] else None
    # A failed unscaled run cannot supply a trustworthy volatility schedule.
    metrics['accounting_valid'] &= s['shadow_failures']==0
    blocks={}
    for label,partition in read(output/'protocol.json')['binding']['partitions'][m['scope']].items():
        if label!='warmup':blocks[label],_=normalized_summary(dict(m,evaluate_ids=partition['session_ids']),path,orders,s['failures'],settings.capital)
    arrays(archive,equity=path,orders=orders,daily=daily,scales=s['scale'],shadow_daily=s['shadow_daily'],shadow_orders=s['shadow_orders'])
    d=dict(key=key,binding=binding,scope=m['scope'],config=config,scenario=name,mode=mode,metrics=metrics,blocks=blocks,
        verification=s['verification'],shadow_verification=s['shadow_verification'],archive=str(archive),archive_sha256=sha(archive),
        path_columns=PATH_COLUMNS,order_columns=ORDER_COLUMNS,seconds=time.perf_counter()-began)
    atomic(dest,d)
    print(f'TRIAL {m["scope"]} {config["id"]} {name}: SR={metrics["Sharpe"]} return={metrics["total_return"]} valid={metrics["accounting_valid"]} coverage={metrics["quote_coverage"]:.4f}',flush=True)
    return d


def select(rows,output):
    candidates=[]
    for c in CATALOG:
        dev=[r for r in rows if r['config']['id']==c['id'] and r['mode']=='minute_proxy' and r['scenario']=='cost1' and r['scope'] in SCOPES[:2]]
        if len(dev)!=2:raise ValueError('Both development cohorts must finish before selection')
        block=[b['Sharpe'] for r in dev for b in r['blocks'].values()]
        median=float(np.median(block)) if len(block)==5 and all(s is not None for s in block) else None
        valid=all(r['metrics']['accounting_valid'] for r in dev)
        active=sum(r['metrics']['active_sessions'] for r in dev)
        qualified=valid and median is not None and median>=1 and active>=20 and all(r['metrics']['total_return']>0 for r in dev)
        candidates.append(dict(id=c['id'],family=c['family'],median_development_block_Sharpe=median,
            valid=valid,active_sessions=active,development_qualified=bool(qualified)))
    shortlist=[]
    for group in ('graph','self'):
        possible=[c for c in candidates if c['valid'] and c['median_development_block_Sharpe'] is not None and
            ((c['family'].startswith('self_'))==(group=='self')) and c['family']!='shuffled_two_hop']
        if possible:
            best=max(possible,key=lambda c:(c['development_qualified'],c['median_development_block_Sharpe'],-CATALOG.index(next(x for x in CATALOG if x['id']==c['id']))))
            shortlist.append(best['id'])
    d=dict(candidates=candidates,shortlist=shortlist,selection_only_uses=list(SCOPES[:2]),
        rule='Best qualified graph and self control; otherwise best accounting-valid diagnostic with defined five-block median',
        later_data_previously_exposed=True)
    dest=Path(output)/'selection.json'
    if dest.exists() and read(dest)!=d:raise ValueError('Changed development selection')
    atomic(dest,d);return d


def ledger(rows,output):
    fields=['scope','strategy','family','mode','scenario','Sharpe','total_return','maximum_drawdown','accounting_valid',
        'quote_coverage','coverage_passed','active_sessions','executed_orders','turnover_dollars','explicit_cost_bps',
        'net_bps_per_traded_dollar','maximum_net_weight','overnight_failures','evaluation_constraint_failures','query_completeness']
    with (Path(output)/'trial_ledger.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(scope=r['scope'],strategy=r['config']['id'],family=r['config']['family'],
            mode=r['mode'],scenario=r['scenario'],**{k:r['metrics'].get(k) for k in fields[5:]}))


def load_fit(output):
    audit=read(Path(output)/'fit.json')
    if sha(Path(output)/'fit.npz')!=audit['fit_sha256']:raise ValueError('Changed fit')
    with np.load(Path(output)/'fit.npz') as z:return {k:z[k] for k in z.files}


def screen(a,protocol):
    rows=[];fitted=None
    for scope in SCOPES:
        if scope==SCOPES[2]:select(rows,a.output)
        m,original=stock_market(a.reference,scope,protocol);del original
        if scope==SCOPES[0]:fitted=fit(m,a.graph,a.output)
        features=sampled_features(m,fitted)
        market_digest={k:digest(m[k]) for k in ('open','close','volume','actions','starts','sector')}
        for c in CATALOG:
            base,decisions=build(m,features,c)
            atomic(Path(a.output)/'decisions'/(scope+'_'+c['id']+'.json'),dict(config=c,decisions=decisions))
            for bps in FEES:
                rows.append(record(m,base,c,f'cost{bps:g}','minute_proxy',proxy_settings(bps),a.output,market_digest))
                ledger(rows,a.output)
        del m,features,base
    atomic(Path(a.output)/'screen.json',dict(rows=rows,trials=len(rows),selection=read(Path(a.output)/'selection.json'),
        historical_data_status='Previously exposed research development, not new confirmatory data',protected_future_sessions_accessed=False))
    return rows


def quote_round(a,protocol):
    selection=read(Path(a.output)/'selection.json');fitted=load_fit(a.output);rows=[]
    selected=[c for c in CATALOG if c['id'] in selection['shortlist']]
    for scope in SCOPES:
        m,original=stock_market(a.reference,scope,protocol);del original
        features=sampled_features(m,fitted);cache=QuoteCache(m,a.output,a.prior,a.env_file)
        bases={c['id']:build(m,features,c)[0] for c in selected}
        md={k:digest(m[k]) for k in ('open','close','volume','actions','starts','sector')}
        # Adaptive sampling changes quote availability, never a signal or policy.
        for iteration in range(1,101):
            missing=set()
            for c in selected:
                for name,settings in quote_scenarios():
                    s=simulate(m,bases[c['id']],settings,cache.values)
                    missing.update(cache.missing(s['orders']));missing.update(cache.missing(s['shadow_orders']))
                    del s
            print(f'QUOTE_CLOSURE {scope} iteration={iteration} missing_stock_times={len(missing)}',flush=True)
            if not missing:break
            cache.fetch(sorted(missing))
        else:raise RuntimeError('Quote closure did not converge; do not report this cohort complete')
        cache.save()
        atomic(Path(a.output)/'quote_closure'/(scope+'.json'),dict(complete=True,iterations=iteration,
            stock_times=int(cache.known.sum()),manifest_sha256=sha(cache.root/'manifest.json'),all_frozen_scenarios=True))
        for c in selected:
            for name,settings in quote_scenarios():
                rows.append(record(m,bases[c['id']],c,name,'quotes',settings,a.output,md,cache=cache))
        del m,features,cache,bases
    atomic(Path(a.output)/'quote_results.json',dict(rows=rows,trials=len(rows),selection=selection))
    return rows


def finish(a):
    from .report import render
    screen_doc=read(Path(a.output)/'screen.json');quote_doc=read(Path(a.output)/'quote_results.json')
    rows=screen_doc['rows']+quote_doc['rows'];ledger(rows,a.output);qualified=[];gates=[]
    for c in quote_doc['selection']['candidates']:
        if c['id'] not in quote_doc['selection']['shortlist']:continue
        proxy=next(r for r in rows if r['config']['id']==c['id'] and r['scope']==SCOPES[2] and r['scenario']=='cost1')
        primary=[r for r in rows if r['config']['id']==c['id'] and r['mode']=='quotes' and r['scenario']=='primary']
        stress=[r for r in rows if r['config']['id']==c['id'] and r['mode']=='quotes' and r['scenario']=='adverse_0p5']
        later=next(r for r in primary if r['scope']==SCOPES[2])
        valid=c['development_qualified'] and proxy['metrics']['accounting_valid'] and proxy['metrics']['total_return']>0 and (proxy['metrics']['Sharpe'] or -999)>=1
        valid &= len(primary)==3 and all(r['metrics']['accounting_valid'] and r['metrics']['coverage_passed'] and r['metrics']['total_return']>0 for r in primary)
        valid &= (later['metrics']['Sharpe'] or -999)>=1 and len(stress)==3 and all(r['metrics']['accounting_valid'] and r['metrics']['total_return']>0 for r in stress)
        gates.append(dict(id=c['id'],qualified=bool(valid),development_qualified=c['development_qualified']))
        if valid:qualified.append(c['id'])
    atomic(Path(a.output)/'summary.json',dict(rows=rows,trials=len(rows),gates=gates,qualified=qualified,
        all_accounting_reconciled=all(r['verification']['passed'] and r['shadow_verification']['passed'] for r in rows),
        protected_future_sessions_accessed=False,all_historical_cohorts_previously_exposed=True,
        universe=250,training_sessions_excluded_from_reported_performance=60))
    render(Path(a.output))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['screen','quotes','auction','finish','report'])
    for name in ['reference','prior','graph','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--env-file',default='.env');a=p.parse_args()
    protocol=freeze(a.reference,a.prior,a.graph,a.output)
    if a.command=='screen':screen(a,protocol)
    elif a.command=='quotes':quote_round(a,protocol)
    elif a.command=='auction':
        from .auction import run
        run(a,protocol)
    elif a.command=='finish':finish(a)
    else:
        from .report import render
        render(Path(a.output))
