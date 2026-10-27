"""Freeze, run, verify and resume the bounded follow-up study."""
import argparse
from dataclasses import asdict
import csv
import hashlib
from pathlib import Path
import time
import numpy as np

from experiments.strategy_search.storage import atomic, arrays, read, sha, identity
from experiments.strategy_search.statistics import scales
from experiments.strategy_validation.analysis import normalized_summary, dollar_daily
from .config import freeze, SCOPES, CATALOG, scenarios
from .data import stock_market, stock_quotes, load_etfs, etf_quotes
from .signals import fit, build_stock, build_etf
from .engine import replay, PATH_COLUMNS, ORDER_COLUMNS


def verify(m,path,orders,settings,quotes):
    """Independent cash/NAV and displayed-price/size checks on every recorded minute."""
    t=len(path);used={};parents={};fills=orders[orders[:,5]!=0]
    trade_cash=np.zeros(t);open_value=np.zeros(t);close_value=np.zeros(t);short_value=np.zeros(t)
    for o in fills:
            i,j=map(int,o[:2]);amount=o[5]
            assert amount==np.trunc(amount)
            assert abs(amount)<=np.floor(m['volume'][i-1,j]*settings.participation)+1e-9
            if quotes is not None:
                bid,ask,bs,az,stamp=quotes[i,j];age=int(m['starts'][i])-int(stamp)
                assert 0<=age<=settings.quote_max_age_seconds*1e9
                expected=(ask if amount>0 else bid)*(1+np.sign(amount)*settings.adverse_bps/10000)
                np.testing.assert_allclose(o[7],expected,rtol=1e-12,atol=1e-10)
                key=(j,int(stamp),int(np.sign(amount)));used[key]=used.get(key,0.)+abs(amount)
                assert used[key]<=(az if amount>0 else bs)+1e-9
            parent=int(o[2]);old_qty,old_notional,old_fees=parents.get(parent,(0.,0.,np.zeros(4)))
            qty=old_qty+abs(amount);notional=old_notional+abs(amount*o[7])
            commission=0. if settings.commission_free else min(max(settings.commission_minimum,
                settings.commission_share*qty),settings.commission_maximum_fraction*notional)
            regulatory=commission*settings.commission_pass_through
            if amount<0:regulatory+=settings.sec_sell_fraction*notional+min(settings.finra_sell_cap,qty*settings.finra_sell_share)
            cumulative=np.array([commission,qty*settings.exchange_taker_share,
                qty*(settings.clearing_share+settings.cat_share),regulatory])
            np.testing.assert_allclose(o[10:14],cumulative-old_fees,rtol=1e-11,atol=1e-9)
            parents[parent]=(qty,notional,cumulative)
            trade_cash[i]+=amount*o[7]+o[10:14].sum()
    # Forward-fill alternating open/close marks separately for each instrument.
    # Vectorized assertions replace repeated assertion setup, not accounting rules.
    for j in range(len(m['symbols'])):
        delta=np.zeros(t);selected=fills[fills[:,1]==j]
        np.add.at(delta,selected[:,0].astype(int),selected[:,5]);held=np.cumsum(delta)
        raw=np.empty(2*t);raw[::2]=m['open'][:,j];raw[1::2]=m['close'][:,j]
        good=np.isfinite(raw)&(raw>0);indices=np.maximum.accumulate(np.where(good,np.arange(2*t)+1,0))
        marks=np.r_[0.,raw][indices]
        open_value+=held*marks[::2];close_value+=held*marks[1::2]
        short_value+=np.maximum(-held*marks[1::2],0)
    borrow=short_value*settings.borrow/(365*1440.)
    cash=settings.capital-np.cumsum(trade_cash+borrow)
    np.testing.assert_allclose(cash,path[:,2],rtol=1e-11,atol=1e-6)
    np.testing.assert_allclose(cash+borrow+open_value,path[:,0],rtol=1e-11,atol=1e-6)
    np.testing.assert_allclose(cash+close_value,path[:,1],rtol=1e-11,atol=1e-6)
    np.testing.assert_allclose(borrow,path[:,12],rtol=1e-11,atol=1e-9)
    maximum=float(max(abs(cash-path[:,2]).max(),abs(cash+close_value-path[:,1]).max()))
    return dict(passed=True,minutes=t,filled_asset_orders=len(fills),maximum_dollar_error=maximum,
                whole_shares=True,quote_causality=quotes is not None)


def execute(m,base,quotes,config,scenario,mode,settings,output):
    output=Path(output);binding=dict(protocol_sha256=sha(output/'protocol.json'),scope=m['scope'],
        config=config,scenario=scenario,mode=mode,settings=asdict(settings),
        base_sha256=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest(),
        market_sha256={k:hashlib.sha256(np.ascontiguousarray(m[k]).tobytes()).hexdigest()
                       for k in ('open','close','volume','actions','starts','sector')},
        etf_bar_manifest_sha256=sha(output/'etf_cache'/(m['scope']+'.bars_manifest.json')) if config['family']=='etf' else None,
        quote_manifest_sha256=sha(output/'etf_cache'/(m['scope']+'.quotes_manifest.json')) if config['family']=='etf'
            else sha(output/'reuse'/(m['scope']+'.json')))
    key=identity(binding)[:24];dest=output/'trials'/(key+'.json');archive=dest.with_suffix('.npz')
    if dest.exists():
        d=read(dest)
        if d['binding']!=binding or sha(archive)!=d['archive_sha256']:raise ValueError('Changed trial or cached result')
        return d
    began=time.perf_counter();q=quotes if mode=='quotes' else None
    shadow,_,_=replay(m,base,np.ones(len(m['meta']['session_ids'])),settings,q)
    shadow_daily=dollar_daily(shadow,m['sessions'],settings.capital);scale=scales(shadow_daily)
    del shadow
    path,orders,failures=replay(m,base,scale,settings,q)
    metrics,daily=normalized_summary(m,path,orders,failures,settings.capital)
    selected=np.isin(m['sessions'],m['evaluate_ids']);active=[]
    for sid in m['evaluate_ids']:
        active.append(bool((path[m['sessions']==sid,7]>0).any()))
    metrics['active_sessions']=sum(active)
    metrics['maximum_net_weight']=float(abs(path[selected,4]).max())
    metrics['overnight_failures']=int((path[selected,16]>1e-8).sum())
    metrics['evaluation_constraint_failures']=int((path[selected,14]>1e-7).sum())
    metrics['posttrade_failure_minutes']=int((path[:,14]>1e-7).sum())
    metrics['coverage_passed']=metrics['quote_coverage']>=.99
    parents=orders[np.isin(m['sessions'][orders[:,0].astype(int)],m['evaluate_ids'])] if len(orders) else orders
    if len(parents):
        _,first=np.unique(parents[:,2],return_index=True);p=parents[first];notional=abs(p[:,4]*p[:,6])
        metrics['first_parent_quote_coverage']=float(notional[p[:,9]==1].sum()/notional.sum()) if notional.sum() else None
    else:metrics['first_parent_quote_coverage']=None
    ix=np.flatnonzero(selected);initial=settings.capital if ix[0]==0 else path[ix[0]-1,1]
    pnl=path[ix[-1],1]-initial;turnover=metrics['turnover_dollars']
    metrics['net_bps_per_traded_dollar']=float(pnl/turnover*10000) if turnover else None
    blocks={};protocol=read(output/'protocol.json')
    for label,partition in protocol['binding']['partitions'][m['scope']].items():
        if label!='warmup':blocks[label],_=normalized_summary(dict(m,evaluate_ids=partition['session_ids']),path,orders,failures,settings.capital)
    proof=verify(m,path,orders,settings,q)
    arrays(archive,equity=path,orders=orders,daily=daily,scales=scale,shadow_daily=shadow_daily)
    d=dict(key=key,binding=binding,config=config,scope=m['scope'],scenario=scenario,mode=mode,
        metrics=metrics,blocks=blocks,verification=proof,archive=str(archive),archive_sha256=sha(archive),
        path_columns=PATH_COLUMNS,order_columns=ORDER_COLUMNS,seconds=time.perf_counter()-began)
    atomic(dest,d)
    print(f'TRIAL {m["scope"]} {config["id"]} {scenario}: SR={metrics["Sharpe"]} return={metrics["total_return"]} valid={metrics["accounting_valid"]} coverage={metrics["quote_coverage"]:.3f}',flush=True)
    return d


def select(rows,output):
    decisions=[]
    for config in CATALOG:
        primary=[r for r in rows if r['config']['id']==config['id'] and r['scenario']=='primary' and r['scope'] in SCOPES[:2]]
        if len(primary)!=2:raise ValueError('Cannot select before both development cohorts finish')
        block_sr=[b['Sharpe'] for r in primary for b in r['blocks'].values()]
        median=float(np.median(block_sr)) if len(block_sr)==5 and all(s is not None for s in block_sr) else None
        eligible=all(r['metrics']['accounting_valid'] and r['metrics']['coverage_passed'] and
                     r['metrics']['total_return']>0 for r in primary)
        eligible &= sum(r['metrics']['active_sessions'] for r in primary)>=20
        eligible &= median is not None and median>=1
        stress=[r for r in rows if r['config']['id']==config['id'] and r['scenario']=='adverse_0p5' and r['scope'] in SCOPES[:2]]
        eligible &= len(stress)==2 and all(r['metrics']['accounting_valid'] and r['metrics']['total_return']>0 for r in stress)
        decisions.append(dict(id=config['id'],family=config['family'],development_qualified=bool(eligible),median_development_block_Sharpe=median))
    d=dict(candidates=decisions,shortlist=[r['id'] for r in decisions if r['development_qualified']],
        selection_only_uses=list(SCOPES[:2]),later_data_previously_exposed=True)
    path=Path(output)/'selection.json'
    if path.exists() and read(path)!=d:raise ValueError('Changed frozen development selection')
    atomic(path,d);return d


def export(rows,output):
    fields=['scope','strategy','family','scenario','mode','Sharpe','total_return','maximum_drawdown',
        'accounting_valid','quote_coverage','first_parent_quote_coverage','coverage_passed','active_sessions',
        'executed_orders','turnover_dollars','explicit_cost_bps','net_bps_per_traded_dollar','maximum_net_weight',
        'overnight_failures','evaluation_constraint_failures']
    with (Path(output)/'trial_ledger.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(scope=r['scope'],strategy=r['config']['id'],family=r['config']['family'],
            scenario=r['scenario'],mode=r['mode'],**{k:r['metrics'].get(k) for k in fields[5:]}))


def run(args):
    output=Path(args.output);protocol=freeze(args.reference,args.prior,output);rows=[];fitted=None
    for scope in SCOPES:
        if scope==SCOPES[2]:select(rows,output)
        m,original=stock_market(args.reference,scope,protocol)
        quotes=stock_quotes(m,args.prior,output)
        if scope==SCOPES[0]:fitted=fit(m,original,args.reference,output)
        em=load_etfs(m,output);eq=etf_quotes(em,output)
        for config in CATALOG:
            if config['family']=='etf':market,quote=em,eq;base,decisions=build_etf(em,config)
            else:
                market,quote=m,quotes;base,decisions=build_stock(m,original,quotes,fitted,config,scenarios()['primary'][1])
            atomic(output/'decisions'/(scope+'_'+config['id']+'.json'),dict(config=config,decisions=decisions,
                information='completed bars, fit-only statistics and quotes at or before signal completion'))
            for name,(mode,settings) in scenarios().items():
                rows.append(execute(market,base,quote,config,name,mode,settings,output));export(rows,output)
        del m,original,quotes,em,eq,base
    selection=read(output/'selection.json');qualified=[]
    for c in selection['candidates']:
        later=next(r for r in rows if r['config']['id']==c['id'] and r['scope']==SCOPES[2] and r['scenario']=='primary')
        stress=next(r for r in rows if r['config']['id']==c['id'] and r['scope']==SCOPES[2] and r['scenario']=='adverse_0p5')
        metric=later['metrics']
        if c['development_qualified'] and metric['accounting_valid'] and metric['coverage_passed'] and metric['total_return']>0 and metric['Sharpe'] is not None and metric['Sharpe']>=1 and stress['metrics']['accounting_valid'] and stress['metrics']['total_return']>0:qualified.append(c['id'])
    atomic(output/'summary.json',dict(trials=len(rows),qualified=qualified,selection=selection,
        all_accounting_verified=all(r['verification']['passed'] for r in rows),
        historical_data_status='Previously research-exposed; none is a fresh confirmatory test',
        protected_future_sessions_accessed=False,rows=rows))
    from .report import render
    render(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','report'])
    p.add_argument('--reference');p.add_argument('--prior');p.add_argument('--output',required=True);a=p.parse_args()
    if a.command=='report':
        from .report import render
        render(Path(a.output))
    else:run(a)
