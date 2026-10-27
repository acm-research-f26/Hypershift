"""Closing-price benchmarks, deliberately ineligible for execution qualification."""
from pathlib import Path
import numpy as np
from experiments.strategy_search.storage import read,sha,atomic,arrays
from experiments.strategy_search.statistics import scales,risk,bootstrap
from experiments.strategy_search.execution import project
from .config import SCOPES,FEES
from .data import stock_market
from .engine import feasible_round,limits

POLICIES=('early_momentum_auction','last60_momentum_auction')


def closing_print(records,date):
    candidates=[c for d in records if d.get('d')==date for c in (d.get('c') or [])
        if c.get('c')=='6' and c.get('p',0)>0 and c.get('s',0)>0]
    if not candidates:return None
    largest=max(c['s'] for c in candidates);best=[c for c in candidates if c['s']==largest]
    if len({c['p'] for c in best})!=1:return None
    return sorted(best,key=lambda c:c['t'])[-1]


def decisions(m,policy,output):
    results=[];files=read(Path(output)/'auctions'/'manifest.json')['files']
    for sid in m['meta']['session_ids']:
        ix=np.flatnonzero(m['sessions']==sid);origin=int(ix[-31]);entry=origin+2;end=int(ix[-1])
        date=str(np.datetime64(int(m['starts'][origin]),'ns').astype('datetime64[D]'))
        dest=Path(output)/'auctions'/(date+'.json')
        if sha(dest)!=files[dest.name]:raise ValueError('Changed auction outcomes')
        doc=read(dest);start=int(ix[0] if policy=='early_momentum_auction' else origin-60)
        stop=int(ix[59] if policy=='early_momentum_auction' else origin)
        a=m['open'][start] if policy=='early_momentum_auction' else m['close'][start]
        b=m['close'][stop];good=m['eligible_full'][origin]&np.isfinite(a)&(a>0)&np.isfinite(b)&(b>0)
        raw=np.divide(b,a,out=np.ones_like(a),where=good)-1
        score=raw-(np.mean(raw[good]) if good.any() else 0.)
        long=np.flatnonzero(good&(score>0));short=np.flatnonzero(good&(score<0))
        long=long[np.argsort(-score[long],kind='stable')[:3]];short=short[np.argsort(score[short],kind='stable')[:3]]
        w=np.zeros(len(a));js=np.r_[long,short]
        if len(long) and len(short):
            v=m['vol'][origin,js];valid=np.isfinite(v)&(v>1e-8)
            w[js[valid]]=np.sign(score[js[valid]])*np.minimum(.04,.005/(v[valid]*np.sqrt(390)))
            w=project(w,m['sector'],True,np.zeros(len(w),bool),np.zeros(len(w)),True)
        # Outcome availability must never change which stocks are selected.
        prints={m['symbols'][j]:closing_print(doc['auctions'].get(m['symbols'][j],[]),date) for j in np.flatnonzero(w)}
        results.append(dict(session=int(sid),date=date,origin=origin,entry=entry,end=end,weights=w.tolist(),
            selected=[m['symbols'][j] for j in np.flatnonzero(w)],prints=prints))
    return results


def replay(m,decision,bps,scale):
    capital=cash=100000.;daily=[];orders=[];curves=[];missing=[];violations=0;active=0
    for k,d in enumerate(decision):
        initial=cash;start=d['entry'];end=d['end'];weights=np.array(d['weights'])*scale[k]
        marks=m['open'][start];clean=np.where(np.isfinite(marks)&(marks>0),marks,0.)
        q=feasible_round(np.trunc(np.divide(weights*cash,clean,out=np.zeros_like(weights),where=clean>0)),clean,cash,m['sector'],.001)
        selected=np.flatnonzero(q);absent=[m['symbols'][j] for j in selected if d['prints'].get(m['symbols'][j]) is None]
        if absent or np.any((weights!=0)&(clean==0)):
            missing.append(dict(session=d['session'],symbols=absent,reason='Required terminal price or entry price missing'))
            daily.append([d['session'],0.,cash,0.,0.,0.]);curves.append(cash);continue
        entry=clean*(1+np.sign(q)*bps/10000);cash-=np.dot(q,entry)
        traded=float(np.dot(abs(q),entry));borrow=0.;active+=bool(len(selected))
        for i in range(start,end+1):
            fresh=np.isfinite(m['close'][i])&(m['close'][i]>0);clean[fresh]=m['close'][i,fresh]
            # Closing print is first used at the terminal outcome, after signal formation.
            if i==end:
                for j in selected:clean[j]=d['prints'][m['symbols'][j]]['p']
            charge=float(np.maximum(-q*clean,0).sum()*.03/(365*1440));borrow+=charge;cash-=charge
            nav=cash+np.dot(q,clean);curves.append(nav)
            violations+=int(limits(q,clean,nav,m['sector'],.0025)>1e-7 or np.any(m['actions'][i]& (q!=0)))
        exit_price=clean*(1-np.sign(q)*bps/10000);cash+=np.dot(q,exit_price);curves.append(cash)
        traded+=float(np.dot(abs(q),exit_price))
        for j in selected:orders.append([d['session'],j,start,end,q[j],entry[j],exit_price[j],d['prints'][m['symbols'][j]]['s']])
        daily.append([d['session'],cash/initial-1,cash,traded,0.,borrow])
    return np.array(daily),np.array(curves),np.array(orders).reshape(-1,8),dict(missing=missing,constraint_minutes=violations,active_sessions=active)


def run(a,protocol):
    output=Path(a.output);rows=[]
    # Auction analysis is released only after the main development selection exists.
    read(output/'selection.json')
    for scope in SCOPES:
        m,old=stock_market(a.reference,scope,protocol);del old
        for policy in POLICIES:
            chosen=decisions(m,policy,output)
            atomic(output/'auction_benchmarks'/(scope+'_'+policy+'_decisions.json'),chosen)
            for bps in FEES:
                shadow,_,_,shadow_audit=replay(m,chosen,bps,np.ones(len(chosen)))
                scale=scales(shadow);daily,curve,orders,audit=replay(m,chosen,bps,scale)
                keep=np.isin(daily[:,0],m['evaluate_ids']);metric=risk(daily[keep,1])
                metric.update(uncertainty=bootstrap(daily[keep,1]),execution_validated=False,
                    accounting_valid=not(audit['missing'] or shadow_audit['missing'] or audit['constraint_minutes'] or shadow_audit['constraint_minutes']),
                    active_sessions=int((daily[keep,3]>0).sum()),turnover_dollars=float(daily[keep,3].sum()),
                    borrow_dollars=float(daily[keep,5].sum()))
                key=scope+'_'+policy+f'_cost{bps:g}';dest=output/'auction_benchmarks'/(key+'.npz')
                arrays(dest,daily=daily,curve=curve,orders=orders,scales=scale,shadow_daily=shadow)
                row=dict(scope=scope,strategy=policy,cost_bps=bps,metrics=metric,audit=audit,shadow_audit=shadow_audit,
                    archive=str(dest),archive_sha256=sha(dest),qualified=False,
                    assumption='Minute-open entry and full closing-print fills; no order routing, depth, queue, impact or imbalance history')
                rows.append(row)
                print(f'AUCTION_BENCHMARK {key} SR={metric["Sharpe"]} return={metric["total_return"]} valid={metric["accounting_valid"]}',flush=True)
        del m
    atomic(output/'auction_benchmarks.json',dict(rows=rows,trials=len(rows),all_execution_validated=False,
        conditions_source='https://docs.alpaca.markets/us/docs/market-data-faq',
        price_rule='Largest reported market-center closing trade, condition6; not proof of achievable primary-listing execution'))
