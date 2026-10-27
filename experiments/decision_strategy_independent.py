"""Read-only, resumable audit of the frozen decision-study tapes.

This module never fits parameters, changes a decision, or reads an unselected
later cohort. Financial reconciliation and risk validity are separate findings.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.strategy_search.storage import atomic, read, sha, utc
from experiments.decision_strategy.config import STUDY, SCOPES
from experiments.decision_strategy_delivery import audit_market, read_rows


def finite_number(value):
    return value is not None and np.isfinite(float(value))


def extend_market(m, output):
    """Guarded adapter has already checked dates before price access."""
    protocol=read(Path(output)/'protocol.json');root=STUDY/'cache'/m['scope']
    meta=m['meta'];archive=np.load(Path(protocol['binding']['market_root'])/'1m/bars.npy',mmap_mode='r')
    volume=np.asarray(archive[meta['begin']:meta['end'],:,4],float)
    etf=np.zeros((len(m['starts']),11));lookup={int(v):i for i,v in enumerate(m['starts'])}
    etfroot=STUDY/'closing_momentum_followup_2026-10-26/etf_cache/bars'
    for date in meta['dates']:
        data=read(etfroot/(date+'.json'))
        for j,symbol in enumerate(m['symbols'][250:]):
            for bar in data['bars'].get(symbol,[]):
                stamp=int(np.datetime64(bar['t'].rstrip('Z'),'ns').astype(np.int64))
                if stamp in lookup:etf[lookup[stamp],j]=bar['v']
    m['volume']=np.c_[volume,etf]
    classes={r['symbol']:r['sector_name'] for r in read(Path(output)/'causal_classifications.json')['rows']}
    from experiments.decision_strategy.data import ETF_MAP
    sectors=sorted(ETF_MAP);m['sector']=np.array([sectors.index(classes[s]) for s in m['symbols'][:250]]+list(range(11)))
    # Reconstruct the last observed open/close marks without using the trader.
    t,n=m['open'].shape;op=np.zeros((t,n));cl=np.zeros((t,n))
    for j in range(n):
        raw=np.empty(2*t);raw[::2]=m['open'][:,j];raw[1::2]=m['close'][:,j]
        good=np.isfinite(raw)&(raw>0);ix=np.maximum.accumulate(np.where(good,np.arange(2*t)+1,0))
        values=np.r_[0.,raw][ix];op[:,j]=values[::2];cl[:,j]=values[1::2]
    m['open_marks']=op;m['close_marks']=cl
    return m


def fill_audit(r,m,shadow=False):
    root=Path(r['root']);prefix='shadow_' if shadow else '';settings=r['settings']
    fills=read_rows(root/(prefix+'fills.csv.gz'));orders=read_rows(root/(prefix+'orders.csv.gz'))
    decisions=read_rows(root/(prefix+'decisions.csv.gz'));parents={};volume=Counter();held=defaultdict(int)
    live=set();nonzero_legs=Counter()
    q=np.zeros(len(m['symbols']),np.int64);cash=settings['capital'];last=-1;max_cash=max_nav=max_net=0.
    max_risk=0.;overfills=0;nonfinite_costs=[];entries=Counter();dispositions=Counter();depth=Counter()
    max_live=0;nonunit=0;missing_quote_files=[];observed_risk=0
    with np.load(root/'minute_tapes.npz') as z:
        path=z['shadow_path' if shadow else 'equity'];holdings=z['shadow_holdings' if shadow else 'holdings']
        scale=z['scale'];daily=z['shadow_daily'];sessions=z['sessions']
    expected_scale=np.zeros(len(daily))
    for k in range(20,len(daily)):
        vol=np.std(daily[k-20:k,1],ddof=1)*np.sqrt(252)
        expected_scale[k]=min(1.,.1/vol) if vol>1e-12 else 1.
    np.testing.assert_allclose(expected_scale,scale,rtol=1e-12,atol=1e-12)
    for d in decisions:
        if not d.get('disposition'):continue
        dispositions[d['disposition']]+=1
        if int(d['arrival'])!=int(d['origin'])+settings['delay']:raise ValueError('Noncausal execution delay')
        if sessions[int(d['arrival'])]!=sessions[int(d['origin'])]:raise ValueError('Decision crossed session')
        if d.get('estimated_cost_dollars') and d.get('gate')=='net_edge':
            estimate=json.loads(d['estimated_cost_dollars'])
            if not all(finite_number(v) for v in estimate.values()):nonfinite_costs.append(dict(origin=d['origin'],symbols=d['symbols'],disposition=d['disposition']))
        if d['disposition']=='entry_requested':
            if not shadow and scale[int(np.searchsorted(daily[:,0],sessions[int(d['arrival'])]))]<=0:
                raise ValueError('Scaled book entered during causal cash warmup')
            symbol=json.loads(d['symbols'])[0];key=(d['session'],symbol);entries[key]+=1
            if entries[key]>1:raise ValueError('Repeated primary ticker/day')
    for f in fills:
        i,j=int(f['minute']),int(f['stock']);side=int(f['shares']);px=float(f['price']);key=f['parent']
        nonunit+=int(abs(side)!=1)
        if i!=last:
            if last>=0:np.testing.assert_array_equal(q,holdings[last])
            cash=settings['capital'] if i==0 else path[i-1,2]
            if i>0:np.testing.assert_array_equal(q,holdings[i-1])
            last=i
        parent_qty,parent_dollars,paid=parents.get(key,(0,0.,np.zeros(4)))
        parent_qty+=abs(side);parent_dollars+=abs(side*px)
        commission=0. if settings['commission_free'] else min(max(settings['commission_minimum'],settings['commission_share']*parent_qty),settings['commission_maximum_fraction']*parent_dollars)
        regulatory=commission*settings['commission_pass_through']
        if side<0:regulatory+=settings['sec_sell_fraction']*parent_dollars+min(settings['finra_sell_cap'],settings['finra_sell_share']*parent_qty)
        total=np.array([commission,parent_qty*settings['exchange_taker_share'],parent_qty*(settings['clearing_share']+settings['cat_share']),regulatory])
        charges=np.array([float(f[k]) for k in ('commission','exchange','clearing','regulatory')])
        if np.any(abs(charges-(total-paid))>1e-9+1e-10*abs(total-paid)):raise ValueError('Independent parent fee mismatch')
        parents[key]=(parent_qty,parent_dollars,total)
        cash-=side*px+charges.sum();q[j]+=side;volume[(i,j)]+=abs(side)
        basket=key.split(':')[0];old=held[(basket,j)];held[(basket,j)]+=side
        nonzero_legs[basket]+=int(held[(basket,j)]!=0)-int(old!=0)
        if nonzero_legs[basket]:live.add(basket)
        else:live.discard(basket)
        max_live=max(max_live,len(live))
        if max_live>3:raise ValueError('More than three live baskets')
        marks=m['open_marks'][i];dollars=q*marks;nav=cash+dollars.sum()
        net=dollars.sum()/nav;gross=abs(dollars).sum()/nav;stock=abs(dollars).max()/nav
        sector=abs(np.bincount(m['sector'],weights=dollars,minlength=11)).max()/nav
        risk=max(0.,gross-1.,abs(net)-.0025,stock-.05,sector-.05)
        max_risk=max(max_risk,risk);max_net=max(max_net,abs(net));observed_risk+=int(risk>1e-10)
        max_cash=max(max_cash,abs(cash-float(f['cash'])));max_nav=max(max_nav,abs(nav-float(f['post_nav'])))
        expected=np.array([float(f['post_nav']),float(f['post_net']),float(f['post_error'])])
        if np.any(abs(np.array([nav,net,risk])-expected)>1e-8+1e-9*abs(expected)):raise ValueError('Independent child NAV/risk mismatch')
    if last>=0:np.testing.assert_array_equal(q,holdings[last])
    for (i,j),shares in volume.items():
        previous=0. if i==0 or sessions[i-1]!=sessions[i] else m['volume'][i-1,j]
        cap=int(np.floor(previous*settings['participation'])) if np.isfinite(previous) else 0
        if shares>cap:overfills+=1
    if max_cash>1e-6 or max_nav>1e-6 or nonunit or overfills:raise ValueError('Independent fill/volume/cash verification failed')
    order_totals=Counter()
    for o in orders:
        if not o.get('stock'):continue
        i,j=int(o['minute']),int(o['stock']);filled=int(o['filled_shares']);order_totals[(i,j)]+=abs(filled)
        if abs(filled)>abs(int(o['intended_shares'])):raise ValueError('Fill exceeds intended quantity')
        if r['mode']=='quotes' and filled:
            stamp=int(m['starts'][i]);p=Path(root).parents[1]/'quotes'/r['scope']/(str(stamp)+'_'+m['symbols'][j]+'.json')
            if not p.exists():missing_quote_files.append(str(p));continue
            quote=read(p)['quote'];bid,ask,bs,az,ts=quote
            if not 0<=stamp-int(ts)<=settings['quote_max_age_seconds']*1e9:raise ValueError('Stale/future filled quote')
            dk=(j,int(ts),int(np.sign(filled)));depth[dk]+=abs(filled)
            if depth[dk]>(az if filled>0 else bs)+1e-9:raise ValueError('Displayed depth reused')
    if dict(order_totals)!=dict(volume):
        if {k:v for k,v in order_totals.items() if v}!=dict(volume):raise ValueError('Fill/attempt quantity disagreement')
    if missing_quote_files:raise ValueError('Unverified filled SIP snapshot')
    return dict(shadow=shadow,independent_financial_and_participation_checks_passed=True,
        filled_children=len(fills),attempts=len(orders),decisions=sum(dispositions.values()),maximum_cash_error_dollars=max_cash,
        maximum_fill_nav_error_dollars=max_nav,maximum_fill_abs_net=max_net,maximum_fill_risk_excess=max_risk,
        fill_risk_failure_children=observed_risk,fill_risk_valid=max_risk<=1e-10,
        minute_risk_failure_count=int((path[:,10:12]>1e-10).any(1).sum()),overnight_failure_count=int((path[:,12]>0).sum()),
        maximum_live_filled_baskets=max_live,participation_overfills=overfills,nonfinite_economic_cost_decisions=nonfinite_costs,
        dispositions=dict(dispositions),causal_past20_volatility_scale_verified=True,
        quote_depth_check='verified on filled snapshots' if r['mode']=='quotes' else 'not applicable: minute-open proxy, not SIP replay')


def main(output,partial=False):
    output=Path(output);receipts=sorted((output/'trials').glob('*/receipt.json'));proofs=[]
    for scope in SCOPES:
        selected=[p for p in receipts if read(p)['scope']==scope]
        todo=[p for p in selected if not (p.parent/'independent_execution.json').exists()]
        if todo:
            if scope==SCOPES[2] and not (output/'selection.json').exists():raise ValueError('Later cohort is locked')
            m=extend_market(audit_market(scope,output),output)
            for p in todo:
                r=read(p);proof=dict(key=r['key'],receipt_sha256=sha(p),auditor_sha256=sha(__file__),
                    scaled=fill_audit(r,m),shadow=fill_audit(r,m,True),verified_UTC=utc())
                atomic(p.parent/'independent_execution.json',proof)
                print('AUDITED',r['key'],flush=True)
            del m
        for p in selected:
            proof=read(p.parent/'independent_execution.json')
            if proof['receipt_sha256']!=sha(p) or proof['auditor_sha256']!=sha(__file__):raise ValueError('Changed audited input or auditor')
            proofs.append(dict(key=proof['key'],sha256=sha(p.parent/'independent_execution.json'),
                nonfinite_costs=sum(len(proof[k]['nonfinite_economic_cost_decisions']) for k in ('scaled','shadow'))))
    result=dict(verified_UTC=utc(),partial=partial,trial_count=len(proofs),proofs=proofs,
        independently_reconciled=True,total_nonfinite_economic_cost_decisions=sum(p['nonfinite_costs'] for p in proofs),
        no_risk_failures_erased=True,protected_future_outcomes_opened=False,
        risk_granularity='Every sequential child fill and available minute open/close; unseen intraminute path unobservable.')
    atomic(output/('independent_execution_partial.json' if partial else 'independent_execution_verification.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(STUDY/'decision_strategy_execution_2026-10-27_v6'));p.add_argument('--partial',action='store_true');a=p.parse_args()
    print(main(a.output,a.partial))
