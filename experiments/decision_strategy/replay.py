"""Minute marks, explicit basket ownership and delayed mandatory exits."""
import numpy as np
from experiments.strategy_validation.config import Execution
from .execution import Book,exposures,integer_targets,round_trip_cost

PATH_COLUMNS=('nav_open','nav_close','cash','gross','net','stock','sector','turnover','fees','borrow','risk_open','risk_close','overnight')


def run(m,candidates,settings,scale,spread_bps,quotes=None,stress='normal'):
    t,n=m['open'].shape;book=Book(n,settings,m['sector']);marks=np.zeros(n)
    tape=np.zeros((t,len(PATH_COLUMNS)));holdings=np.zeros((t,n),np.int32)
    requested=set();decisions=[];baskets=[];completed=[];attribution=[]
    queue={};sid_index=-1;used=set();serial=0;end=-1
    for c in candidates:queue.setdefault(c['origin']+settings.delay,[]).append(c)
    for i in range(t):
        new=i==0 or m['sessions'][i]!=m['sessions'][i-1]
        if new:
            sid_index+=1;used=set();end=i
            while end+1<t and m['sessions'][end+1]==m['sessions'][i]:end+=1
        # Most cost-gated clocks are cash. This shortcut changes no decision,
        # fills, fees or risk calculations and retains every cash minute.
        if not baskets and not book.q.any() and i not in queue:
            good=np.isfinite(m['open'][i])&(m['open'][i]>0);marks[good]=m['open'][i,good]
            good=np.isfinite(m['close'][i])&(m['close'][i]>0);marks[good]=m['close'][i,good]
            tape[i,:3]=book.cash
            continue
        fresh=np.isfinite(m['open'][i])&(m['open'][i]>0);marks[fresh]=m['open'][i,fresh]
        before=exposures(book.q,marks,book.cash,m['sector']);risk_open=before['error']
        old_fills=len(book.fills);oldcash=book.cash
        closing=i>=end-4
        # Exits use only a completed bar whose availability and delay precede
        # this arrival. A maximum-hold exit is always requested on schedule.
        origin=i-settings.delay
        for b in baskets:
            if not b['exit'] and origin>=b['origin'] and m['sessions'][origin]==m['sessions'][b['origin']]:
                js=b['js'];p=m['close'][origin,js];p0=b['signal_prices']
                if np.isfinite(p).all() and (p>0).all() and (p0>0).all():
                    r=p/p0-1;relative=r[0]-r[1:].mean()
                    signed=b['side']*relative
                    close_gap=signed>=.5*abs(b['prediction'])
                    # Prediction-sign reversal from an observed relative gap
                    # crossing its fitted recovery target; no future refit.
                    reverse=False
                    if 'decision_forecast_indices' in m:
                        slot=int(np.searchsorted(m['decision_forecast_indices'],origin))
                        if slot<len(m['decision_forecast_indices']) and m['decision_forecast_indices'][slot]==origin:
                            current=m['decision_forecasts'][slot,b['js'][0]]
                            reverse=np.isfinite(current) and np.sign(current)!=b['side']
                    adverse=False
                    if b['design']=='D03' and 'industry_residual15' in m:
                        residual=m['industry_residual15'][origin,b['js'][0]]
                        entry=b['entry_deviation']
                        if np.isfinite(residual) and abs(entry)>0:
                            close_gap=np.sign(entry)*residual<=.5*abs(entry)
                            adverse=np.sign(entry)*residual>=1.5*abs(entry)
                    if close_gap or reverse or adverse:b['exit']=True;b['exit_reason']='gap_or_stop'
            if i>=b['arrival']+b['horizon'] or closing or risk_open>1e-10:
                b['exit']=True;b['exit_reason']='closing' if closing else 'risk' if risk_open>1e-10 else 'timeout'
            if not b['exit'] and i>b['arrival']+2 and not np.array_equal(b['held'],b['target']):
                b['exit']=True;b['exit_reason']='abandoned_partial_entry'
        for b in list(baskets):
            js=b['js'];delta=-b['held'] if b['exit'] else b['target']-b['held']
            if b['exit'] and not b['held'].any():
                completed.append(dict(id=b['id'],session=int(m['sessions'][i]),arrival=b['arrival'],exit=i,
                    completed_roundtrip=bool(b['ever'].all()),exit_reason='unfilled_entry_expired',stocks=js,ever_filled=b['ever'].tolist()))
                baskets.remove(b);continue
            if not delta.any():continue
            reason='exit' if b['exit'] else 'entry';parent=b['id']+':'+reason
            targets={j:int(book.q[j]+delta[k]) for k,j in enumerate(js)}
            for j,v in targets.items():
                if v!=book.q[j]:requested.add((i,j))
            qdata=quotes.get(i,js) if quotes is not None else None
            held_before=book.q.copy()
            volume=m['volume'][i-1].copy() if not new else np.zeros(n)
            if quotes is None:volume[~fresh]=0
            book.execute(i,int(m['starts'][i]),targets,marks,volume,qdata,parent,reason,stress)
            filled=book.q[js]-held_before[js];b['held']+=filled
            b['ever'] |= filled!=0
            if not b['held'].any() and b['exit']:
                complete=bool(b['ever'].all())
                completed.append(dict(id=b['id'],session=int(m['sessions'][i]),arrival=b['arrival'],exit=i,
                    completed_roundtrip=complete,exit_reason=b['exit_reason'],stocks=js,ever_filled=b['ever'].tolist()))
                baskets.remove(b)
        if not closing and risk_open<=1e-10:
            for c in queue.get(i,[]):
                if len(baskets)>=3:break
                j=c['stock'];js=[j,*c['hedges']]
                if j in used or len(set(js))!=len(js) or m['sessions'][c['origin']]!=m['sessions'][i]:continue
                if scale[sid_index]<=0:continue
                # Eligibility is from the completed signal, never the next
                # available outcome. Missing current quotes are explicit.
                nav=exposures(book.q,marks,book.cash,m['sector'])['nav']
                vol=m['vol'][c['origin'],j]*np.sqrt(390)
                dollars=min(.04,.0025/max(vol,1e-8))*nav*scale[sid_index]
                sides=np.r_[c['side'],np.full(len(js)-1,-c['side'])]
                weights=np.r_[1.,np.full(len(js)-1,1/(len(js)-1))]
                volume_now=m['volume'][i-1,js] if not new else np.zeros(len(js))
                capacity=np.maximum(0,np.floor(np.nan_to_num(volume_now,nan=0)*settings.participation))
                capacity-=np.array([book.used_volume.get((i,x),0) for x in js])
                capacity=np.maximum(capacity,0)
                target=integer_targets(marks[js],sides,weights,dollars,nav,m['sector'][js],book.q[js],capacity)
                decision=dict(**c,arrival=i,session=int(m['sessions'][i]),symbols=[m['symbols'][x] for x in js],quantities=target.tolist())
                if not target.any():decision['disposition']='integer_or_operating_risk_rejected';decisions.append(decision);continue
                if c['gate']=='net_edge' and (not np.isfinite(c['edge_se']) or c['edge_mean']<=c['edge_se']):
                    decision['disposition']='nonpositive_uncertain_gross_edge';decisions.append(decision);continue
                projected=(book.q+np.bincount(js,weights=target,minlength=n))*marks
                if (abs(projected).max()>.04*nav or abs(projected).sum()>.30*nav or abs(projected.sum())>.001*nav
                    or any(abs(projected[m['sector']==g].sum())>.03*nav for g in np.unique(m['sector']))):
                    decision['disposition']='aggregate_operating_rejected';decisions.append(decision);continue
                entry_spread=marks[js]*spread_bps[js]/10000
                if quotes is not None:
                    for x in js:requested.add((i,x))
                    qdata=quotes.get(i,js)
                    good=all(x in qdata and qdata[x] is not None and 0<=int(m['starts'][i])-int(qdata[x][4])<=settings.quote_max_age_seconds*1e9 and 0<qdata[x][0]<=qdata[x][1] and qdata[x][2]>0 and qdata[x][3]>0 for x in js)
                    if not good:
                        # Missing-price decisions still contribute intended
                        # notional to coverage; never drop them from its base.
                        for k,x in enumerate(js):
                            qx=qdata.get(x);age=int(m['starts'][i])-int(qx[4]) if qx is not None else None
                            valid=qx is not None and 0<=age<=settings.quote_max_age_seconds*1e9 and 0<qx[0]<=qx[1] and qx[2]>0 and qx[3]>0
                            book.attempts.append(dict(minute=i,arrival_ns=int(m['starts'][i]),stock=x,parent='quote_check:'+str(c['origin'])+':'+str(j)+':'+str(x),
                                reason='entry_quote_check',intended_shares=int(target[k]),filled_shares=0,benchmark=float(marks[x]),
                                quote_valid=bool(valid),quote_age_ns=age,quote_timestamp_ns=int(qx[4]) if qx is not None else None,
                                query_complete=x in qdata,intended_notional=float(abs(target[k]*marks[x])),fees=0.))
                        decision['disposition']='missing_fresh_entry_quote';decisions.append(decision);continue
                    entry_spread=np.array([qdata[x][1]-qdata[x][0] for x in js])
                    depth=np.array([max(0.,(qdata[x][3] if sides[k]>0 else qdata[x][2])-book.used_depth.get((x,int(qdata[x][4]),int(sides[k])),0)) for k,x in enumerate(js)])
                    capacity=np.minimum(capacity,np.floor(depth))
                    target=integer_targets(marks[js],sides,weights,dollars,nav,m['sector'][js],book.q[js],capacity)
                    decision['quantities']=target.tolist()
                    if not target.any():decision['disposition']='depth_or_integer_rejected';decisions.append(decision);continue
                    projected=(book.q+np.bincount(js,weights=target,minlength=n))*marks
                    if (abs(projected).max()>.04*nav or abs(projected).sum()>.30*nav or abs(projected.sum())>.001*nav
                        or any(abs(projected[m['sector']==g].sum())>.03*nav for g in np.unique(m['sector']))):
                        decision['disposition']='depth_resized_aggregate_risk_rejected';decisions.append(decision);continue
                cost=round_trip_cost(target,marks[js],entry_spread,spread_bps[js],c['horizon'],Execution())
                gross=float(abs(target[0]*marks[j])*c['edge_mean']);uncertainty=float(abs(target[0]*marks[j])*c['edge_se'])
                decision.update(expected_gross_dollars=gross,uncertainty_dollars=uncertainty,estimated_cost_dollars=cost)
                if c['gate']=='net_edge' and (not np.isfinite(uncertainty) or gross-uncertainty<=cost['total']):
                    decision['disposition']='net_edge_abstain';decisions.append(decision);continue
                serial+=1;identifier='basket'+str(serial);used.add(j)
                b=dict(id=identifier,js=js,target=target,held=np.zeros(len(js),int),ever=np.zeros(len(js),bool),
                    arrival=i,origin=c['origin'],horizon=c['horizon'],side=c['side'],prediction=c['prediction'],
                    design=c.get('design',''),entry_deviation=c.get('entry_industry_residual',c['relative']),signal_prices=m['close'][c['origin'],js].copy(),exit=False)
                baskets.append(b);decision.update(disposition='entry_requested',basket=identifier)
                targets={x:int(book.q[x]+target[k]) for k,x in enumerate(js)}
                for x in js:requested.add((i,x))
                held_before=book.q.copy();qdata=quotes.get(i,js) if quotes is not None else None
                volume=m['volume'][i-1].copy() if not new else np.zeros(n)
                if quotes is None:volume[~fresh]=0
                book.execute(i,int(m['starts'][i]),targets,marks,volume,qdata,identifier+':entry','entry',stress)
                b['held']+=book.q[js]-held_before[js];b['ever'] |= b['held']!=0
                decisions.append(decision)
        nav_open=exposures(book.q,marks,book.cash,m['sector'])['nav']
        good=np.isfinite(m['close'][i])&(m['close'][i]>0);marks[good]=m['close'][i,good]
        borrow=float(np.maximum(-book.q*marks,0).sum()*settings.borrow/(365*1440))
        book.cash-=borrow;post=exposures(book.q,marks,book.cash,m['sector'])
        minute_fills=book.fills[old_fills:]
        fees=sum(sum(v[k] for k in ('commission','exchange','clearing','regulatory')) for v in minute_fills)
        turnover=sum(abs(v['shares']*v['price']) for v in minute_fills)
        overnight=float(abs(book.q*marks).sum()) if i==end else 0.
        tape[i]=[nav_open,post['nav'],book.cash,post['gross'],post['net'],post['stock'],post['sector'],turnover,fees,borrow,risk_open,post['error'],overnight]
        holdings[i]=book.q
        if book.q.any() and np.any((book.q!=0)&m['actions'][i]):
            attribution.append(dict(minute=i,type='corporate_action_exposure_failure'))
    return dict(path=tape,holdings=holdings,fills=book.fills,orders=book.attempts,decisions=decisions,
        roundtrips=completed,requests=sorted(requested),unliquidated=[dict(id=b['id'],held=b['held'].tolist()) for b in baskets],
        action_failures=attribution)


def verify(m,result,settings):
    """Rebuild shares, fees, cash and NAV independently, retaining risk failures."""
    t,n=m['open'].shape;delta=np.zeros((t,n),np.int32);paid=np.zeros(t);tradecash=np.zeros(t)
    parents={};max_fill_error=0.
    for f in result['fills']:
        i,j=f['minute'],f['stock'];q=f['shares'];p=f['price'];key=f['parent']
        assert q==int(q)
        qty,notional,previous=parents.get(key,(0,0.,np.zeros(4)))
        qty+=abs(q);notional+=abs(q*p)
        commission=0. if settings.commission_free else min(max(settings.commission_minimum,settings.commission_share*qty),settings.commission_maximum_fraction*notional)
        regulatory=commission*settings.commission_pass_through
        if q<0:regulatory+=settings.sec_sell_fraction*notional+min(settings.finra_sell_cap,qty*settings.finra_sell_share)
        total=np.array([commission,qty*settings.exchange_taker_share,qty*(settings.clearing_share+settings.cat_share),regulatory])
        actual=np.array([f[k] for k in ('commission','exchange','clearing','regulatory')])
        np.testing.assert_allclose(actual,total-previous,rtol=1e-10,atol=1e-9)
        parents[key]=(qty,notional,total)
        delta[i,j]+=q;paid[i]+=actual.sum();tradecash[i]+=q*p
        max_fill_error=max(max_fill_error,f['post_error'])
    held=np.cumsum(delta,axis=0);np.testing.assert_array_equal(held,result['holdings'])
    open_value=np.zeros(t);close_value=np.zeros(t);short_value=np.zeros(t)
    for j in range(n):
        raw=np.empty(2*t);raw[::2]=m['open'][:,j];raw[1::2]=m['close'][:,j]
        good=np.isfinite(raw)&(raw>0);idx=np.maximum.accumulate(np.where(good,np.arange(2*t)+1,0));marks=np.r_[0.,raw][idx]
        open_value+=held[:,j]*marks[::2];close_value+=held[:,j]*marks[1::2];short_value+=np.maximum(-held[:,j]*marks[1::2],0)
    borrow=short_value*settings.borrow/(365*1440);cash=settings.capital-np.cumsum(tradecash+paid+borrow)
    p=result['path'];np.testing.assert_allclose(cash,p[:,2],atol=1e-6,rtol=1e-11)
    np.testing.assert_allclose(cash+borrow+open_value,p[:,0],atol=1e-6,rtol=1e-11)
    np.testing.assert_allclose(cash+close_value,p[:,1],atol=1e-6,rtol=1e-11)
    np.testing.assert_allclose(borrow,p[:,9],atol=1e-8,rtol=1e-10)
    return dict(accounting_reconciled=True,maximum_dollar_error=float(abs(cash+close_value-p[:,1]).max()),
        maximum_fill_risk_excess=max_fill_error,fill_risk_valid=max_fill_error<=1e-10,
        open_close_risk_valid=bool((p[:,10:12]<=1e-10).all()),no_overnight=not bool((p[:,12]>0).any()),
        no_action_failures=not result['action_failures'],unliquidated=result['unliquidated'])
