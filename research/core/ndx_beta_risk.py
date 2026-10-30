"""Trailing basket beta controls, using preceding-close returns only."""
import numpy as np
from ndx_drawdown_engine import risk_inputs,daily_allocation,exposure_series
from ndx_retention import hedged_targets


def trailing_beta(data,ids,weights,qqq,window):
    if window not in (60,120):raise ValueError('Fixed beta windows are60or120')
    q=np.asarray(qqq,float);qr=np.r_[np.nan,q[1:]/q[:-1]-1]
    beta=[];counts=[]
    for row,i in enumerate(ids):
        day=int(data['origin'][i]);start=max(0,day-window+1)
        active=weights[row]>0
        r=data['returns'][start:day+1,active];market=qr[start:day+1]
        valid=np.isfinite(r).all(1)&np.isfinite(market)
        count=int(valid.sum());counts.append(count)
        value=1.
        if count>=max(20,window//2):
            basket=r[valid]@weights[row,active];market=market[valid]
            x=basket-basket.mean();y=market-market.mean();denominator=float(y@y)
            if denominator>1e-8:value=float(x@y/denominator)
        beta.append(np.clip(value,0.,3.))
    return np.asarray(beta),np.asarray(counts)


def risk_targets(data,ids,weights,market,policy):
    risk=risk_inputs(data,daily_allocation(data,ids,weights));origin=data['origin'][ids]
    target=.15 if policy=='cash15' else .20
    long=exposure_series(dict(vol='basket',window=20,target=target),market,risk)[origin]
    beta=np.full(len(ids),np.nan);counts=np.zeros(len(ids),int)
    if policy in ('original20','cash15'):
        result=hedged_targets(weights,long,market['qqqvol20'][origin],.75)
    elif policy in ('beta60','beta120'):
        beta,counts=trailing_beta(data,ids,weights,market['qqq'],int(policy[4:]))
        long=np.minimum(long,1/(1+beta))
        short=np.minimum(np.minimum(.75,beta*long),.20/np.maximum(market['qqqvol20'][origin],.05))
        result=np.column_stack((weights*long[:,None],-short))
    else:raise ValueError('Unknown fixed risk policy')
    assert np.max(np.abs(result).sum(1))<=1+1e-12
    detail=dict(beta=beta,complete_return_days=counts,long_target=long,short_target=-result[:,-1],
                basket_volatility=risk['basketvol20'][origin])
    return result,detail
