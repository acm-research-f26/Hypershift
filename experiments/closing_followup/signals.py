"""Price/volume policies; fitting is restricted to the first sixty earlier sessions."""
import numpy as np
from experiments.strategy_validation.fees import charges
from experiments.strategy_search.storage import arrays, atomic, read, sha
from .config import CATALOG


def session_features(m, sd):
    result=[]
    for sid in m['meta']['session_ids']:
        ix=np.flatnonzero(m['sessions']==sid); decision=int(ix[len(ix)-30])
        ok=np.isfinite(m['open'][ix[0]]) & (m['open'][ix[0]]>0)
        ok &= np.isfinite(m['close'][ix[59]]) & (m['close'][ix[59]]>0)
        ok &= np.isfinite(m['close'][decision]) & (m['close'][decision]>0)
        early=np.log(np.divide(m['close'][ix[59]],m['open'][ix[0]],out=np.ones(len(sd)),where=ok))
        current=np.log(np.divide(m['close'][decision],m['open'][ix[0]],out=np.ones(len(sd)),where=ok))
        relative=early.copy(); late=current.copy(); breadth=np.zeros(len(sd))
        for g in np.unique(m['sector']):
            js=np.flatnonzero((m['sector']==g)&ok)
            if len(js)<3: continue
            relative[js]-=early[js].mean();late[js]-=current[js].mean()
            consistent=(early[js]*current[js]>0).astype(float)
            breadth[js]=(consistent.sum()-consistent)/max(1,len(js)-1)
        z=np.divide(relative,sd,out=np.zeros(len(sd)),where=np.isfinite(sd)&(sd>0))
        early_volume=np.nansum(m['volume'][ix[:60]],axis=0)
        result.append(dict(sid=int(sid),ix=ix,decision=decision,early=early,current=current,
                           relative=relative,late=late,z=z,ok=ok,breadth=breadth,early_volume=early_volume))
    return result


def fit(m, base, reference, output):
    from pathlib import Path
    ref=Path(reference);dest=Path(output)/'fit.npz'
    with np.load(ref/'fit.npz') as z: sd=z['S10_60_0_None_0_0_sd'] if 'S10_60_0_None_0_0_sd' in z else None
    if sd is None: raise ValueError('Original S10 training normalization missing')
    features=session_features(m,sd)
    train=features[:60]
    volumes=np.array([x['early_volume'] for x in train])
    dollar=np.array([np.nanmedian(m['volume'][x['ix'][-35:]]*m['close'][x['ix'][-35:]],axis=0) for x in train])
    liquidity=np.nanmedian(dollar,axis=0)
    liquid=liquidity>=np.nanmedian(liquidity)
    xs=[];ys=[]
    for f in train:
        ix=f['ix'];i=f['decision'];entry=i+2;exit=int(ix[-5])
        good=f['ok'] & (base[i]!=0) & np.isfinite(m['open'][entry]) & (m['open'][entry]>0)
        good &= np.isfinite(m['open'][exit]) & (m['open'][exit]>0)
        future=np.log(np.divide(m['open'][exit],m['open'][entry],out=np.ones(len(sd)),where=good))
        for g in np.unique(m['sector']):
            js=np.flatnonzero((m['sector']==g)&np.isfinite(m['open'][entry])&(m['open'][entry]>0)&np.isfinite(m['open'][exit])&(m['open'][exit]>0))
            if len(js)>1:
                sector_future=np.log(m['open'][exit,js]/m['open'][entry,js]).mean()
                future[js]-=sector_future
        js=np.flatnonzero(good)
        xs.extend(abs(f['z'][js]));ys.extend(np.sign(base[i,js])*future[js])
    design=np.column_stack([np.ones(len(xs)),xs])
    beta=np.linalg.lstsq(design,np.asarray(ys),rcond=None)[0] if len(xs)>=100 else np.zeros(2)
    fitted=dict(sd=sd,early_volume=np.nanmedian(volumes,axis=0),liquid=liquid,liquidity=liquidity,beta=beta)
    if dest.exists():
        with np.load(dest) as z:
            for k,v in fitted.items():np.testing.assert_array_equal(z[k],v)
    else:arrays(dest,**fitted)
    atomic(Path(output)/'fit.json',dict(training_session_ids=m['meta']['session_ids'][:60],
        fitted_sha256=sha(dest),regression_observations=len(xs),edge_intercept=float(beta[0]),edge_slope=float(beta[1]),
        liquid_stocks=int(liquid.sum()),performance_excluded=True))
    return fitted


def fresh_quote(q, now, seconds=5):
    bid,ask,bs,az,stamp=q
    return bool(np.isfinite([bid,ask,bs,az]).all() and 0<bid<=ask and bs>0 and az>0 and 0<=int(now)-int(stamp)<=seconds*1e9)


def build_stock(m, original, quotes, fitted, config, settings):
    if config['family']=='control':return original.copy(),[]
    result=np.zeros_like(original);decisions=[]
    for f in session_features(m,fitted['sd']):
        i=f['decision'];ix=f['ix'];weights=original[i].copy()
        eligible=f['ok'] & m['eligible_full'][i] & fitted['liquid'] & (weights!=0)
        rvol=np.divide(f['early_volume'],fitted['early_volume'],out=np.zeros_like(weights),where=fitted['early_volume']>0)
        predicted=np.maximum(0.,fitted['beta'][0]+fitted['beta'][1]*abs(f['z']))
        hurdles=np.full(len(weights),np.nan)
        if config['family']=='participation':
            eligible &= (rvol>=config['rvol']) & (abs(f['z'])>=1.) & (f['breadth']>=config['breadth'])
        else:
            if config['cost_gate']:
                for j in np.flatnonzero(eligible):
                    # This snapshot is at the completed signal close, one minute BEFORE arrival.
                    q=quotes[i+1,j]
                    if not fresh_quote(q,m['completed'][i]):eligible[j]=False;continue
                    bid,ask,bs,az,stamp=q;mid=(bid+ask)/2
                    shares=np.floor(abs(weights[j])*settings.capital/mid)
                    if shares<=0:eligible[j]=False;continue
                    fee=sum(charges(shares,ask,settings))+sum(charges(-shares,bid,settings))
                    hurdles[j]=(ask-bid)/mid+fee/(shares*mid)+.0001
                    eligible[j] &= predicted[j]>hurdles[j]
            positive=np.flatnonzero(eligible&(weights>0));negative=np.flatnonzero(eligible&(weights<0))
            keep=np.r_[positive[np.argsort(-abs(f['z'][positive]),kind='stable')[:config['names_per_side']]],
                       negative[np.argsort(-abs(f['z'][negative]),kind='stable')[:config['names_per_side']]]]
            eligible[:]=False;eligible[keep]=True
        weights[~eligible]=0
        if not np.any(weights>0) or not np.any(weights<0):weights[:]=0
        result[i:ix[-1]+1]=weights
        for j in np.flatnonzero(weights):
            decisions.append(dict(session=f['sid'],signal_completed_ns=int(m['completed'][i]),ticker=m['symbols'][j],
                direction=int(np.sign(weights[j])),raw_weight=float(weights[j]),early_score=float(f['z'][j]),
                early_rvol=float(rvol[j]),peer_persistence=float(f['breadth'][j]),predicted_edge_bps=float(predicted[j]*10000),
                hurdle_bps=float(hurdles[j]*10000) if np.isfinite(hurdles[j]) else None))
    return result,decisions


def build_etf(m,config):
    result=np.zeros_like(m['open']);decisions=[]
    for sid in m['meta']['session_ids']:
        ix=np.flatnonzero(m['sessions']==sid);i=int(ix[-30]);n=len(m['symbols'])
        ok=(m['open'][ix[0]]>0)&(m['close'][ix[59]]>0)&(m['close'][i]>0)&m['eligible_full'][i]
        early=np.log(np.divide(m['close'][ix[59]],m['open'][ix[0]],out=np.ones(n),where=ok))
        current=np.log(np.divide(m['close'][i],m['open'][ix[0]],out=np.ones(n),where=ok))
        weights=np.zeros(n)
        if config['kind']=='market_pair':
            if ok[:2].all() and (early[1]-early[0])*(current[1]-current[0])>0:
                weights[:2]=np.array([-1,1])*.04*np.sign(early[1]-early[0])
        else:
            js=np.arange(2,n);valid=js[ok[js]]
            if len(valid)>=6:
                er=early[valid]-early[valid].mean();cr=current[valid]-current[valid].mean()
                up=valid[(er>0)&(cr>0)];down=valid[(er<0)&(cr<0)]
                up=up[np.argsort(-early[up],kind='stable')[:2]];down=down[np.argsort(early[down],kind='stable')[:2]]
                if len(up)==len(down)==2:weights[up]=.04;weights[down]=-.04
        result[i:ix[-1]+1]=weights
        for j in np.flatnonzero(weights):decisions.append(dict(session=int(sid),signal_completed_ns=int(m['completed'][i]),
            ticker=m['symbols'][j],direction=int(np.sign(weights[j])),raw_weight=float(weights[j]),early_return=float(early[j])))
    return result,decisions
