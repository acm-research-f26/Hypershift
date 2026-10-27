"""Same-horizon expanding OOF calibration, with session-purged boundaries."""
import numpy as np


def ridge(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    good=np.isfinite(x).all(1)&np.isfinite(y)
    x,y=x[good],y[good]
    if len(y)<100:raise ValueError('Insufficient training observations')
    mean=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-10,sd,1.)
    z=np.c_[np.ones(len(x)),(x-mean)/sd]
    penalty=np.eye(z.shape[1])*1e-3;penalty[0,0]=0
    beta=np.linalg.solve(z.T@z+penalty,z.T@y)
    return dict(mean=mean,sd=sd,beta=beta,observations=len(y))


def predict(model,x):
    return np.c_[np.ones(len(x)),(x-model['mean'])/model['sd']]@model['beta']


def calibrate(scores,outcomes,sessions):
    scores,outcomes,sessions=map(np.asarray,(scores,outcomes,sessions))
    good=np.isfinite(scores)&np.isfinite(outcomes)&(scores!=0)
    scores,outcomes,sessions=scores[good],outcomes[good],sessions[good]
    if len(scores)<1000 or len(np.unique(sessions))<20:
        return dict(valid=False,observations=len(scores),sessions=len(np.unique(sessions)),bins=[])
    cuts=np.unique(np.quantile(abs(scores),np.arange(1,10)/10))
    b=np.searchsorted(cuts,abs(scores));rows=[]
    signed=np.sign(scores)*outcomes
    for k in range(len(cuts)+1):
        mask=b==k
        means=np.array([signed[mask&(sessions==s)].mean() for s in np.unique(sessions[mask])])
        good_bin=len(means)>=5
        rows.append(dict(mean=float(signed[mask].mean()) if mask.any() else 0.,
            se=float(means.std(ddof=1)/np.sqrt(len(means))) if good_bin else np.inf,
            sessions=len(means),observations=int(mask.sum()),valid=good_bin))
    return dict(valid=True,observations=len(scores),sessions=len(np.unique(sessions)),cuts=cuts,bins=rows)


def edge(calibration,score):
    if not calibration['valid']:return 0.,np.inf
    b=int(np.searchsorted(calibration['cuts'],abs(score)))
    r=calibration['bins'][b]
    return r['mean'],r['se']


def fold_sessions(ids):
    if len(ids)<60:raise ValueError('First60 training sessions required')
    return [(list(ids[:n]),list(ids[n:n+10])) for n in (20,30,40,50)]


def labels(m,origins,stocks,hedges,horizon):
    """Arithmetic relative returns: equal-dollar P&L per leading dollar.

    Features may be log returns; a dollar-P&L target must be arithmetic. Labels
    never enter tradability or decisions.
    """
    origins,stocks,hedges=map(np.asarray,(origins,stocks,hedges))
    a=origins+2;b=a+horizon;safe=np.minimum(b,len(m['sessions'])-1)
    js=np.c_[stocks,hedges]
    before=m['open'][np.minimum(a,len(m['sessions'])-1)[:,None],js]
    after=m['open'][safe[:,None],js]
    good=(b<len(m['sessions']))&(m['sessions'][safe]==m['sessions'][origins])
    good &= np.isfinite(before).all(1)&np.isfinite(after).all(1)&(before>0).all(1)&(after>0).all(1)
    y=np.full(len(origins),np.nan)
    r=np.divide(after,before,out=np.ones_like(after,dtype=float),where=(before>0)&(after>0))-1
    y[good]=r[good,0]-r[good,1:].mean(1)
    return y
