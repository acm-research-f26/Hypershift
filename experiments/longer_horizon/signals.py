"""Sparse causal shock events; every learned quantity is restricted to train60."""
from pathlib import Path
import numpy as np
from experiments.strategy_search.features import window_sum
from experiments.strategy_search.execution import project
from experiments.strategy_search.storage import atomic,arrays,sha,read


def residual_window(m):
    valid=m['valid'];count=valid.sum(1);mean=np.divide(np.where(valid,m['raw'],0).sum(1),count,out=np.zeros(len(count)),where=count>0)
    residual=np.where(valid,m['raw']-mean[:,None],0.)
    return window_sum(residual,valid,m['sessions'],5)


def fit(m,graph,output):
    train_ids=m['meta']['session_ids'][:60];audit=read(Path(graph)/'network_fit.json')
    if audit['details']['training_sessions']!=train_ids:raise ValueError('Graph training membership mismatch')
    x,ok=residual_window(m);train=np.isin(m['sessions'],train_ids);good=ok&train[:,None]
    count=good.sum(0);mean=np.divide(np.where(good,x,0).sum(0),count,out=np.zeros(x.shape[1]),where=count>0)
    sd=np.sqrt(np.divide(np.where(good,(x-mean)**2,0).sum(0),np.maximum(count-1,1)))
    sd=np.where((count>=100)&(sd>1e-8),sd,np.nan)
    with np.load(Path(graph)/'networks.npz') as z:
        fitted=dict(sd=sd,direct=z['5_direct'],two_hop=z['5_chain'],shuffled_two_hop=z['5_shuffled_chain'])
    dest=Path(output)/'fit.npz'
    if dest.exists():
        with np.load(dest) as z:
            for k,v in fitted.items():np.testing.assert_array_equal(z[k],v)
    else:arrays(dest,**fitted)
    atomic(Path(output)/'fit.json',dict(training_ids=train_ids,fit_sha256=sha(dest),stocks=len(sd),performance_excluded=True,
        graph_training_audit_sha256=sha(Path(graph)/'network_fit.json')))
    return fitted


def sampled_features(m,fitted):
    x,ok=residual_window(m)
    indices=[]
    for sid in m['meta']['session_ids']:
        ix=np.flatnonzero(m['sessions']==sid)
        indices.extend(ix[np.arange(60,min(211,len(ix)-9),15)])
    indices=np.array(indices,int);valid=ok[indices]&np.isfinite(fitted['sd'])[None,:]
    own=np.divide(x[indices],fitted['sd'],out=np.zeros_like(x[indices]),where=valid)
    features={'indices':indices,'own':own,'valid':valid}
    for family in ['direct','two_hop','shuffled_two_hop']:
        w=fitted[family];denominator=valid.astype(float)@abs(w).T
        value=np.where(valid,own,0)@w.T
        features[family]=np.divide(value,denominator,out=np.zeros_like(value),where=denominator>=.8)
        features[family+'_valid']=denominator>=.8
    for level in ['industry','sector']:
        values=np.zeros_like(own);available=np.zeros_like(valid)
        for g in np.unique(m[level]):
            js=np.flatnonzero(m[level]==g);v=own[:,js];good=valid[:,js]
            count=good.sum(1)[:,None]-good;total=np.where(good,v,0).sum(1)[:,None]-np.where(good,v,0)
            values[:,js]=np.divide(total,count,out=np.zeros_like(v),where=count>=2);available[:,js]=count>=2
        features[level]=values;features[level+'_valid']=available
    return features


def build(m,features,config):
    base=np.zeros_like(m['open']);decisions=[];used=set();family=config['family']
    for row,i in enumerate(features['indices']):
        sid=int(m['sessions'][i])
        if sid in used:continue
        ix=np.flatnonzero(m['sessions']==sid);exit_origin=min(int(i+config['hold']),int(ix[-7]))
        own=features['own'][row];valid=features['valid'][row]&m['eligible_full'][i]
        if family in ('self_momentum','self_reversal'):
            score=own.copy()*(1 if family=='self_momentum' else -1);eligible=valid&(abs(own)>=2)
        else:
            leader=features[family][row];score=leader-own
            eligible=valid&features[family+'_valid'][row]&(abs(leader)>=2)&(np.sign(leader)*own<.25*abs(leader))
        long=np.flatnonzero(eligible&(score>0));short=np.flatnonzero(eligible&(score<0))
        long=long[np.argsort(-score[long],kind='stable')[:3]];short=short[np.argsort(score[short],kind='stable')[:3]]
        if not len(long) or not len(short):continue
        target=np.zeros(base.shape[1]);js=np.r_[long,short]
        vol=m['vol'][i,js];good=np.isfinite(vol)&(vol>1e-8)
        target[js[good]]=np.sign(score[js[good]])*np.minimum(.04,.005/(vol[good]*np.sqrt(390.)))
        target=project(target,m['sector'],True,np.zeros(len(target),bool),np.zeros(len(target)),True)
        if not(target>0).any() or not(target<0).any():continue
        base[i:exit_origin]=target;used.add(sid)
        decisions.append(dict(session=sid,origin=int(i),signal_completed_ns=int(m['completed'][i]),exit_origin=exit_origin,
            stocks=[dict(index=int(j),ticker=m['symbols'][j],weight=float(target[j]),score=float(score[j]),own_z=float(own[j])) for j in np.flatnonzero(target)]))
    return base,decisions
