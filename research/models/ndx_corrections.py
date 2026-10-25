"""Frozen-core corrections, chronological gates and fully disclosed proxies."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F

from ndx_broad_models import prepare, normalize
from pit_research_engine import fold
from robust_models import make_model
from run_ndx_alpaca_training import feature_data
from run_robust_research import ensemble_predictions
from research_engine import target_weights


def prepared():
    broad=prepare(); base=broad['base']; locations=np.searchsorted(broad['origin'],base['origin'])
    np.testing.assert_array_equal(broad['origin'][locations],base['origin'])
    feature=feature_data()[0]
    windows=np.stack([feature[t-15:t+1] for t in base['origin']])
    intraday=windows.transpose(0,2,1,3).reshape(len(windows),windows.shape[2],-1)
    spy=pd.read_csv(Path(base['manifest']['root'])/'etf_adj_close.csv')['SPY'].to_numpy()
    return dict(base=base,raw=dict(intraday_window=intraday,rich_features=broad['views']['intraday'][locations]),
                spy_active=base['y']*100-(spy[base['exit']]/spy[base['entry']]-1)[:,None]*100,
                trend_exposure=broad['trend_exposure'])


def partitions(base,year):
    train,val,test=fold(base,year)
    dates=base['panel'].dates
    month=dates[base['origin'][val]].month
    h2=val[month>=7]
    h1=val[(month<=6)&(base['exit'][val]<base['origin'][h2[0]])]
    if not len(h1) or not len(h2) or max(base['exit'][h1])>=min(base['origin'][h2]):
        raise ValueError('Validation chronology/purge failure')
    return train,val,test,h1,h2


class Correction(nn.Module):
    def __init__(self,features,hidden=16):
        super().__init__()
        self.first=nn.Linear(features,hidden)
        self.last=nn.Linear(hidden,1,bias=False)
        nn.init.zeros_(self.last.weight)

    def forward(self,x,core,eligible):
        eligible=eligible.bool(); count=eligible.sum(1,keepdim=True).clamp_min(1)
        core=core.detach()
        clean=torch.where(eligible,core,torch.zeros_like(core))
        mean=clean.sum(1,keepdim=True)/count
        variance=torch.where(eligible,(clean-mean).square(),torch.zeros_like(clean)).sum(1,keepdim=True)/count
        scale=variance.sqrt().clamp_min(1e-4)
        x=torch.where(eligible[...,None],x,torch.zeros_like(x))
        u=torch.tanh(self.last(F.gelu(self.first(x))).squeeze(-1))
        u=torch.where(eligible,u,torch.zeros_like(u))
        delta=.5*scale*torch.where(eligible,u-u.sum(1,keepdim=True)/count,torch.zeros_like(u))
        return clean+delta,delta,scale


def correction_loss(score,delta,scale,target,eligible,observed,active,kind):
    selected=eligible.bool()&observed.bool()
    anchor=.01*((delta/scale).square()*eligible).sum()/eligible.sum().clamp_min(1)
    if kind=='top_rank':
        # Percentile gains avoid the negative-return gain ambiguity. Rank order
        # uses only observed labels; candidate-position discounts include every
        # eligible stock, including any whose eventual label is unavailable.
        wins=selected[:,:,None]&selected[:,None,:]&(target[:,:,None]>target[:,None,:])
        rank=1+(selected[:,None,:]&(target[:,None,:]<target[:,:,None])).sum(2)
        gain=(rank-1)/selected.sum(1,keepdim=True).clamp_min(1)
        order=torch.argsort(score.detach().masked_fill(~eligible.bool(),-torch.inf),dim=1,descending=True,stable=True)
        positions=torch.empty_like(order); positions.scatter_(1,order,torch.arange(score.shape[1],device=score.device)[None,:].expand_as(order))
        discount=torch.where(positions<10,1/torch.log2(positions.float()+2),0.)
        ideal=torch.sort(gain.masked_fill(~selected,0.),dim=1,descending=True).values
        d=1/torch.log2(torch.arange(score.shape[1],device=score.device).float()+2)
        d[10:]=0
        idcg=(ideal*d).sum(1).clamp_min(1e-6)
        weight=(gain[:,:,None]-gain[:,None,:]).abs()*(discount[:,:,None]-discount[:,None,:]).abs()/idcg[:,None,None]
        weight=torch.where(wins,weight,0.)
        pair=F.softplus(-(score[:,:,None]-score[:,None,:])/scale[:,:,None])
        return ((pair*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-6)).mean()+anchor
    if kind!='net_utility':
        raise ValueError('Unknown correction objective')
    weights=torch.softmax((score/scale/.5).masked_fill(~eligible.bool(),-torch.inf),dim=1)
    complete=(~eligible.bool()|observed.bool()).all(1)
    if not complete.any():
        return anchor
    utility=(weights*torch.where(selected,active,torch.zeros_like(active))).sum(1)[complete]
    concentration=F.relu(weights[complete].square().sum(1)-.1).mean()
    # Consecutive chronological origins only, with observed returns at both.
    adjacent=complete[1:]&complete[:-1]
    turnover=(weights[1:]-weights[:-1]).abs().sum(1)[adjacent].mean() if adjacent.any() else weights.sum()*0
    return -utility.mean()+.1*utility.var(unbiased=False)+concentration+.1*turnover+anchor


def top_weights(prediction,mask):
    # Exactly the account engine's fractional boundary-tie rule.
    return np.stack([target_weights(p,m,10,'equal') for p,m in zip(prediction,mask)])


def proxy(prediction,core,mask,labelmask,y):
    weights=top_weights(prediction,mask); baseline=top_weights(core,mask)
    complete=((weights==0)|labelmask).all(1)&((baseline==0)|labelmask).all(1)
    def values(w):
        turnover=np.abs(np.diff(w,axis=0,prepend=np.zeros_like(w[:1]))).sum(1)
        return (w*np.where(labelmask,y,0.)).sum(1)-.1*turnover
    delta=(values(weights)-values(baseline))[complete]
    return dict(observations=int(complete.sum()),omitted=int((~complete).sum()),
                mean=float(delta.mean()) if len(delta) else None,
                variance=float(delta.var()) if len(delta) else None,
                wins=int((delta>0).sum()),score=float(delta.mean()-.1*delta.var()) if len(delta) else -1e30)


def choose(candidates,statistics):
    # Zero is first and cannot be displaced by a numerical tie.
    best=dict(candidate=None,statistics=dict(score=0.))
    for candidate,statistic in zip(candidates,statistics):
        if statistic['score']>best['statistics']['score']+1e-7:
            best=dict(candidate=candidate,statistics=statistic)
    return best


def accepted(statistic,candidate):
    return bool(candidate is not None and statistic['observations']>=12 and
                statistic['mean']>0 and statistic['wins']*2>=statistic['observations'])


def core_model(nodes,groups):
    return make_model('think_mix',nodes,groups,causal_membership=True)


def normalized(data,train):
    return {name:normalize(raw,data['base']['mask'],train) for name,raw in data['raw'].items()}


def ensemble(available,mask):
    return ensemble_predictions(available,mask)
