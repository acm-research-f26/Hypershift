"""Matched original-core models with causally aligned extended-hours branches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from pit_research_engine import load_data,fold
from prepare_ndx_extended_hours import OUT as FEATURES,FIELDS,sha
from robust_models import make_model
from run_robust_research import scale_inputs


def prepared():
    review=json.loads((FEATURES/'REVIEW.json').read_text())
    if review['status']!='passed_extended_hours_input_gate' or sha(FEATURES/'features.npz')!=review['feature_sha256']:
        raise ValueError('Extended-hours review missing or changed')
    data=load_data()
    with np.load(FEATURES/'features.npz') as saved:
        np.testing.assert_array_equal(saved['dates'],data['panel'].dates.strftime('%Y-%m-%d'))
        np.testing.assert_array_equal(saved['tickers'],data['panel'].tickers)
        cubes={s:saved[s].copy() for s in ['after_hours','premarket']}
    return data,cubes


def raw_auxiliary(data,cubes,view):
    shape=(*data['x'].shape[:-1],2*len(FIELDS));result=np.zeros(shape,dtype='float32')
    if view not in ['daily','after_hours','after_hours_premarket']:raise ValueError('Unknown information view')
    if view!='daily':
        result[...,:len(FIELDS)]=np.stack([cubes['after_hours'][t-15:t+1] for t in data['origin']])
    if view=='after_hours_premarket':
        # Same 16 daily slots, with the following EXCHANGE session's PM window.
        result[...,len(FIELDS):]=np.stack([cubes['premarket'][t-14:t+2] for t in data['origin']])
    return result


def scale_auxiliary(raw,mask,train):
    values=raw[train].transpose(0,2,1,3)[mask[train]];finite=np.isfinite(values)
    count=finite.sum((0,1),keepdims=True)
    mean=np.divide(np.where(finite,values,0).sum((0,1),keepdims=True,dtype=float),count,
                   out=np.zeros((1,1,raw.shape[-1])),where=count>0)
    variance=np.divide(np.where(finite,(values-mean)**2,0).sum((0,1),keepdims=True),count,
                       out=np.ones_like(mean),where=count>0)
    std=np.sqrt(variance).clip(1e-5)
    x=(np.nan_to_num(np.clip((raw-mean)/std,-8,8),nan=0)*.2).astype('float32')
    return x.transpose(0,2,1,3).reshape(len(x),x.shape[2],-1),mean,std


def timing(data,calendar):
    dates=data['panel'].dates[data['entry']]
    clock=lambda value:pd.to_datetime(dates.strftime('%Y-%m-%d')+' '+value).tz_localize('America/New_York').tz_convert('UTC')
    cutoff=clock('09:15');decision=clock('09:20');fill=pd.DatetimeIndex(calendar.close_utc.reindex(dates))
    if not ((cutoff<decision)&(decision<fill)).all():raise ValueError('Feature/decision/execution chronology failure')
    return pd.DataFrame(dict(origin=data['origin'],entry=data['entry'],
        prior_regular_close=data['panel'].dates[data['origin']].strftime('%Y-%m-%d'),
        feature_cutoff_utc=cutoff.astype(str),decision_utc=decision.astype(str),fill_utc=fill.astype(str)))


class ExtendedModel(nn.Module):
    def __init__(self,nodes,groups,view):
        super().__init__();self.view=view
        # Instantiate core FIRST so its initialization matches every old seed.
        self.core=make_model('think_mix',nodes,groups,causal_membership=True)
        self.first=nn.Linear(16*2*len(FIELDS),16)
        self.last=nn.Linear(16,1,bias=False);nn.init.zeros_(self.last.weight)

    def forward(self,x,extra,eligible):
        score,_=self.core(x,eligible)
        if self.view=='daily':return score,None
        mask=eligible.bool();count=mask.sum(1,keepdim=True).clamp_min(1)
        clean=torch.where(mask,score.detach(),torch.zeros_like(score))
        mean=clean.sum(1,keepdim=True)/count
        sd=(torch.where(mask,(clean-mean)**2,torch.zeros_like(clean)).sum(1,keepdim=True)/count).sqrt().clamp_min(1e-4)
        extra=torch.where(mask[...,None],extra,torch.zeros_like(extra))
        u=torch.tanh(self.last(F.gelu(self.first(extra))).squeeze(-1))
        u=torch.where(mask,u,torch.zeros_like(u));center=u.sum(1,keepdim=True)/count
        delta=.5*sd*torch.where(mask,u-center,torch.zeros_like(u))
        return score+delta,None


def predict(model,x,extra,mask,ids,device,batch=8):
    model.eval();result=[]
    with torch.no_grad():
        for sub in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch)))):
            args=[torch.from_numpy(value[sub]).to(device) for value in [x,extra,mask.astype('float32')]]
            result.append(model(*args)[0].cpu().numpy())
    return np.concatenate(result)
