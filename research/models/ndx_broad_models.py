"""Causal multi-horizon features and shared, variable-universe predictors."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from ndx_ranking_objectives import lagged_betas, rank_ic
from pit_research_engine import load_data,fold
from run_ndx_alpaca_training import feature_data


class SharedNetwork(nn.Module):
    def __init__(self,features,hidden=32):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(features,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def forward(self,x,eligible):
        mask = eligible.bool()[...,None]
        local = self.local(torch.where(mask,x,torch.zeros_like(x)))
        local = torch.where(mask,local,torch.zeros_like(local))
        context = local.sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
        score = self.head(torch.cat([local,context.expand_as(local)],dim=-1)).squeeze(-1)
        return torch.where(eligible.bool(),score,torch.zeros_like(score))


def stability(prediction,target,mask,months):
    values = rank_ic(prediction,target,mask)
    series = pd.Series(values).groupby(np.asarray(months)).mean()
    if not np.isfinite(series).all():
        raise ValueError('Invalid monthly validation ranking score')
    return float(series.median()-.25*series.std(ddof=0))


def prepare(base=None,intraday=None,dollar_volume=None,benchmark=None,verify_parent_mask=True):
    base = load_data() if base is None else base
    dates = base['panel'].dates; names = base['panel'].tickers
    price = pd.DataFrame(base['marks'],index=dates,columns=names)
    economic = base['panel'].close
    dollar_volume = (pd.read_csv(Path(base['manifest']['root'])/'dollar_volume.csv',index_col=0).to_numpy()
                     if dollar_volume is None else dollar_volume)
    dv = pd.DataFrame(dollar_volume,index=dates,columns=names)
    q = (pd.read_csv(Path(base['manifest']['root'])/'etf_adj_close.csv',index_col=0)['QQQ'].to_numpy(float)
         if benchmark is None else np.asarray(benchmark,dtype=float))
    market = pd.Series(q,index=dates); qr = market.pct_change(fill_method=None)
    returns = price.pct_change(fill_method=None); adv = dv.rolling(20).mean()
    basic = np.stack([returns,price/price.rolling(5).mean()-1,price/price.rolling(20).mean()-1,
                      returns.rolling(20).std(),np.log((dv/adv).clip(.01,100))],axis=-1)
    finite = pd.DataFrame(np.isfinite(basic).all(-1),index=dates).rolling(16).sum().eq(16).to_numpy()
    next_day = np.minimum(np.arange(len(dates))+1,len(dates)-1)
    eligible = (base['membership'][next_day]&base['tradable']&finite&
                (pd.DataFrame(base['tradable']).rolling(40).sum().to_numpy()==40)&(adv.to_numpy()>=1e6))
    if verify_parent_mask:
        np.testing.assert_array_equal(eligible[base['origin']],base['mask'])
    features,feature_names = [],[]
    def add(name,values):
        value = np.asarray(values,dtype='float32')
        if value.ndim==1:
            value = np.broadcast_to(value[:,None],price.shape)
        features.append(value); feature_names.append(name)
    for window in [1,5,10,21,63,126,252]:
        r = price/price.shift(window)-1
        add(f'return_{window}',r)
        add(f'qqq_relative_return_{window}',r.to_numpy()-(market/market.shift(window)-1).to_numpy()[:,None])
    for window in [5,20,50,200]:
        add(f'ma_distance_{window}',price/price.rolling(window).mean()-1)
    for window in [20,60,126]:
        add(f'volatility_{window}',returns.rolling(window).std())
    for window in [20,63]:
        add(f'downside_volatility_{window}',returns.clip(upper=0).rolling(window).std())
    for window in [63,252]:
        add(f'drawdown_{window}',price/price.rolling(window).max()-1)
    for window in [20,60]:
        add(f'log_relative_dollar_volume_{window}',np.log((dv/dv.rolling(window).mean()).clip(.01,100)))
    add('log_adv20',np.log(adv.clip(lower=1)))
    add('volatility_ratio_20_126',returns.rolling(20).std()/returns.rolling(126).std())
    beta,beta_support = lagged_betas(returns.to_numpy(),qr.to_numpy())
    add('lagged_shrunk_qqq_beta',beta); add('beta_history_supported',beta_support)
    for window in [5,21,63,126]:
        add(f'market_return_{window}',market/market.shift(window)-1)
    for window in [50,200]:
        add(f'market_ma_distance_{window}',market/market.rolling(window).mean()-1)
    add('market_volatility_63',qr.rolling(63).std())
    for window in [21,63,126]:
        relative = (price/price.shift(window)-1).where(eligible)
        add(f'cross_sectional_trend_rank_{window}',relative.rank(axis=1,pct=True)-.5)
    daily = np.stack(features,axis=-1)
    intraday = feature_data()[0] if intraday is None else intraday
    extra = []
    for window in [1,5,20]:
        values = intraday if window==1 else np.stack([
            pd.DataFrame(intraday[...,j]).rolling(window,min_periods=window).mean().to_numpy()
            for j in range(8)],axis=-1)
        extra.append(values)
        feature_names.extend([f'intraday_{window}_{j}' for j in range(8)])
    extra = np.concatenate(extra,axis=-1).astype('float32')
    zero = np.full_like(extra,np.nan)
    zero[...,[7,15,23]]=0.
    origin = np.arange(260,len(dates)-1)
    entry = origin+1; exit_ = np.minimum(entry+5,len(dates)-1)
    raw_target = economic[exit_]/economic[entry]-1
    observed = eligible[origin]&np.isfinite(raw_target)&((exit_-entry)==5)[:,None]
    raw_target = np.nan_to_num(raw_target)*100
    q_target = q[exit_]/q[entry]-1
    targets = dict(raw=raw_target,residual=raw_target-beta[origin]*q_target[:,None]*100)
    return dict(base=base,dates=dates,origin=origin,entry=entry,exit=exit_,mask=eligible[origin],labelmask=observed,
                views=dict(daily=np.concatenate([daily,zero],axis=-1)[origin],
                           intraday=np.concatenate([daily,extra],axis=-1)[origin]),
                targets=targets,feature_names=feature_names,
                trend_exposure=np.where(q>=market.rolling(200).mean().to_numpy(),1.,.5)[base['origin']],
                daily_trend_exposure=np.where(q>=market.rolling(200).mean().to_numpy(),1.,.5)[origin],
                benchmark=q)


def splits(data,year):
    dates = data['dates']; signal = dates[data['origin']]; outcome = dates[data['exit']]
    parent_test = fold(data['base'],year)[2]
    test = np.searchsorted(data['origin'],data['base']['origin'][parent_test])
    np.testing.assert_array_equal(data['origin'][test],data['base']['origin'][parent_test])
    cutoff = signal[test[0]]
    lower = max(pd.Timestamp('2019-07-01'),pd.Timestamp(year-5,1,1))
    train = np.flatnonzero((signal>=lower)&(outcome<pd.Timestamp(year-1,1,1)))
    validation = np.flatnonzero((signal>=pd.Timestamp(year-1,1,1))&(outcome<cutoff))
    if outcome[train].max()>=signal[validation].min() or outcome[validation].max()>=signal[test].min():
        raise ValueError('Overlapping train/validation/test labels')
    return train,validation,test


def normalize(raw,mask,train):
    values = raw[train][mask[train]]
    finite = np.isfinite(values); count = finite.sum(0)
    mean = np.divide(np.where(finite,values,0.).sum(0,dtype=float),count,out=np.zeros(raw.shape[-1]),where=count>0)
    squared = np.where(finite,(values-mean)**2,0.).sum(0)
    std = np.sqrt(np.divide(squared,count,out=np.ones_like(mean),where=count>0)).clip(1e-5)
    x = np.nan_to_num(np.clip((raw-mean)/std,-8,8),nan=0.).astype('float32')
    return x,mean,std


def predict_network(model,x,mask,ids,device,batch=32):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[sub]).to(device),torch.from_numpy(mask[sub]).to(device)).cpu().numpy()
            for sub in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch))))])
