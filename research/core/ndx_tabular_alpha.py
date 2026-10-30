"""Shared causal predictors: long-history styles plus existing AH/PM inputs."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def causal_styles(data,qqq):
    price=pd.DataFrame(data['marks']);ret=pd.DataFrame(data['returns'])
    q=pd.Series(qqq);qr=q.pct_change(fill_method=None)
    values=[];names=[]
    def add(name,value):names.append(name);values.append(np.asarray(value,float))
    for h in (1,5,10,21,63,126,252):add(f'momentum_{h}',price/price.shift(h)-1)
    add('momentum_252_skip21',price.shift(21)/price.shift(252)-1)
    vol={h:ret.rolling(h,min_periods=h).std()*np.sqrt(252) for h in (5,20,60,126)}
    for h,value in vol.items():add(f'volatility_{h}',value)
    for h in (21,63,126,252):add(f'risk_scaled_momentum_{h}',(price/price.shift(h)-1)/vol[20].clip(lower=.05))
    for h in (63,252):add(f'drawdown_{h}',price/price.rolling(h,min_periods=h).max()-1)
    for h in (20,60,200):add(f'mean_distance_{h}',price/price.rolling(h,min_periods=h).mean()-1)
    add('volatility_ratio_5_60',vol[5]/vol[60].clip(lower=.05))
    add('volatility_ratio_20_126',vol[20]/vol[126].clip(lower=.05))
    add('downside_volatility_20',np.sqrt(ret.clip(upper=0).pow(2).rolling(20,min_periods=20).mean()*252))
    add('log_adv20',np.log(np.maximum(data['adv'],1)))
    covariance=ret.mul(qr,axis=0).rolling(60,min_periods=60).mean()-ret.rolling(60,min_periods=60).mean().mul(qr.rolling(60,min_periods=60).mean(),axis=0)
    add('beta_60',covariance.div(qr.rolling(60,min_periods=60).var(ddof=0).clip(lower=1e-8),axis=0))
    context=np.stack([(q/q.shift(21)-1).to_numpy(),(q/q.shift(63)-1).to_numpy(),
                      (qr.rolling(20,min_periods=20).std()*np.sqrt(252)).to_numpy()],axis=-1)
    return np.stack(values,axis=-1),names,context


def feature_cube(data,x,extra,qqq):
    styles,names,context=causal_styles(data,qqq)
    styles=styles[data['origin']];context=context[data['origin']]
    sequence=np.concatenate((x.transpose(0,2,1,3),extra.reshape(len(extra),extra.shape[1],16,20)),axis=-1)
    # All AH/PM channels already use the frozen train-only original scaler.
    summaries=np.concatenate((sequence[:,:,-1],sequence[:,:,-5:].mean(2),sequence.mean(2)),axis=-1)
    raw=np.concatenate((styles,summaries),axis=-1)
    ranks=np.full_like(raw,.5,dtype=float)
    for i in range(len(raw)):
        for j in range(raw.shape[-1]):
            mask=data['mask'][i]&np.isfinite(raw[i,:,j])
            if mask.any():ranks[i,mask,j]=rankdata(raw[i,mask,j])/mask.sum()
    context=np.broadcast_to(context[:,None,:],(*styles.shape[:2],3))
    features=np.concatenate((styles,ranks,context),axis=-1).astype('float32')
    sequence_names=[f'{summary}_channel{j}' for summary in ('last','mean5','mean16') for j in range(25)]
    feature_names=names+['rank_'+n for n in names+sequence_names]+['qqq_momentum21','qqq_momentum63','qqq_vol20']
    assert features.shape[-1]==len(feature_names)==130
    return features,feature_names


def standardize_train_only(features,train,observed):
    values=features[train][observed[train]]
    mean=np.nanmean(np.where(np.isfinite(values),values,np.nan),axis=0)
    mean=np.nan_to_num(mean)
    clean=np.where(np.isfinite(values),values,mean)
    sd=np.maximum(clean.std(0),1e-5)
    result=np.clip((np.where(np.isfinite(features),features,mean)-mean)/sd,-8,8).astype('float32')
    assert np.isfinite(result).all()
    return result,mean,sd


def normalized_labels(y,observed):
    count=observed.sum(1,keepdims=True).clip(1)
    mean=np.where(observed,y,0).sum(1,keepdims=True)/count
    clean=np.where(observed,y-mean,0)
    sd=np.maximum(np.sqrt((clean**2).sum(1,keepdims=True)/count),.1)
    return np.where(observed,np.clip(clean/sd,-3,3),0).astype('float32')
