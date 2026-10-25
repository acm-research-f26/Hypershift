"""Causal common feature adapter and finite reference-network training."""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ndx_loss_optimizer import prepared_fold
from ndx_reference_models import ROOT,make_reference,wavelet_snapshot
from run_ndx_alpaca_training import sha,write
from training_device import configure_device,tensor_on

RUN=ROOT/'runs/ndx_reference_networks_20261024'
CONFIG=ROOT/'config/ndx_reference_networks_20261024.json'
KINDS=['master','dynamic_restricted','prism']


def factor_history(dates):
    path=ROOT/'data/nasdaq_monthly_pit/jkp_themes_20261024/usa_all_themes_daily_vw_cap.zip'
    with zipfile.ZipFile(path) as z:frame=pd.read_csv(z.open(z.namelist()[0]),parse_dates=['date'])
    frame=frame[(frame.location=='usa')&(frame.freq=='daily')&(frame.weighting=='vw_cap')]
    wide=frame.pivot(index='date',columns='name',values='ret').sort_index().reindex(dates)
    if wide.shape[1]!=13:raise ValueError('Expected thirteen JKP themes')
    # Match the author's lagged twenty-session cumulative-return construction.
    return np.expm1(np.log1p(wide.shift(1)).rolling(20,min_periods=20).sum()).to_numpy(dtype='float32')


def prepare(year):
    data,x,extra,tr,va,te,groups=prepared_fold(year)
    features=np.concatenate([x.transpose(0,2,1,3),extra.reshape(len(extra),extra.shape[1],16,20)],axis=-1)
    root=Path(data['manifest']['root']);prices=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)
    volume=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0,parse_dates=True)
    columns=[]
    for ticker in ['SPY','QQQ']:
        p=prices[ticker];r=p.pct_change(fill_method=None);dv=volume[ticker]
        columns.extend([r,p/p.rolling(5).mean()-1,p/p.rolling(20).mean()-1,r.rolling(20).std(),np.log((dv/dv.rolling(20).mean()).clip(.01,100))])
    market=np.stack(columns,axis=-1)
    market=np.stack([market[t-15:t+1] for t in data['origin']])
    mean=market[tr].mean((0,1),keepdims=True);std=market[tr].std((0,1),keepdims=True).clip(1e-5)
    market=(np.clip((market-mean)/std,-8,8)*.2).astype('float32')
    prior=factor_history(data['panel'].dates)[data['origin']][:,None,:]
    required=np.concatenate([tr,va,te])
    if not np.isfinite(features[required]).all() or not np.isfinite(market[required]).all() or not np.isfinite(prior[required]).all():
        raise ValueError('Nonfinite reference input')
    close=data['panel'].close;future=np.full((*data['y'].shape,10),np.nan,dtype='float32')
    for h in range(1,11):
        valid=data['entry']+h<len(close)
        future[valid,:,h-1]=(close[data['entry'][valid]+h]/close[data['entry'][valid]]-1)*100
    futuremask=data['mask']&np.isfinite(future).all(-1)
    dates=data['panel'].dates;train_boundary=dates[data['origin'][va[0]]];validation_boundary=dates[data['origin'][te[0]]]
    def purged(sub,boundary):
        ends=data['entry'][sub]+10
        return sub[(ends<len(dates))&(dates[np.minimum(ends,len(dates)-1)]<boundary)]
    ptr=purged(tr,train_boundary);pva=purged(va,validation_boundary)
    snapshot=wavelet_snapshot(groups,len(data['panel'].tickers))
    return dict(data=data,features=features,market=market,prior=prior,train=tr,validation=va,test=te,
                pretrain=ptr,prevalidation=pva,future=future,futuremask=futuremask,snapshot=snapshot,
                market_mean=mean,market_std=std)


def inputs(p,i,kind,device):
    x=tensor_on(p['features'][i],device)
    context=tensor_on(p['prior' if kind=='prism' else 'market'][i],device)
    eligible=tensor_on(p['data']['mask'][i],device)
    return x,context,eligible


def predictions(model,p,ids,kind,device):
    model.eval();output=[]
    with torch.no_grad():
        for i in ids:output.append(model(*inputs(p,i,kind,device))[0].cpu().numpy())
    return np.stack(output)


def train_one(p,kind,year,seed,plan):
    folder=RUN/kind/str(year);folder.mkdir(parents=True,exist_ok=True);record=folder/f'{seed}_record.json'
    if record.exists():
        result=json.loads(record.read_text())
        for name,h in result['hashes'].items():
            if sha(folder/name)!=h:raise ValueError('Changed reference fit')
        return result
    device=configure_device('cuda');torch.manual_seed(seed)
    model=make_reference(kind,p['features'].shape[1],p['snapshot']).to(device)
    histories={};states={};epochs={}
    stages=['representation','prediction'] if kind=='prism' else ['prediction']
    for stage in stages:
        if kind=='prism' and stage=='prediction':model.freeze_representation()
        if kind=='prism':opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.0001,weight_decay=1e-5)
        else:opt=torch.optim.Adam(model.parameters(),lr=.00001 if kind=='master' else .0001)
        tr=p['pretrain'] if stage=='representation' else p['train']
        va=p['prevalidation'] if stage=='representation' else p['validation']
        steps=len(tr)*plan['epochs'];step=0;best=np.inf;chosen=0;stale=0;state=None;history=[]
        for epoch in range(1,plan['epochs']+1):
            model.train();losses=[]
            for i in tr:
                opt.zero_grad(set_to_none=True);x,context,eligible=inputs(p,i,kind,device)
                if kind=='prism':
                    warmup=max(1,int(.05*steps));factor=step/warmup if step<warmup else .5*(1+np.cos(np.pi*(step-warmup)/max(1,steps-warmup)))
                    for group in opt.param_groups:group['lr']=.0001*factor
                if stage=='representation':
                    loss=model.pretrain_loss(x,context,tensor_on(p['future'][i],device),eligible)
                else:
                    score,aux=model(x,context,eligible);mask=tensor_on(p['data']['labelmask'][i],device)
                    target=tensor_on(p['data']['y'][i]*100,device)
                    mse=(score-target).square()[mask].mean()
                    loss=torch.sqrt(mse+1e-8) if kind=='dynamic_restricted' else mse
                    if kind=='prism':loss=loss+.001*aux
                if not torch.isfinite(loss):raise ValueError(f'Nonfinite training: {kind}/{year}/{seed}/{stage}')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite reference gradient')
                opt.step();step+=1;losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                if stage=='representation':
                    values=[]
                    for i in va:
                        x,context,eligible=inputs(p,i,kind,device)
                        values.append(float(model.pretrain_loss(x,context,tensor_on(p['future'][i],device),eligible)))
                    value=float(np.mean(values))
                else:
                    score=predictions(model,p,va,kind,device)
                    value=float(((score-p['data']['y'][va]*100)**2)[p['data']['labelmask'][va]].mean())
            history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_loss=value))
            if not np.isfinite(value):raise ValueError('Nonfinite reference validation')
            if value<best-1e-7:
                best=value;chosen=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=plan['patience']:break
        model.load_state_dict(state);histories[stage]=history;epochs[stage]=chosen
        if stage=='representation':states[stage]=state
    vp,tp=[predictions(model,p,p[sub],kind,device) for sub in ['validation','test']]
    if not np.isfinite(tp).all():raise ValueError('Nonfinite reference forecasts')
    cp=folder/f'{seed}.pt';pred=folder/f'{seed}_predictions.npz';hp=folder/f'{seed}_history.json'
    torch.save(dict(kind=kind,year=year,seed=seed,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    representation_state=states.get('representation'),epochs=epochs),cp)
    np.savez_compressed(pred,validation=vp,test=tp);write(hp,histories)
    result=dict(kind=kind,year=year,seed=seed,chosen_epochs=epochs,parameter_count=sum(v.numel() for v in model.parameters()),
                validation_mse=float(((vp-p['data']['y'][p['validation']]*100)**2)[p['data']['labelmask'][p['validation']]].mean()),
                hashes={p.name:sha(p) for p in [cp,pred,hp]})
    write(record,result);return result
