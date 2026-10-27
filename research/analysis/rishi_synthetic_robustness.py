"""Fixed missingness/membership synthetic gates; never reads market outcomes."""
import json
import time
import numpy as np
import torch
from rishi_synthetic_gate import DOC,make_model

CASES=['missing_own','coverage_null','membership_own','membership_peer']
OUT=DOC/'synthetic_robustness1'


def make_case(case):
    if case not in CASES:raise ValueError(case)
    rng=np.random.default_rng(27102029+CASES.index(case));result={}
    for split,count in [('train',128),('test',64)]:
        x=rng.normal(0,.2,(count,8,16,25)).astype('float32')
        latent=rng.normal(size=(count,8)).astype('float32')
        x[:,:,:,0]=latent[:,:,None]*.2
        missing=np.zeros((count,8),dtype=bool)
        eligible=np.ones((count,8),dtype=bool)
        if case in ['missing_own','coverage_null']:
            missing=rng.random((count,8))<(.25 if split=='train' else .60)
            x[missing, :,5:24]=0
            x[:,:,:,24]=np.where(missing,0.,.2)[:,:,None]
            if case=='coverage_null':x[:,:,:,0]=0
        else:
            low,high=(6,8) if split=='train' else (4,6)
            eligible[:]=False;eligible[:,:2]=True
            for j in range(count):
                active=int(rng.integers(low,high+1))
                eligible[j,rng.choice(np.arange(2,8),active-2,replace=False)]=True
            x[~eligible]=np.nan
        target=latent[:,np.arange(8)^1].copy() if case=='membership_peer' else latent.copy()
        supervised=eligible&eligible[:,np.arange(8)^1] if case=='membership_peer' else eligible.copy()
        result[split]={k:torch.from_numpy(v) for k,v in dict(x=x,latent=latent,target=target,
            eligible=eligible,supervised=supervised,missing=missing).items()}
    return result


def masked_score(prediction,target,supervised):
    p,y=prediction[supervised],target[supervised]
    if not len(p) or not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Invalid supervised values')
    mse=float((p-y).square().mean());zero=float(y.square().mean())
    return dict(mse=mse,zero_mse=zero,r2_zero=1-mse/zero,observations=len(p))


def forecast(model,x,eligible,arm):
    if arm=='full':return model(x,None,eligible)[0]
    m=model.network
    x=torch.where(eligible[:,None,None],x,torch.zeros_like(x))
    x=m.positional_encoding(m.feature_embedding(x));x=m.individual_attn(x)
    x=m.mixing_layer(torch.cat([x,m.intensity_func(x)],dim=-1));x,_=m.temporal_attn(x)
    return torch.where(eligible,m.mlp(x),torch.zeros(len(x)))


def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Preserve previous robustness evidence')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    started=time.monotonic();records=[]
    for case in CASES:
        data=make_case(case)
        for seed in [7,19,42]:
            for arm in ['full','temporal_only']:
                model=make_model(seed);opt=torch.optim.Adam(model.parameters(),lr=.0001)
                fit_start=time.monotonic()
                for step in range(800):
                    if time.monotonic()-started>900:
                        (OUT/'INCOMPLETE.json').write_text(json.dumps(dict(reason='runtime cap',case=case,seed=seed,arm=arm,
                            completed_fits=len(records),partial_updates=step)))
                        return
                    model.train();opt.zero_grad(set_to_none=True);j=step%128;p=data['train']
                    score=forecast(model,p['x'][j],p['eligible'][j],arm);s=p['supervised'][j]
                    loss=torch.sqrt((score[s]-p['target'][j,s]).square().mean()+1e-8)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite masked synthetic loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    if step==19 and not records:
                        t=time.monotonic()-fit_start
                        smoke=dict(steps=20,seconds=t,projected_total_seconds=t/20*24*800,
                            scope='Resource smoke before held-out evaluation, no recipe changes')
                        (OUT/'RUNTIME_SMOKE.json').write_text(json.dumps(smoke,indent=2));print(json.dumps(smoke),flush=True)
                model.eval();p=data['test']
                with torch.no_grad():
                    prediction=torch.stack([forecast(model,x,e,arm) for x,e in zip(p['x'],p['eligible'])])
                    if not torch.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
                    bad=p['x'].clone();bad[~p['eligible']]=12345.
                    alternative=torch.stack([forecast(model,x,e,arm) for x,e in zip(bad,p['eligible'])])
                    torch.testing.assert_close(prediction,alternative,rtol=0,atol=0)
                row=dict(case=case,seed=seed,arm=arm,updates=800,seconds=time.monotonic()-fit_start,
                    train_active_mean=float(data['train']['eligible'].sum(1).float().mean()),
                    test_active_mean=float(p['eligible'].sum(1).float().mean()),
                    train_missing_rate=float(data['train']['missing'].float().mean()),
                    test_missing_rate=float(p['missing'].float().mean()),
                    inactive_nan_garbage_invariance=True,**masked_score(prediction,p['target'],p['supervised']))
                records.append(row);(OUT/f'{case}_{seed}_{arm}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    (OUT/'RESULTS.json').write_text(json.dumps(dict(status='complete',fits=records,seconds=time.monotonic()-started,
        scope='Synthetic observable-signal recovery and negative controls only; not financial signal attribution'),indent=2))


if __name__=='__main__':main()
