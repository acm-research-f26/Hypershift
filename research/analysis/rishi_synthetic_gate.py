"""Actual dynamic architecture recovery on frozen synthetic tasks; no market data."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
import ndx_reference_models as reference

ROOT=Path.cwd()
DOC=ROOT/'docs/rishi_mechanism_20261027'
OUT=DOC/'synthetic_gate1'


def make_data():
    rng=np.random.default_rng(27102028)
    result={}
    identity=np.linspace(-1,1,8,dtype='float32')
    identity/=np.sqrt(np.mean(identity**2))
    for split,count in [('train',128),('test',64)]:
        x=rng.normal(0,.2,(count,8,16,25)).astype('float32')
        signal=rng.normal(size=(count,8)).astype('float32')
        x[:,:,:,0]=signal[:,:,None]*.2
        result[split+'_x']=torch.from_numpy(x)
        result[split+'_y']={
            'own':torch.from_numpy(x[:,:,0,0]/.2),
            'peer':torch.from_numpy((x[:,:,0,0]/.2)[:,np.arange(8)^1].copy()),
            'identity':torch.from_numpy(np.broadcast_to(identity,(count,8)).copy())}
    return result


def score_recovery(prediction,target):
    mse=float((prediction-target).square().mean())
    baseline=float(target.square().mean())
    return dict(mse=mse,zero_mse=baseline,r2_zero=1-mse/baseline)


def make_model(seed):
    torch.manual_seed(seed)
    source=DOC/'source_snapshot'
    spec=importlib.util.spec_from_file_location('synthetic_gate_pinned_layers',source/'layer.py')
    layers=importlib.util.module_from_spec(spec);spec.loader.exec_module(layers)
    cls=reference.definitions(source/'model.py',['Model'],vars(layers))['Model']
    original=reference.dynamic_class
    try:
        reference.dynamic_class=lambda:cls
        return reference.make_reference('dynamic_restricted',8,reference.wavelet_snapshot([{j,j+1} for j in [0,2,4,6]],8))
    finally:
        reference.dynamic_class=original


def forecast(model,x,arm):
    eligible=torch.ones(8,dtype=torch.bool)
    if arm=='full':return model(x,None,eligible)[0]
    m=model.network
    x=m.positional_encoding(m.feature_embedding(x))
    x=m.individual_attn(x)
    x=m.mixing_layer(torch.cat([x,m.intensity_func(x)],dim=-1))
    x,_=m.temporal_attn(x)
    return m.mlp(x)


def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Preserve existing synthetic batch')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data=make_data();started=time.monotonic();records=[]
    for task in ['own','peer','identity']:
        for seed in [7,19,42]:
            for arm in ['full','temporal_only']:
                if time.monotonic()-started>900:
                    (OUT/'INCOMPLETE.json').write_text(json.dumps(dict(reason='fixed runtime cap',finished=len(records))))
                    return
                model=make_model(seed)
                optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
                fit_start=time.monotonic()
                for step in range(800):
                    model.train();optimizer.zero_grad(set_to_none=True)
                    j=step%128
                    output=forecast(model,data['train_x'][j],arm)
                    loss=torch.sqrt((output-data['train_y'][task][j]).square().mean()+1e-8)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite synthetic loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if step==19 and not records:
                        seconds=time.monotonic()-fit_start
                        smoke=dict(steps=20,seconds=seconds,projected_total_seconds=seconds/20*18*800,
                                   scope='Runtime estimate before any held-out evaluation; fixed recipe unchanged')
                        (OUT/'RUNTIME_SMOKE.json').write_text(json.dumps(smoke,indent=2))
                        print(json.dumps(smoke),flush=True)
                model.eval()
                with torch.no_grad():
                    pred=torch.stack([forecast(model,x,arm) for x in data['test_x']])
                result=dict(task=task,seed=seed,arm=arm,updates=800,seconds=time.monotonic()-fit_start,
                            **score_recovery(pred,data['test_y'][task]))
                records.append(result)
                (OUT/f'{task}_{seed}_{arm}.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(result),flush=True)
    (OUT/'RESULTS.json').write_text(json.dumps(dict(status='complete',fits=records,
        seconds=time.monotonic()-started,scope='Synthetic recovery only; not financial performance or trained-checkpoint attribution'),indent=2))


if __name__=='__main__':main()
