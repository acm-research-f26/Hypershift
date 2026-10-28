"""Non-promoting local numerical conditioning diagnostic for one fixed batch."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ndx_reference_training import prepare
from ndx_extended_hours import ExtendedModel
from training_device import configure_device,device_metadata
from rishi_raw_replay import load_selected_checkpoint,PARENT

DOC=Path('docs/rishi_mechanism_20261027')
LOCAL=Path('results/RISHI_SOURCE_DEVICE_TRACE_LOCAL_20261028_V2')
CAP=1.-1e-5


def scalar_map(x):
    values=np.asarray(x,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite endpoint')
    return np.sign(values)*np.arctanh(np.minimum(abs(values),CAP))


def main():
    out=DOC/'execution_batch41_logmap_precision';out.mkdir(parents=True,exist_ok=False)
    device=configure_device('cuda');p=prepare(2022);checkpoint=load_selected_checkpoint(PARENT/'2022/7.pt')
    previous=np.load(LOCAL/'FORECASTS.npz',allow_pickle=False);previous_layers=np.load(LOCAL/'ACTIVATIONS.npz',allow_pickle=False)
    ids=previous['batch_ids'];locations=np.flatnonzero(np.isin(p['test'],ids))
    with np.load(PARENT/'2022/7_predictions.npz',allow_pickle=False) as original:
        published=original['test'].copy()
    np.testing.assert_array_equal(p['test'][locations],ids)
    chunks=np.array_split(np.arange(len(p['test'])),max(1,int(np.ceil(len(p['test'])/8))))
    if not any(np.array_equal(x,locations) for x in chunks):raise ValueError('Noncanonical batch')
    x=p['features'][...,:5].transpose(0,2,1,3).copy();extra=p['features'][...,5:].reshape(len(p['features']),p['features'].shape[1],320).copy()
    eligible=p['data']['mask'][ids];captures={};outputs={}
    for name,where,dtype in [('cpu32','cpu',torch.float32),('cuda32','cuda',torch.float32),('cuda_repeat','cuda',torch.float32),('cpu64','cpu',torch.float64)]:
        model=ExtendedModel(len(p['data']['panel'].tickers),[set(g) for g in checkpoint['groups']],'after_hours_premarket')
        model.load_state_dict(checkpoint['state_dict']);model.to(device=where,dtype=dtype);model.eval();layers={};handles=[]
        def hook(key):
            def capture(module,args,value):layers[key]=value.detach().cpu().numpy().copy()
            return capture
        for key in ('core.base.spatial','core.base.last','core.base'):
            handles.append(dict(model.named_modules())[key].register_forward_hook(hook(key)))
        args=[torch.from_numpy(value[ids]).to(device=where,dtype=dtype) for value in (x,extra,p['data']['mask'].astype('float32'))]
        with torch.no_grad():outputs[name]=model(*args)[0].detach().cpu().numpy().copy()
        captures[name]=layers
        for handle in handles:handle.remove()
    np.testing.assert_array_equal(outputs['cuda32'],outputs['cuda_repeat']);np.testing.assert_array_equal(outputs['cuda32'],previous['batch'])
    for key,prior in [('core.base.spatial','layer_0003'),('core.base.last','layer_0005'),('core.base','layer_0006')]:
        np.testing.assert_array_equal(captures['cuda32'][key],previous_layers[prior])
    cpu=captures['cpu32']['core.base.last'][:,0,:,0].astype(float);gpu=captures['cuda32']['core.base.last'][:,0,:,0].astype(float)
    basecpu=captures['cpu32']['core.base'].astype(float);basegpu=captures['cuda32']['core.base'].astype(float)
    radius=np.maximum(abs(cpu),abs(gpu));delta=np.abs(cpu-gpu);mapped=np.abs(scalar_map(cpu)-scalar_map(gpu));actual=np.abs(basecpu-basegpu)
    labels=['<.9','.9-.99','.99-.999','.999-.9999','.9999-.99999','>=.99999']
    bins=np.digitize(radius,[.9,.99,.999,.9999,CAP],right=False)
    rows=[]
    for i,j in np.argwhere(eligible):
        r=float(radius[i,j]);rows.append(dict(origin=int(ids[i]),security_id=str(p['data']['panel'].tickers[j]),cpu_endpoint=cpu[i,j],cuda_endpoint=gpu[i,j],
            radius=r,band=labels[bins[i,j]],clipped=bool(r>=CAP),raw_delta=delta[i,j],mapped_delta=mapped[i,j],base_delta=actual[i,j],
            unclipped_derivative=(1/(1-r*r) if r<CAP else None),saved_error=float(abs(outputs['cuda32'][i,j]-published[locations[i],j]))))
    frame=pd.DataFrame(rows);frame.to_csv(out/'CELLS.csv',index=False)
    frame.groupby('band',sort=False).agg(cells=('radius','size'),radius_max=('radius','max'),raw_max=('raw_delta','max'),
        mapped_max=('mapped_delta','max'),base_max=('base_delta','max'),raw_median=('raw_delta','median'),
        mapped_median=('mapped_delta','median')).reset_index().to_csv(out/'BANDS.csv',index=False)
    np.savez_compressed(out/'ENDPOINTS.npz',ids=ids,security_ids=np.asarray(p['data']['panel'].tickers,dtype='U'),eligible=eligible,
        cpu_endpoint=cpu,cuda_endpoint=gpu,cpu_base=basecpu,cuda_base=basegpu,cpu64_endpoint=captures['cpu64']['core.base.last'][:,0,:,0])
    paths=[Path(__file__),DOC/'LOGMAP_PRECISION_PROTOCOL.md',PARENT/'2022/7.pt',PARENT/'2022/7_predictions.npz',LOCAL/'FORECASTS.npz',LOCAL/'ACTIVATIONS.npz',LOCAL/'TRACE.json',Path('research/models/think_model.py'),Path('research/models/ndx_extended_hours.py')]
    ref=frame[(frame.origin==186)&(frame.security_id=='ADBE')]
    if len(ref)!=1:raise ValueError('Reference cell missing')
    (out/'RESULTS.json').write_text(json.dumps(dict(environment=device_metadata(device),eligible_cells=len(frame),
        max_saved_error=float(frame.saved_error.max()),reference=ref.iloc[0].replace({np.nan:None}).to_dict(),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limits='Local numerical diagnosis only; original checkpoint acceptance unchanged'),indent=2)+'\n')
    print(frame.groupby('band',sort=False)[['raw_delta','mapped_delta','base_delta']].agg(['size','max','median']).to_string())
    print(ref.to_string(index=False))


if __name__=='__main__':main()
