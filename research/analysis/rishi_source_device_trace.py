"""Transferable no-training original parent checkpoint GPU trace."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from ndx_reference_training import prepare
from ndx_extended_hours import ExtendedModel,predict
from training_device import configure_device,device_metadata


def array_digest(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(value.dtype).encode());h.update(json.dumps(value.shape).encode());h.update(value.tobytes());return h.hexdigest()


def batch_for_origin(ids,batch,origin):
    ids=np.asarray(ids)
    if int(np.sum(ids==origin))!=1:raise ValueError('Expected unique diagnostic origin')
    return next(part for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch)))) if origin in part)


def safe_load(path):
    allowed={'numpy.dtype','numpy._core.multiarray.scalar'}
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(path))-allowed:raise ValueError('Unexpected checkpoint globals')
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Int64DType]):
        return torch.load(path,map_location='cpu',weights_only=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose a fresh trace directory')
    args.out.mkdir(parents=True)
    root=Path('runs/ndx_extended_hours_20261024/after_hours_premarket/2022');cp_path=root/'7.pt';export_path=root/'7_predictions.npz'
    device=configure_device('cuda');env=device_metadata(device);p=prepare(2022);data=p['data'];n=len(data['panel'].tickers)
    if list(data['panel'].tickers).count('ADBE')!=1:raise ValueError('Expected unique ADBE security ID')
    x=p['features'][...,:5].transpose(0,2,1,3).copy();extra=p['features'][...,5:].reshape(len(p['features']),n,320).copy()
    mask=data['mask'];ids=batch_for_origin(p['test'],8,186)
    cp=safe_load(cp_path);model=ExtendedModel(n,[set(g) for g in cp['groups']],'after_hours_premarket')
    model.load_state_dict(cp['state_dict']);model.to(device);model.eval()
    captured={};metadata=[];handles=[];calls={}
    def hook(name):
        def capture(module,inputs,value):
            if isinstance(value,tuple):value=value[0]
            if torch.is_tensor(value):
                number=calls.get(name,0);calls[name]=number+1;key=f'layer_{len(metadata):04d}'
                arr=value.detach().cpu().numpy().copy();captured[key]=arr
                metadata.append(dict(key=key,module=name,call=number,shape=list(arr.shape),dtype=str(arr.dtype)))
        return capture
    for name,module in model.named_modules():
        if name:handles.append(module.register_forward_hook(hook(name)))
    tensors=[torch.from_numpy(v[ids]).to(device) for v in (x,extra,mask.astype('float32'))]
    with torch.no_grad():batch=model(*tensors)[0].cpu().numpy()
    for h in handles:h.remove()
    outputs={}
    for split in ('validation','test'):outputs[split]=predict(model,x,extra,mask,p[split],device,8)
    with np.load(export_path,allow_pickle=False) as saved:
        errors={split:float(np.max(np.abs(outputs[split]-saved[split]))) for split in outputs}
    positions=np.flatnonzero(np.isin(p['test'],ids));difference=float(np.max(np.abs(outputs['test'][positions]-batch)))
    np.savez_compressed(args.out/'ACTIVATIONS.npz',**captured)
    np.savez_compressed(args.out/'FORECASTS.npz',validation=outputs['validation'],test=outputs['test'],batch=batch,
        validation_ids=p['validation'],test_ids=p['test'],batch_ids=ids,security_ids=np.asarray(data['panel'].tickers,dtype='U'))
    files=[cp_path,export_path,Path(__file__)]
    modules=['ndx_reference_training','ndx_extended_hours','training_device','robust_models','research_models','think_model',
        'pit_research_engine','ndx_loss_optimizer','run_robust_research','prepare_ndx_extended_hours']
    code_hashes={name:hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest() for name in modules}
    report=dict(environment=env,year=2022,seed=7,family='B',reference_origin=186,reference_ticker='ADBE',
        batch_ids=ids.tolist(),layers=metadata,all_input_hashes={k:array_digest(v) for k,v in dict(x=x,extra=extra,mask=mask,validation_ids=p['validation'],test_ids=p['test'],stock_ids=np.asarray(data['panel'].tickers,dtype='U')).items()},
        file_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in files},code_modules_sha256=code_hashes,forecast_max_error_pp=errors,batch_vs_full_max_error=difference,
        limits='One checkpoint; trace only, no training or gate amendment; source machine verification remains separate.')
    (args.out/'TRACE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'layers':len(metadata),'batch_ids':ids.tolist(),'max_error_pp':errors,'batch_vs_full_max_error':difference},indent=2))


if __name__=='__main__':main()