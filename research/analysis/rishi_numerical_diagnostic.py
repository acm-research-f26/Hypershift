"""Non-promoting precision diagnostic under NUMERICAL_REPLAY_AMENDMENT."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from ndx_reference_training import prepare
from ndx_extended_hours import ExtendedModel,predict
from training_device import configure_device,device_metadata
from rishi_raw_replay import load_selected_checkpoint,PARENT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('docs/rishi_mechanism_20261027/execution_batch6_numerical'))
    parser.add_argument('--full-cuda',action='store_true')
    parser.add_argument('--reference-cuda-worst',action='store_true',help='Use the previously observed full-CUDA test maximum instead of the CPU maximum')
    args=parser.parse_args();out=args.out
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    device=configure_device('cuda');environment=device_metadata(device)
    p=prepare(2022);checkpoint=load_selected_checkpoint(PARENT/'2022/7.pt')
    previous=np.load('results/RISHI_MECHANISM_REPLAY_20261027_PORTABLE/B_2022_7.npz')['test']
    saved=np.load(PARENT/'2022/7_predictions.npz')['test']
    r,j=np.unravel_index(np.abs(previous-saved).argmax(),saved.shape)
    if args.reference_cuda_worst:
        prior=json.loads(Path('docs/rishi_mechanism_20261027/execution_batch6_numerical_full_v2/DIAGNOSTIC.json').read_text())['full_cuda']['test']
        r=int(np.flatnonzero(p['test']==prior['max_origin'])[0])
        j=int(np.flatnonzero(np.asarray(p['data']['panel'].tickers)==prior['max_security_id'])[0])
    batches=np.array_split(np.arange(len(p['test'])),max(1,int(np.ceil(len(p['test'])/8))))
    locations=next(x for x in batches if r in x);ids=p['test'][locations]
    x=p['features'][...,:5].transpose(0,2,1,3).copy()
    extra=p['features'][...,5:].reshape(len(p['features']),p['features'].shape[1],320).copy()
    mask=p['data']['mask'];outputs={};captures={}
    for name,where,dtype in [('cpu32','cpu',torch.float32),('cuda32','cuda',torch.float32),('cuda32_repeat','cuda',torch.float32),('cpu64','cpu',torch.float64)]:
        model=ExtendedModel(len(p['data']['panel'].tickers),[set(g) for g in checkpoint['groups']],'after_hours_premarket')
        model.load_state_dict(checkpoint['state_dict']);model.to(device=where,dtype=dtype);model.eval();layers={};handles=[]
        def hook(key):
            def capture(module,args,value):
                if isinstance(value,tuple):value=value[0]
                if torch.is_tensor(value):layers[key]=value.detach().cpu().double().numpy().copy()
            return capture
        for key,module in model.named_modules():
            if key:handles.append(module.register_forward_hook(hook(key)))
        tensor_args=[torch.from_numpy(value[ids]).to(device=where,dtype=dtype) for value in [x,extra,mask.astype('float32')]]
        with torch.no_grad():result=model(*tensor_args)[0].detach().cpu().double().numpy()
        outputs[name]=result;captures[name]=layers
        for handle in handles:handle.remove()
        del model
    expected=saved[locations];summary=[]
    for key,value in outputs.items():summary.append(dict(path=key,max_saved_error=float(np.max(np.abs(value-expected))),
        reference_kind='prior_full_cuda_maximum' if args.reference_cuda_worst else 'prior_cpu_maximum',
        reference_origin_id=int(p['test'][r]),reference_stock=p['data']['panel'].tickers[j],
        at_reference=float(value[np.flatnonzero(locations==r)[0],j])))
    rows=[]
    for key,a in captures['cpu32'].items():
        b=captures['cuda32'][key];double=captures['cpu64'][key]
        row=dict(module=key,shape=list(a.shape),cpu_cuda_max_error=float(np.max(np.abs(a-b))),
                 cpu32_cpu64_max_error=float(np.max(np.abs(a-double))))
        if a.ndim>0 and a.shape[-1]>1:
            norm=np.linalg.norm(a,axis=-1)
            row.update(max_last_axis_norm=float(norm.max()),fraction_norm_gt_0999=float((norm>.999).mean()))
        rows.append(row)
    value=dict(environment=environment,checkpoint='B/2022/7',diagnostic_only=True,
        batch_origin_ids=ids.tolist(),summary=summary,
        cuda_repeat_max_error=float(np.max(np.abs(outputs['cuda32']-outputs['cuda32_repeat']))),
        cpu_batch_vs_prior_max_error=float(np.max(np.abs(outputs['cpu32']-previous[locations]))),
        layers=rows,limit='FP64 is a diagnostic, not source replay. No acceptance tolerance or source implementation changed.')
    if args.full_cuda:
        model=ExtendedModel(len(p['data']['panel'].tickers),[set(g) for g in checkpoint['groups']],'after_hours_premarket')
        model.load_state_dict(checkpoint['state_dict']);model.to('cuda');model.eval()
        full={}
        with np.load(PARENT/'2022/7_predictions.npz') as original:
            for split in ['validation','test']:
                replay=predict(model,x,extra,mask,p[split],device,8)
                repeat=predict(model,x,extra,mask,p[split],device,8)
                delta=np.abs(replay-original[split]);a,b=np.unravel_index(delta.argmax(),delta.shape)
                full[split]=dict(max_error=float(delta.max()),repeat_error=float(np.max(np.abs(repeat-replay))),
                    max_origin=int(p[split][a]),max_security_id=p['data']['panel'].tickers[b])
                if split=='test':full[split]['isolated_batch_difference']=float(np.max(np.abs(replay[locations]-outputs['cuda32'])))
        value['full_cuda']=full
    (out/'DIAGNOSTIC.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='layers'},indent=2))


if __name__=='__main__':main()
