"""Matched objective training; no test scoring before all 48 fits complete."""
from pathlib import Path
import json,hashlib,time
import numpy as np,torch
from ndx_loss_optimizer import prepared_fold,optimizer_for
from ndx_extended_hours import ExtendedModel,predict
from training_device import tensor_on,configure_device
from rishi_historical_objectives import pairwise,portfolio

ROOT=Path('results/HISTORICAL_OBJECTIVE_FITS_20261029')
YEARS=(2022,2023,2024,2025);SEEDS=(7,19,1828801906,1868046758);MODES=('mse','pairwise','portfolio')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)
def loss_value(score,target,mask,mode):
 if mode=='mse':
  return (score[mask]-target[mask]).square().mean()
 if mode=='pairwise':return pairwise(score,target,mask)
 if mode=='portfolio':return portfolio(score,target/100,mask,.002)
 raise ValueError(mode)

def main():
 ROOT.mkdir(parents=True,exist_ok=True);device=configure_device('cuda');lock=ROOT/'PROTOCOL_LOCK.json';inputs=[Path(__file__),Path('research/analysis/rishi_historical_objectives.py'),Path('docs/historical_sharpe_20261029/FINANCIAL_OBJECTIVE_PROTOCOL.md'),Path('data/nasdaq_monthly_pit/extended_hours_features_20261024/features.npz')];hashes={str(p):sha(p) for p in inputs}
 if lock.exists():
  if json.loads(lock.read_text())!=hashes:raise ValueError('training inputs changed')
 else:atomic_json(lock,hashes)
 manifest={};start=time.monotonic()
 for year in YEARS:
  d,x,extra,tr,va,te,groups=prepared_fold(year);dates=d['panel'].dates;assert dates[d['exit'][tr]].max()<dates[d['origin'][va]].min() and dates[d['exit'][va]].max()<dates[d['origin'][te]].min()
  for seed in SEEDS:
   for mode in MODES:
    folder=ROOT/str(year)/str(seed)/mode;folder.mkdir(parents=True,exist_ok=True);record=folder/'record.json'
    if record.exists():
     obj=json.loads(record.read_text());assert obj['status']=='complete'
     for name,h in obj['hashes'].items():assert sha(folder/name)==h
     manifest[f'{year}/{seed}/{mode}']=sha(record);continue
    if any(folder.iterdir()):raise FileExistsError(f'preserve incomplete fit: {folder}')
    torch.manual_seed(seed);model=ExtendedModel(len(d['panel'].tickers),groups,'after_hours_premarket').to(device);opt=optimizer_for(model,'adamw',.001);target=d['y']*100;best=float('inf');state=None;stale=0;history=[];chosen=0
    for epoch in range(1,31):
     model.train();total=0.;count=0
     for j in range(0,len(tr),8):
      sub=tr[j:j+8];opt.zero_grad(set_to_none=True);score,_=model(tensor_on(x[sub],device),tensor_on(extra[sub],device),tensor_on(d['mask'][sub].astype('float32'),device));mask=tensor_on(d['labelmask'][sub],device).bool();y=tensor_on(target[sub],device);loss=loss_value(score,y,mask,mode)
      if not torch.isfinite(loss):raise ValueError(f'nonfinite loss {folder} epoch {epoch}')
      loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
      if not torch.isfinite(norm):raise ValueError(f'nonfinite gradient {folder} epoch {epoch}')
      opt.step();total+=float(loss.detach())*len(sub);count+=len(sub)
     vp=predict(model,x,extra,d['mask'],va,device,8);val=float(((vp-target[va])**2)[d['labelmask'][va]].mean());assert np.isfinite(val);history.append(dict(epoch=epoch,training_loss=total/count,validation_mse=val))
     if val<best-1e-7:best=val;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};chosen=epoch;stale=0
     else:stale+=1
     if stale>=6:break
    assert state is not None;model.load_state_dict(state);vp,tp=[predict(model,x,extra,d['mask'],ix,device,8) for ix in (va,te)];assert np.isfinite(tp[d['mask'][te]]).all();torch.save(dict(state_dict=state,year=year,seed=seed,mode=mode,epoch=chosen,groups=[sorted(g) for g in groups]),folder/'checkpoint.pt');np.savez_compressed(folder/'predictions.npz',validation=vp,test=tp,validation_ids=va,test_ids=te);atomic_json(folder/'history.json',history);obj=dict(status='complete',year=year,seed=seed,mode=mode,chosen_epoch=chosen,validation_mse=best,epochs_run=len(history),hashes={name:sha(folder/name) for name in ('checkpoint.pt','predictions.npz','history.json')});atomic_json(record,obj);manifest[f'{year}/{seed}/{mode}']=sha(record);print(f'{year}/{seed}/{mode}: {len(history)} epochs; elapsed {time.monotonic()-start:.0f}s',flush=True)
 assert len(manifest)==48;atomic_json(ROOT/'COMPLETE.json',dict(fits=manifest,protocol_hashes=hashes));print('All 48 fits locked; no test metrics computed',flush=True)
if __name__=='__main__':main()
