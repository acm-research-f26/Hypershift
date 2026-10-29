"""Evaluate fixed trained objective outputs only after every fit is locked."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

def average_ranks(scores,mask):
 scores=np.asarray(scores);mask=np.asarray(mask,bool)
 if scores.ndim!=3 or scores.shape[1:]!=mask.shape or not np.isfinite(scores[:,mask]).all():raise ValueError('invalid scores')
 ranks=[pd.DataFrame(a).where(mask).rank(axis=1,pct=True).to_numpy() for a in scores]
 return np.mean(ranks,axis=0)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from pit_research_engine import load_data,fold
 from rishi_mechanism_risk import POLICIES
 import ndx_drawdown_engine as engine
 root=Path('results/HISTORICAL_OBJECTIVE_FITS_20261029');complete=json.loads((root/'COMPLETE.json').read_text());assert len(complete['fits'])==48
 for p,h in complete['protocol_hashes'].items():assert sha(p)==h
 for key,h in complete['fits'].items():
  folder=root/key;record=folder/'record.json';assert sha(record)==h;rec=json.loads(record.read_text());assert rec['status']=='complete'
  for name,digest in rec['hashes'].items():assert sha(folder/name)==digest
 out=Path('docs/historical_sharpe_20261029/objective_accounts');dest=Path('results/HISTORICAL_OBJECTIVE_ACCOUNTS_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True);d=load_data();market=engine.market_inputs(d);years=(2022,2023,2024,2025);ids=np.concatenate([fold(d,y)[2] for y in years]);mask=d['mask'][ids];dates=d['panel'].dates[int(d['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str);manifest=pd.read_csv('docs/ndx_artifacts_20261027/manifest.csv').set_index('path');dynamic=Path('runs/ndx_reference_networks_20261024/dynamic_restricted');seeds=(7,19,1828801906,1868046758);groups={**{str(s):(s,) for s in seeds},'original_two':seeds[:2],'new_two':seeds[2:],'all_four':seeds};raw={};dyn={}
 for seed in seeds:
  dy=[]
  for year in years:
   _,_,test=fold(d,year);folder=dynamic/str(year)
   for name in (f'{seed}_predictions.npz',f'{seed}.pt'):
    file=folder/name;assert sha(file)==manifest.loc[file.as_posix(),'sha256']
   z=np.load(folder/f'{seed}_predictions.npz',allow_pickle=False);assert z['test'].shape==(len(test),len(d['panel'].tickers));dy.append(z['test'])
  dyn[seed]=np.concatenate(dy)
  for mode in ('mse','pairwise','portfolio'):
   pieces=[]
   for year in years:
    _,_,test=fold(d,year);z=np.load(root/str(year)/str(seed)/mode/'predictions.npz',allow_pickle=False);np.testing.assert_array_equal(z['test_ids'],test);pieces.append(z['test'])
   raw[mode,seed]=np.concatenate(pieces)
 metrics=[];cal=[];paths={};targets={}
 for mode in ('mse','pairwise','portfolio'):
  for group,selected in groups.items():
   parent=average_ranks(np.stack([raw[mode,s] for s in selected]),mask);peer=average_ranks(np.stack([dyn[s] for s in selected]),mask)
   for family,score in (('parent',parent),('blend',(parent+peer)/2)):
    for kind in ('issuer5','equal10'):
     w=engine.allocations(d,ids,score,kind);risk=engine.risk_inputs(d,engine.daily_allocation(d,ids,w));exp=engine.exposure_series(POLICIES['vol20' if kind=='issuer5' else 'full'],market,risk);m,path,ledger=engine.simulate(d,ids,w,exp,'weekly',detailed=True);assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
     key=f'{mode}__{group}__{family}__{kind}';paths[key]=path;targets[key]=w;metrics.append(dict(id=key,mode=mode,group=group,family=family,kind=kind,**m,mean_exposure=float(ledger.equity_exposure.mean())));ledger.to_csv(dest/(key+'_ledger.csv'),index=False)
     for year in years:
      ix=pd.DatetimeIndex(dates).year==year;cal.append(dict(id=key,year=year,calendar_return=float(np.prod(1+path[ix])-1)))
  print(mode,flush=True)
 assert len(metrics)==84;pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(cal).to_csv(out/'CALENDAR.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'TARGETS.npz',ids=ids,security_ids=np.asarray(d['panel'].tickers,dtype=str),**targets);files=[Path(__file__),Path('docs/historical_sharpe_20261029/FINANCIAL_EVALUATION_PROTOCOL.md'),root/'COMPLETE.json'];(out/'HASHES.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2));print(pd.DataFrame(metrics).sort_values('sharpe',ascending=False).head(12)[['id','sharpe','annualized_return','maximum_drawdown']].to_string(index=False))
if __name__=='__main__':main()
