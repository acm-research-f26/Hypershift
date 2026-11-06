"""Frozen historical top-k, confidence and retention search."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import target_weights
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import POLICIES

GROUPS=('original_five','new_five','ten_seed')
SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')

def targets(scores,mask,k,mode):
 scores=np.asarray(scores);mask=np.asarray(mask,bool);result=np.zeros_like(scores,float)
 if scores.shape!=mask.shape or mode not in ('equal','confidence','retention') or not np.isfinite(scores[mask]).all():raise ValueError('invalid selection inputs')
 for i in range(len(scores)):
  score=scores[i].copy()
  if mode=='retention' and i:score=np.where(mask[i],score+0.02*(result[i-1]>0),score)
  result[i]=target_weights(score,mask[i],k,'confidence' if mode=='confidence' else 'equal')
 return result

def main():
 out=Path('docs/historical_sharpe_20261029/selection_batch1');dest=Path('results/HISTORICAL_SHARPE_SELECTION_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True)
 data=load_data();market=engine.market_inputs(data);ref=np.load('runs/ndx_drawdown_20261026/batch1/paths.npz',allow_pickle=False);books={};parity=[];metrics=[];years=[];paths={};alltargets={}
 for group in GROUPS:
  z=np.load(SOURCE/f'scores_E_{group}.npz',allow_pickle=False);ids=z['ids'];score=z['scores'];mask=data['mask'][ids];dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str);np.testing.assert_array_equal(dates,ref['dates']);books[group]=(ids,score,mask)
  w=targets(score,mask,5,'equal');risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,w))
  for policy in ('full','vol20'):
   exposure=engine.exposure_series(POLICIES[policy],market,risk);_,path,_=engine.simulate(data,ids,w,exposure,'weekly',multiplier=1,detailed=True);key=f'equal5/{"full" if policy=="full" else "basketvol20_target20_weekly"}/{group}';loc=np.flatnonzero(ref['ids']==key);assert len(loc)==1
   err=float(np.max(abs(path-ref['returns'][:,loc[0]])));assert err<1e-10;parity.append(dict(group=group,policy=policy,max_error=err))
 for group,(ids,score,mask) in books.items():
  for k in (3,5,10,15):
   for mode in ('equal','confidence','retention'):
    w=targets(score,mask,k,mode);assert np.allclose(w.sum(1),1) and np.all(w[~mask]==0);risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,w))
    for policy in ('full','vol20'):
     exp=engine.exposure_series(POLICIES[policy],market,risk);m,path,ledger=engine.simulate(data,ids,w,exp,'weekly',multiplier=1,detailed=True);assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
     key=f'{group}__k{k}__{mode}__{policy}';paths[key]=path;alltargets[key]=w;ledger.to_csv(dest/(key+'_ledger.csv'),index=False);metrics.append(dict(id=key,group=group,k=k,mode=mode,policy=policy,**m,mean_exposure=float(ledger.equity_exposure.mean())))
     for y in (2022,2023,2024,2025):
      ix=pd.DatetimeIndex(dates).year==y;years.append(dict(id=key,year=y,calendar_return=float(np.prod(1+path[ix])-1)))
  print(group,flush=True)
 assert len(metrics)==72 and len(parity)==6
 pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(years).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'PARITY.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'TARGETS.npz',ids=ids,security_ids=np.asarray(data['panel'].tickers,dtype=str),**alltargets)
 inputs=[Path(__file__),Path('docs/historical_sharpe_20261029/SELECTION_PROTOCOL.md')]+[SOURCE/f'scores_E_{g}.npz' for g in GROUPS];(out/'HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
 print(pd.DataFrame(metrics).sort_values('sharpe',ascending=False).head(12)[['id','sharpe','annualized_return','maximum_drawdown','turnover_annual']].to_string(index=False))

if __name__=='__main__':main()
