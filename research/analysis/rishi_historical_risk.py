"""Frozen risk-control grid on existing selection targets."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from pit_research_engine import load_data
import ndx_drawdown_engine as engine

GROUPS=('original_five','new_five','ten_seed')
def smooth(x):
 x=np.asarray(x,float)
 if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('invalid exposure')
 y=x.copy()
 for i in range(1,len(y)):y[i]=.25*x[i]+.75*y[i-1]
 return y

def main():
 out=Path('docs/historical_sharpe_20261029/risk_batch1');dest=Path('results/HISTORICAL_SHARPE_RISK_20261029');source=Path('results/HISTORICAL_SHARPE_SELECTION_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True)
 d=load_data();mkt=engine.market_inputs(d);t=np.load(source/'TARGETS.npz');ref=np.load(source/'PATHS.npz');ids=t['ids'];dates=ref['dates'];np.testing.assert_array_equal(t['security_ids'],d['panel'].tickers)
 books={};parity=[];metrics=[];annual=[];paths={};exposures={}
 for group in GROUPS:
  for rule in ('equal','retention'):
   key=f'{group}__k5__{rule}__vol20';w=t[key];risk=engine.risk_inputs(d,engine.daily_allocation(d,ids,w));spec={'vol':'basket','window':20,'target':.20};exp=engine.exposure_series(spec,mkt,risk);_,path,_=engine.simulate(d,ids,w,exp,'weekly',detailed=True);error=float(np.max(abs(path-ref[key])));assert error<1e-10;parity.append(dict(group=group,rule=rule,max_error=error));books[group,rule]=(w,risk)
 for (group,rule),(w,risk) in books.items():
  for window in (20,60):
   for target in (.15,.20,.25,.30):
    for smoothing in (False,True):
     spec={'vol':'basket','window':window,'target':target};raw=engine.exposure_series(spec,mkt,risk);exp=smooth(raw) if smoothing else raw;m,path,ledger=engine.simulate(d,ids,w,exp,'weekly',detailed=True);assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
     key=f'{group}__{rule}__w{window}__t{int(target*100)}__s{int(smoothing)}';paths[key]=path;exposures[key]=exp;ledger.to_csv(dest/(key+'_ledger.csv'),index=False);metrics.append(dict(id=key,group=group,rule=rule,window=window,target=target,smoothing=smoothing,**m,mean_exposure=float(ledger.equity_exposure.mean())))
     for y in (2022,2023,2024,2025):
      ix=pd.DatetimeIndex(dates).year==y;annual.append(dict(id=key,year=y,calendar_return=float(np.prod(1+path[ix])-1)))
  print(group,rule,flush=True)
 assert len(metrics)==96 and len(parity)==6
 pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(annual).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'PARITY.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'EXPOSURES.npz',**exposures)
 files=[source/'TARGETS.npz',source/'PATHS.npz',Path(__file__),Path('docs/historical_sharpe_20261029/RISK_PROTOCOL.md')];(out/'HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
 print(pd.DataFrame(metrics).sort_values('sharpe',ascending=False).head(12)[['id','sharpe','annualized_return','maximum_drawdown','turnover_annual']].to_string(index=False))
if __name__=='__main__':main()
