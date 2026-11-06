"""Fixed QQQ hedge grid on existing issuer and retention targets."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import run_ndx_drawdown_research as study
from ndx_drawdown_engine import daily_allocation,risk_inputs,exposure_series,market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq,simulate_signed
from audit_ndx_weak_market_short import value_replay
from rishi_mechanism_risk import POLICIES

GROUPS=('original_five','new_five','ten_seed')
def positions(w,long,vol,cap):
 w=np.asarray(w);long=np.asarray(long);vol=np.asarray(vol)
 if not 0<=cap<=1 or w.ndim!=2 or long.shape!=(len(w),) or vol.shape!=long.shape or not np.isfinite(w).all() or not np.isfinite(long).all() or not np.isfinite(vol).all():raise ValueError('bad hedge inputs')
 short=np.minimum(np.minimum(cap,1-long),.2/np.maximum(vol,.05));return np.column_stack((w*long[:,None],-short))

def main():
 out=Path('docs/historical_sharpe_20261029/hedge_batch1');dest=Path('results/HISTORICAL_SHARPE_HEDGE_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True)
 d=study.load_data();ids=np.concatenate([study.fold(d,y)[2] for y in (2022,2023,2024,2025)]);market=market_inputs(d);combined=append_qqq(d,etf_data(d,'QQQ'));issuer=np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz');ret=np.load('results/HISTORICAL_SHARPE_SELECTION_20261029/TARGETS.npz');np.testing.assert_array_equal(ids,issuer['ids']);np.testing.assert_array_equal(ids,ret['ids']);np.testing.assert_array_equal(d['panel'].tickers,issuer['security_ids']);np.testing.assert_array_equal(d['panel'].tickers,ret['security_ids']);source=pd.read_csv('docs/ndx_model_hedge_20261027/metrics.csv').set_index('id');dates=d['panel'].dates[int(d['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str);books={};parity=[];metrics=[];annual=[];paths={};targets={};maxerror=0.
 for group in GROUPS:
  for rule in ('issuer','retention'):
   w=issuer[f'after_hours_premarket__{group}__issuer5'] if rule=='issuer' else ret[f'{group}__k5__retention__vol20'];risk=risk_inputs(d,daily_allocation(d,ids,w));long=exposure_series(POLICIES['vol20'],market,risk)[d['origin'][ids]];vol=market['qqqvol20'][d['origin'][ids]];books[group,rule]=(w,long,vol)
  for cap,recipe in ((0.,'baseline'),(.5,'always/hedge50')):
   desired=positions(*books[group,'issuer'],cap);m,path,_=simulate_signed(combined,ids,desired,.03,1.);independent=value_replay(combined,ids,desired,.03,1.);delta=float(np.max(abs(path-independent)));assert delta<1e-12
   record=source.loc[f'{recipe}/{group}/c1/b0.03'];error=max(abs(m[k]-record[k]) for k in ('sharpe','annualized_return','maximum_drawdown'));assert error<1e-10;parity.append(dict(group=group,recipe=recipe,metric_error=error,independent_daily_error=delta))
 for (group,rule),book in books.items():
  for cap in (0.,.25,.5,.75):
   desired=positions(*book,cap);assert np.max(abs(desired).sum(1))<=1+1e-12
   for borrow in (.03,.06):
    for cost in (1,2):
     m,path,ledger=simulate_signed(combined,ids,desired,borrow,cost,detailed=True);other=value_replay(combined,ids,desired,borrow,cost);delta=float(np.max(abs(path-other)));assert delta<1e-12;maxerror=max(maxerror,delta)
     assert ledger.date.tolist()==dates.tolist() and ledger.nav.min()>0;key=f'{group}__{rule}__cap{int(cap*100):02d}__b{int(borrow*100):02d}__c{cost}';metrics.append(dict(id=key,group=group,rule=rule,cap=cap,borrow=borrow,cost=cost,**m));paths[key]=path;targets[key]=desired;ledger.to_csv(dest/(key+'_ledger.csv'),index=False)
     for year in (2022,2023,2024,2025):
      ix=pd.DatetimeIndex(dates).year==year;annual.append(dict(id=key,year=year,calendar_return=float(np.prod(1+path[ix])-1)))
  print(group,rule,flush=True)
 assert len(metrics)==96 and len(parity)==6
 pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(annual).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'PARITY.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'TARGETS.npz',ids=ids,**targets)
 files=[Path(__file__),Path('docs/historical_sharpe_20261029/HEDGE_PROTOCOL.md'),Path('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz'),Path('results/HISTORICAL_SHARPE_SELECTION_20261029/TARGETS.npz')];(out/'HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8');print('max independent day error',maxerror);print(pd.DataFrame(metrics).sort_values('sharpe',ascending=False).head(12)[['id','sharpe','annualized_return','maximum_drawdown','turnover_annual']].to_string(index=False))
if __name__=='__main__':main()
