"""Frozen seed comparisons using immutable saved forecasts and account engines."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import run_ndx_drawdown_research as study
import ndx_drawdown_engine as engine
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq,simulate_signed
from audit_ndx_weak_market_short import value_replay
from rishi_historical_selection import targets
from rishi_historical_hedge import positions
from rishi_mechanism_risk import POLICIES

SEEDS=(7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941)
RECIPES=(('ret75','retention',.75),('ret50','retention',.5),('issuer75','issuer',.75),('equal0','equal',0.),('ret0','retention',0.),('issuer50','issuer',.5))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=Path('docs/historical_sharpe_20261029/seed_batch1');dest=Path('results/HISTORICAL_SEED_INVESTIGATION_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True);d=study.load_data();ids=np.concatenate([study.fold(d,y)[2] for y in (2022,2023,2024,2025)]);mask=d['mask'][ids];manifest=pd.read_csv('docs/ndx_artifacts_20261027/manifest.csv').set_index('path');families={};hashes={};market=engine.market_inputs(d);combined=append_qqq(d,etf_data(d,'QQQ'));dates=d['panel'].dates[int(d['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
 for kind,root in [('B',Path('runs/ndx_extended_hours_20261024/after_hours_premarket')),('D',Path('runs/ndx_reference_networks_20261024/dynamic_restricted'))]:
  for seed in SEEDS:
   rows=[]
   for year in (2022,2023,2024,2025):
    for name in (f'{seed}.pt',f'{seed}_predictions.npz'):
     p=root/str(year)/name;h=sha(p);assert h==manifest.loc[p.as_posix(),'sha256'];hashes[p.as_posix()]=h
    z=np.load(root/str(year)/f'{seed}_predictions.npz',allow_pickle=False);_,_,te=study.fold(d,year);assert z['test'].shape==(len(te),len(d['panel'].tickers));rows.append(z['test'])
   raw=np.concatenate(rows);assert np.isfinite(raw[mask]).all();families[kind,seed]=pd.DataFrame(raw.astype('float64')).where(mask).rank(axis=1,pct=True).to_numpy()
 groups={'original_five':SEEDS[:5],'new_five':SEEDS[5:],'ten_seed':SEEDS,**{str(s):(s,) for s in SEEDS}};scores={};weights={};paths={};metrics=[];cal=[];parity=[];checks=[];prior=np.load('results/HISTORICAL_SHARPE_HEDGE_20261029/PATHS.npz',allow_pickle=False);np.testing.assert_array_equal(prior['dates'],dates)
 for group,seeds in groups.items():
  score=.5*(sum(families['B',s] for s in seeds)/len(seeds))+.5*(sum(families['D',s] for s in seeds)/len(seeds));scores[group]=score
  if group in ('original_five','new_five','ten_seed'):
   old=np.load(f'results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_E_{group}.npz');np.testing.assert_array_equal(old['ids'],ids);np.testing.assert_allclose(score[mask],old['scores'][mask],rtol=0,atol=1e-12)
  for recipe,rule,cap in RECIPES:
   w=engine.allocations(d,ids,score,'issuer5') if rule=='issuer' else targets(score,mask,5,rule);risk=engine.risk_inputs(d,engine.daily_allocation(d,ids,w));long=engine.exposure_series(POLICIES['vol20'],market,risk)[d['origin'][ids]];desired=positions(w,long,market['qqqvol20'][d['origin'][ids]],cap);m,path,ledger=simulate_signed(combined,ids,desired,.03,1.,detailed=True);independent=value_replay(combined,ids,desired,.03,1.);delta=float(np.max(abs(path-independent)));assert delta<1e-12 and ledger.date.tolist()==dates.tolist() and ledger.nav.min()>0
   if group in ('original_five','new_five','ten_seed') and recipe in ('ret75','ret50','issuer75'):
    key=f'{group}__{"issuer" if rule=="issuer" else "retention"}__cap{int(cap*100):02d}__b03__c1';error=float(np.max(abs(path-prior[key])));assert error<1e-10;parity.append(dict(group=group,recipe=recipe,max_error=error))
   key=f'{group}__{recipe}';paths[key]=path;weights[key]=desired;ledger.to_csv(dest/(key+'_ledger.csv'),index=False);metrics.append(dict(id=key,group=group,recipe=recipe,**m));checks.append(dict(id=key,independent_daily_error=delta))
   for year in (2022,2023,2024,2025):
    ix=pd.DatetimeIndex(dates).year==year;cal.append(dict(id=key,year=year,calendar_return=float(np.prod(1+path[ix])-1)))
  print('Completed',group,flush=True)
 assert len(metrics)==78 and len(parity)==9
 pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(cal).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'PARITY.csv',index=False);pd.DataFrame(checks).to_csv(out/'ACCOUNT_CHECKS.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths);np.savez_compressed(dest/'TARGETS.npz',ids=ids,security_ids=np.asarray(d['panel'].tickers+['ETF:QQQ'],dtype=str),**weights);np.savez_compressed(dest/'SCORES.npz',ids=ids,**scores)
 for p in (Path(__file__),Path('docs/historical_sharpe_20261029/SEED_INVESTIGATION_PROTOCOL.md')):hashes[str(p)]=sha(p)
 (out/'HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');m=pd.DataFrame(metrics);print(m[~m.group.isin(['original_five','new_five','ten_seed'])].groupby('recipe').sharpe.agg(['min','median','max']).to_string())
if __name__=='__main__':main()
