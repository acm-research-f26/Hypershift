"""Independent explicit chronological risk and account checks."""
from pathlib import Path
import numpy as np,pandas as pd
from pit_research_engine import load_data

def main():
 root=Path('results/HISTORICAL_SHARPE_RISK_20261029');sel=Path('results/HISTORICAL_SHARPE_SELECTION_20261029');out=Path('docs/historical_sharpe_20261029/risk_batch1');p=np.load(root/'PATHS.npz');e=np.load(root/'EXPOSURES.npz');t=np.load(sel/'TARGETS.npz');m=pd.read_csv(out/'METRICS.csv');a=pd.read_csv(out/'CALENDAR.csv');d=load_data();ids=t['ids'];orig=d['origin'][ids];n=len(d['panel'].dates);err=0.;cache={}
 for group in ('original_five','new_five','ten_seed'):
  for rule in ('equal','retention'):
   w=t[f'{group}__k5__{rule}__vol20']
   for window in (20,60):
    risk=np.full(n,.30)
    for day in range(window-1,n):
     j=np.searchsorted(orig,day,side='right')-1
     if j<0:continue
     active=w[j]>0;history=np.nan_to_num(d['returns'][day-window+1:day+1,active],nan=0.)@w[j,active];risk[day]=max(float(history.std(ddof=1)*np.sqrt(252)),.05)
    cache[group,rule,window]=risk
 for row in m.itertuples():
  raw=np.minimum(1.,row.target/cache[row.group,row.rule,row.window]);expected=raw.copy()
  if row.smoothing:
   for j in range(1,n):expected[j]=.25*raw[j]+.75*expected[j-1]
  err=max(err,float(np.max(abs(expected-e[row.id]))));r=p[row.id];ledger=pd.read_csv(root/(row.id+'_ledger.csv'));wealth=np.r_[1.,np.cumprod(1+r)];sr=(r.mean()-(1.03**(1/252)-1))*np.sqrt(252)/r.std(ddof=1);c=wealth[-1]**(252/len(r))-1;dd=(wealth/np.maximum.accumulate(wealth)-1).min();err=max(err,abs(sr-row.sharpe),abs(c-row.annualized_return),abs(dd-row.maximum_drawdown),float(np.max(abs(ledger.nav/1e6-wealth[1:]))))
  for y in (2022,2023,2024,2025):
   ix=pd.DatetimeIndex(p['dates']).year==y;err=max(err,abs(np.prod(1+r[ix])-1-a[(a.id==row.id)&(a.year==y)].calendar_return.iloc[0]))
 assert len(m)==96 and len(a)==384 and err<1e-10
 print(f'96 exposures, paths and metrics, 384 calendar returns verified; maximum error {err:.3g}')
if __name__=='__main__':main()
