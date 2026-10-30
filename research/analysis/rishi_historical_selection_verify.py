"""Independent explicit target and path checks for the fixed selection grid."""
from pathlib import Path
import numpy as np,pandas as pd
from pit_research_engine import load_data

def main():
 root=Path('results/HISTORICAL_SHARPE_SELECTION_20261029');out=Path('docs/historical_sharpe_20261029/selection_batch1');p=np.load(root/'PATHS.npz');t=np.load(root/'TARGETS.npz');m=pd.read_csv(out/'METRICS.csv');a=pd.read_csv(out/'CALENDAR.csv');d=load_data();ids=t['ids'];mask=d['mask'][ids];np.testing.assert_array_equal(t['security_ids'],d['panel'].tickers);maximum=0.;checked=0
 for group in ('original_five','new_five','ten_seed'):
  s=np.load(f'results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_E_{group}.npz')['scores']
  for k in (3,5,10,15):
   for mode in ('equal','confidence','retention'):
    old=np.zeros(s.shape[1]); key=f'{group}__k{k}__{mode}__full';actual=t[key]
    for i in range(len(ids)):
     active=np.flatnonzero(mask[i]);v=s[i].copy()
     if mode=='retention':v[active]+=.02*(old[active]>0)
     cutoff=sorted(v[active],reverse=True)[min(k,len(active))-1];above=active[v[active]>cutoff];tie=active[v[active]==cutoff];w=np.zeros(s.shape[1]);w[above]=1.;w[tie]=(min(k,len(active))-len(above))/len(tie)
     if mode=='confidence':w[active]*=np.exp(np.clip((v[active]-v[active].mean())/max(v[active].std(),1e-8),-2,2))
     w/=w.sum();maximum=max(maximum,float(np.max(abs(w-actual[i]))));old=w;checked+=1
    np.testing.assert_array_equal(actual,t[f'{group}__k{k}__{mode}__vol20'])
 for row in m.itertuples():
  r=p[row.id];ledger=pd.read_csv(root/(row.id+'_ledger.csv'));wealth=np.r_[1.,np.cumprod(1+r)];sh=(r.mean()-(1.03**(1/252)-1))*np.sqrt(252)/r.std(ddof=1);c=wealth[-1]**(252/len(r))-1;dd=(wealth/np.maximum.accumulate(wealth)-1).min();maximum=max(maximum,abs(sh-row.sharpe),abs(c-row.annualized_return),abs(dd-row.maximum_drawdown),float(np.max(abs(ledger.nav/1e6-wealth[1:]))))
  for y in (2022,2023,2024,2025):
   ix=pd.DatetimeIndex(p['dates']).year==y;maximum=max(maximum,abs(np.prod(1+r[ix])-1-a[(a.id==row.id)&(a.year==y)].calendar_return.iloc[0]))
 assert checked==7236 and len(m)==72 and len(a)==288 and maximum<1e-10
 print(f'{checked} targets, 72 paths/metrics, 288 annual returns verified, max error {maximum:.3g}')
if __name__=='__main__':main()
