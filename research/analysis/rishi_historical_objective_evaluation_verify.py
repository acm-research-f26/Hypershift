"""Reconstruct fixed objective-evaluation targets and verify account paths."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from pit_research_engine import load_data,fold

GROUPS={'7':(7,),'19':(19,),'1828801906':(1828801906,),'1868046758':(1868046758,), 'original_two':(7,19),'new_two':(1828801906,1868046758),'all_four':(7,19,1828801906,1868046758)}
ISSUERS={'GOOG':'ALPHABET','GOOGL':'ALPHABET','FOX':'FOX','FOXA':'FOX','LBTYA':'LIBERTY_GLOBAL','LBTYK':'LIBERTY_GLOBAL'}
def ranks(scores,mask):
 result=np.full(mask.shape,np.nan)
 for i in range(len(mask)):
  active=np.flatnonzero(mask[i]);result[i,active]=np.mean([rankdata(x[i,active].astype('float64'),method='average')/len(active) for x in scores],axis=0)
 return result

def weights(score,mask,names,kind):
 eligible=np.flatnonzero(mask);out=np.zeros(len(mask))
 if kind=='equal10':
  k=min(10,len(eligible));cut=sorted(score[eligible],reverse=True)[k-1];above=eligible[score[eligible]>cut];ties=eligible[score[eligible]==cut];out[above]=1/k;out[ties]=(k-len(above))/(k*len(ties));return out
 groups={}
 for i in eligible:groups.setdefault(ISSUERS.get(names[i],names[i]),[]).append(i)
 values={name:max(score[j] for j in nodes) for name,nodes in groups.items()};k=min(5,len(values));cut=sorted(values.values(),reverse=True)[k-1];above=[name for name in groups if values[name]>cut];tied=[name for name in groups if values[name]==cut]
 for name in above+tied:
  amount=1/k if name in above else (k-len(above))/(k*len(tied));members=[j for j in groups[name] if score[j]==values[name]];out[members]=amount/len(members)
 return out

def main():
 root=Path('results/HISTORICAL_OBJECTIVE_ACCOUNTS_20261029');train=Path('results/HISTORICAL_OBJECTIVE_FITS_20261029');out=Path('docs/historical_sharpe_20261029/objective_accounts');p=np.load(root/'PATHS.npz');t=np.load(root/'TARGETS.npz');m=pd.read_csv(out/'METRICS.csv');cal=pd.read_csv(out/'CALENDAR.csv');d=load_data();ids=t['ids'];mask=d['mask'][ids];names=d['panel'].tickers;np.testing.assert_array_equal(t['security_ids'],names);years=(2022,2023,2024,2025);cache={};dyn={};err=0.;checked=0
 for seed in (7,19,1828801906,1868046758):
  dyn[seed]=np.concatenate([np.load(f'runs/ndx_reference_networks_20261024/dynamic_restricted/{y}/{seed}_predictions.npz')['test'] for y in years])
  for mode in ('mse','pairwise','portfolio'):cache[mode,seed]=np.concatenate([np.load(train/str(y)/str(seed)/mode/'predictions.npz')['test'] for y in years])
 for mode in ('mse','pairwise','portfolio'):
  for group,seeds in GROUPS.items():
   parent=ranks([cache[mode,s] for s in seeds],mask);peer=ranks([dyn[s] for s in seeds],mask)
   for family,score in (('parent',parent),('blend',(parent+peer)/2)):
    for kind in ('issuer5','equal10'):
     key=f'{mode}__{group}__{family}__{kind}';actual=t[key]
     for i in range(len(ids)):
      expected=weights(score[i],mask[i],names,kind);err=max(err,float(np.max(abs(expected-actual[i]))));checked+=1
 for row in m.itertuples():
  ret=p[row.id];ledger=pd.read_csv(root/(row.id+'_ledger.csv'));wealth=np.r_[1.,np.cumprod(1+ret)];sh=(ret.mean()-(1.03**(1/252)-1))*np.sqrt(252)/ret.std(ddof=1);c=wealth[-1]**(252/len(ret))-1;dd=(wealth/np.maximum.accumulate(wealth)-1).min();err=max(err,abs(sh-row.sharpe),abs(c-row.annualized_return),abs(dd-row.maximum_drawdown),float(np.max(abs(ledger.nav/1e6-wealth[1:]))))
  for y in years:
   ix=pd.DatetimeIndex(p['dates']).year==y;err=max(err,abs(np.prod(1+ret[ix])-1-cal[(cal.id==row.id)&(cal.year==y)].calendar_return.iloc[0]))
 print(f'diagnostic targets={checked}, metrics={len(m)}, calendar={len(cal)}, max={err:.12g}',flush=True)
 assert checked==84*201 and len(m)==84 and len(cal)==336 and err<1e-10
 print(f'{checked} target rows, 84 account metrics/NAV and 336 calendar returns verified; max error {err:.3g}')
if __name__=='__main__':main()
