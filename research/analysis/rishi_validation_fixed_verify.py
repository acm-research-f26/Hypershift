"""Rebuild fixed validation issuer targets with pandas and explicit boundary ties."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study


def main():
    d=study.load_data();names=np.asarray(d['panel'].tickers);saved=np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz',allow_pickle=False)
    rows=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch48_validation_fixed/ORIGINS.csv').set_index(['group','fill_date'])
    roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')]
    mapping={'GOOG':'ALPHABET','GOOGL':'ALPHABET','FOX':'FOX','FOXA':'FOX','LBTYA':'LIBERTY_GLOBAL','LBTYK':'LIBERTY_GLOBAL'}
    checked=0;maximum=0.
    for group,seeds in study.GROUPS.items():
        offset=0
        for year in (2022,2023,2024,2025):
            _,val,test=study.fold(d,year);families=[]
            for root in roots:
                per=[]
                for seed in seeds:
                    with np.load(root/str(year)/f'{seed}_predictions.npz',allow_pickle=False) as z:
                        p=pd.DataFrame(z['validation']).where(d['mask'][val]);per.append(p.rank(axis=1,pct=True))
                families.append(sum(per)/len(per))
            blend=(families[0]+families[1])/2; count=blend.count();mean=blend.mean();valid=count>=5
            score=mean.where(valid,mean[valid].median()).to_numpy()
            for j,i in enumerate(test):
                buckets={}
                for n in np.flatnonzero(d['mask'][i]): buckets.setdefault(mapping.get(names[n],names[n]),[]).append(n)
                best={name:max(score[n] for n in ns) for name,ns in buckets.items()}
                threshold=sorted(best.values(),reverse=True)[4]; high=sum(v>threshold for v in best.values());tied=sum(v==threshold for v in best.values())
                w=np.zeros(len(names))
                for name,ns in buckets.items():
                    value=best[name];budget=(.2 if value>threshold else (5-high)/(5*tied) if value==threshold else 0.)
                    tops=[n for n in ns if score[n]==value];w[tops]=budget/len(tops)
                a=saved[f'after_hours_premarket__{group}__issuer5'][offset+j]
                date=str(d['panel'].dates[d['entry'][i]].date());row=rows.loc[(group,date)]
                assert row.shared_names==np.count_nonzero((w>0)&(a>0))
                assert row.fallback_eligible==np.count_nonzero(d['mask'][i]&~valid.to_numpy())
                assert row.fallback_selected==np.count_nonzero((w>0)&~valid.to_numpy())
                maximum=max(maximum,abs(row.target_weight_overlap-np.minimum(w,a).sum()));checked+=1
            offset+=len(test)
    assert checked==603 and maximum<1e-12
    print(f'{checked} fixed-preference targets independently checked; max overlap error {maximum:.3g}')


if __name__=='__main__':main()
