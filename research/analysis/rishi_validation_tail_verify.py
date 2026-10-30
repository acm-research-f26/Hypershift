"""Independent pandas reconstruction of fixed matched-ID validation tail sets."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study


def main():
    data=study.load_data(); names=list(data['panel'].tickers)
    rows=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch47_validation_tail/CELLS.csv')
    roots={'B':Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),
           'D':Path('runs/ndx_reference_networks_20261024/dynamic_restricted')}
    checked=0
    for year in (2022,2023,2024,2025):
        _,val,test=study.fold(data,year)
        for group,seeds in study.GROUPS.items():
            values={}
            for family,root in roots.items():
                splits={'validation':[],'test':[]}
                for seed in seeds:
                    with np.load(root/str(year)/f'{seed}_predictions.npz',allow_pickle=False) as z:
                        for label,ids in [('validation',val),('test',test)]:
                            f=pd.DataFrame(z[label],columns=names).where(data['mask'][ids])
                            splits[label].append(f.rank(axis=1,pct=True))
                values[family]={k:sum(v)/len(v) for k,v in splits.items()}
            values['E']={k:(values['B'][k]+values['D'][k])/2 for k in ('validation','test')}
            for family in ('B','D','E'):
                v,t=values[family]['validation'],values[family]['test']; common=(v.count()>=5)&(t.count()>=5)
                a=v.mean()[common];b=t.mean()[common]; median=b.median()
                first=sorted(a.index,key=lambda s:(-a[s],s)); second=sorted(b.index,key=lambda s:(-b[s],s))
                for k in (5,10):
                    row=rows[(rows.year==year)&(rows.group==group)&(rows.family==family)&(rows.k==k)].iloc[0]
                    overlap=len(set(first[:k])&set(second[:k])); assert row.matched==len(a)
                    assert row.validation_top=='|'.join(first[:k]) and row.test_top=='|'.join(second[:k])
                    assert row.intersection==overlap and abs(row.jaccard-overlap/(2*k-overlap))<1e-12
                    assert abs(row.validation_top_above_test_median-np.mean(b[first[:k]]>median))<1e-12
                    checked+=1
    assert checked==72;print('72 matched-ID upper-tail cells independently verified')


if __name__=='__main__':main()
