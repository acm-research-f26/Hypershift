"""Independently verify style overlap with explicit security and issuer ties."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import run_ndx_drawdown_research as study


def target(score,mask,names,kind):
    mapping={'GOOG':'ALPHABET','GOOGL':'ALPHABET','FOX':'FOX','FOXA':'FOX','LBTYA':'LIBERTY_GLOBAL','LBTYK':'LIBERTY_GLOBAL'}
    buckets={}
    for j in np.flatnonzero(mask):
        key=mapping.get(names[j],names[j]) if kind=='issuer5' else names[j];buckets.setdefault(key,[]).append(j)
    best={name:max(score[j] for j in values) for name,values in buckets.items()};k=5 if kind=='issuer5' else 10
    threshold=sorted(best.values(),reverse=True)[k-1];higher=sum(v>threshold for v in best.values());tied=sum(v==threshold for v in best.values());w=np.zeros(len(score))
    for name,values in buckets.items():
        val=best[name];budget=(1/k if val>threshold else (k-higher)/(k*tied) if val==threshold else 0.)
        chosen=[j for j in values if score[j]==val];w[chosen]=budget/len(chosen)
    return w


def main():
    data=study.load_data();names=np.asarray(data['panel'].tickers);ids=np.concatenate([study.fold(data,y)[2] for y in (2022,2023,2024,2025)])
    sources={'issuer5':np.load('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029/TARGETS.npz',allow_pickle=False),'equal10':np.load('results/RISHI_VALIDATION_TOP10_20261029/TARGETS.npz',allow_pickle=False)}
    common=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch58_validation_style_partial/COMMON_STOCKS.csv');rows=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch60_validation_style_overlap/ORIGINS.csv').set_index(['group','year','origin','control','allocation'])
    assert rows.index.is_unique;maximum=0.;checked=0
    for group in ('original_five','new_five','ten_seed'):
        cursor=0
        for year in (2022,2023,2024,2025):
            _,_,test=study.fold(data,year);f=common[(common.group==group)&(common.year==year)].set_index('security_id');ix={n:j for j,n in enumerate(names)};support=np.isin(names,f.index);scores={}
            for label,col in [('beta','beta252'),('liquidity','log10adv20')]:
                a=np.full(len(names),np.nan);v=rankdata(f[col].to_numpy(),method='average')/len(f)
                for name,value in zip(f.index,v):a[ix[name]]=value
                scores[label]=a
            scores['equal_blend']=(scores['beta']+scores['liquidity'])/2
            for label,raw in scores.items():
                score=raw.copy();score[~support]=np.median(score[support])
                for kind,z in sources.items():
                    for j,i in enumerate(test):
                        w=target(score,data['mask'][i],names,kind);ref=z[f'{group}__fixed'][cursor+j];r=rows.loc[(group,year,int(i),label,kind)]
                        assert r.shared_names==np.count_nonzero((w>0)&(ref>0))
                        assert r.fallback_eligible==np.count_nonzero(data['mask'][i]&~support) and r.fallback_selected==np.count_nonzero((w>0)&~support)
                        maximum=max(maximum,abs(r.weight_overlap-np.minimum(w,ref).sum()));checked+=1
            cursor+=len(test)
    assert checked==3618 and maximum<1e-12
    print(f'{checked} target comparisons verified with explicit ties; max error {maximum:.3g}')


if __name__=='__main__':main()
