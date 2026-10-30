"""Compare fixed validation-rank issuer selections to existing test targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
from ndx_drawdown_engine import allocations
from rishi_validation_stable_rank import ranks


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch48_validation_fixed')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    data=study.load_data(); saved=np.load('results/RISHI_PARENT_OVERLAY_20261028/TARGETS.npz',allow_pickle=False)
    ids=np.concatenate([study.fold(data,y)[2] for y in (2022,2023,2024,2025)])
    np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['security_ids'],data['panel'].tickers)
    roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')]
    rows=[];hashes={}
    for group,seeds in study.GROUPS.items():
        cursor=0; existing=saved[f'after_hours_premarket__{group}__issuer5']
        for year in (2022,2023,2024,2025):
            _,val,test=study.fold(data,year);assert data['exit'][val].max()<data['origin'][test].min()
            families=[]
            for root in roots:
                rank=[]
                for seed in seeds:
                    path=root/str(year)/f'{seed}_predictions.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as z:
                        assert z['validation'].shape==data['mask'][val].shape
                        rank.append(ranks(z['validation'],data['mask'][val]))
                families.append(np.mean(rank,axis=0))
            blend=(families[0]+families[1])/2;count=np.isfinite(blend).sum(0)
            mean=np.divide(np.nansum(blend,axis=0),count,out=np.full(len(count),np.nan),where=count>0)
            qualified=(count>=5)&np.isfinite(mean);assert qualified.any()
            mean[~qualified]=np.median(mean[qualified]);scores=np.broadcast_to(mean,(len(test),len(mean)))
            fixed=allocations(data,test,scores,'issuer5');actual=existing[cursor:cursor+len(test)]
            for j,i in enumerate(test):
                a,b=fixed[j],actual[j]
                rows.append(dict(group=group,year=year,fill_date=str(data['panel'].dates[data['entry'][i]].date()),
                    shared_names=int(np.sum((a>0)&(b>0))),target_weight_overlap=float(np.minimum(a,b).sum()),
                    fallback_eligible=int(np.sum(data['mask'][i]&~qualified)),fallback_selected=int(np.sum((a>0)&~qualified))))
            cursor+=len(test)
    frame=pd.DataFrame(rows);assert len(frame)==603
    frame.to_csv(out/'ORIGINS.csv',index=False);(out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(frame.groupby(['group','year'])[['shared_names','target_weight_overlap','fallback_eligible','fallback_selected']].mean().to_string())
    print('rows',len(frame),'input files',len(hashes))


if __name__=='__main__':main()
