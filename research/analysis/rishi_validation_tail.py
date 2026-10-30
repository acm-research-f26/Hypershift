"""Fixed retrospective validation/test matched-ID upper-tail comparison."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
from rishi_validation_stable_rank import ranks


def main():
    out = Path('docs/rishi_mechanism_20261027/execution_batch47_validation_tail')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    data = study.load_data(); names = np.asarray(data['panel'].tickers)
    paths = {'B':Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),
             'D':Path('runs/ndx_reference_networks_20261024/dynamic_restricted')}
    rows = []; hashes = {}
    for year in (2022,2023,2024,2025):
        _, val, test = study.fold(data,year)
        assert data['exit'][val].max() < data['origin'][test].min()
        for group,seeds in study.GROUPS.items():
            values = {}
            for family,root in paths.items():
                per = {'validation':[],'test':[]}
                for seed in seeds:
                    path = root / str(year) / f'{seed}_predictions.npz'
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as z:
                        for split,ids in [('validation',val),('test',test)]:
                            assert z[split].shape == data['mask'][ids].shape
                            per[split].append(ranks(z[split],data['mask'][ids]))
                values[family] = {k:np.mean(v,axis=0) for k,v in per.items()}
            values['E'] = {k:(values['B'][k]+values['D'][k])/2 for k in ('validation','test')}
            for family in ('B','D','E'):
                mean=[]; counts=[]
                for split in ('validation','test'):
                    x=values[family][split]; count=np.isfinite(x).sum(0); counts.append(count)
                    mean.append(np.divide(np.nansum(x,axis=0),count,out=np.full(len(names),np.nan),where=count>0))
                common=(counts[0]>=5)&(counts[1]>=5); ids=np.flatnonzero(common)
                assert len(ids)>=10
                ranked=[ids[np.lexsort((names[ids],-x[ids]))] for x in mean]
                median=np.median(mean[1][ids])
                for k in (5,10):
                    a,b=ranked[0][:k],ranked[1][:k]; overlap=len(set(a)&set(b))
                    rows.append(dict(year=year,group=group,family=family,k=k,matched=len(ids),intersection=overlap,
                        jaccard=overlap/(2*k-overlap),validation_top_above_test_median=float(np.mean(mean[1][a]>median)),
                        validation_top='|'.join(names[a]),test_top='|'.join(names[b])))
    assert len(rows)==72
    frame=pd.DataFrame(rows); frame.to_csv(out/'CELLS.csv',index=False)
    (out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(frame[(frame.group=='ten_seed')&(frame.family=='E')].to_string(index=False))
    print('cells',len(frame),'files hashed',len(hashes))


if __name__=='__main__':main()
