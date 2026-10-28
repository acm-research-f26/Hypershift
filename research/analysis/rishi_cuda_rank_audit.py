"""Independent reconstruction of local CUDA diagnostic counts from raw arrays."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ndx_reference_training import prepare
from rishi_raw_replay import SEEDS,GROUPS,PARENT,REFERENCE
from research_engine import target_weights
from run_robust_research import ensemble_predictions


def stats(a,b,mask):
    ranks=top5=top10=0
    for x,y,m in zip(a,b,mask):
        ranks+=int(not np.array_equal(rankdata(x[m]),rankdata(y[m])))
        top5+=int(not np.array_equal(target_weights(x,m,5,'equal'),target_weights(y,m,5,'equal')))
        top10+=int(not np.array_equal(target_weights(x,m,10,'equal'),target_weights(y,m,10,'equal')))
    return ranks,top5,top10


def main():
    doc=Path('docs/rishi_mechanism_20261027/execution_batch18_cuda_rank');large=Path('results/RISHI_CUDA_RANK_DIAGNOSTIC_20261028')
    check=pd.read_csv(doc/'CHECKPOINT_SPLITS.csv');ensemble=pd.read_csv(doc/'ENSEMBLES.csv');receipt=json.loads((doc/'RESULTS.json').read_text())
    if receipt['failures'] or len(check)!=160 or len(ensemble)!=36 or not (ensemble.status=='complete').all():raise ValueError('Incomplete diagnostic')
    raw_hashes={};seen=0;seen_ensemble=0
    for year in (2022,2023,2024,2025):
        p=prepare(year);mask=p['data']['mask'];scores={}
        for family in ('B','D'):
            folder=(PARENT if family=='B' else REFERENCE/'dynamic_restricted')/str(year)
            for seed in SEEDS:
                path=large/f'{family}_{year}_{seed}.npz';raw_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path,allow_pickle=False) as local,np.load(folder/f'{seed}_predictions.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(local['security_ids'],p['data']['panel'].tickers)
                    for split in ('validation','test'):
                        ids=p[split];np.testing.assert_array_equal(local[split+'_ids'],ids);a=local[split];b=saved[split]
                        row=check[(check.year==year)&(check.family==family)&(check.seed==seed)&(check['split']==split)]
                        if len(row)!=1:raise ValueError('Missing/duplicate checkpoint split')
                        row=row.iloc[0];np.testing.assert_allclose(np.max(abs(a-b)),row.max_value_error,atol=1e-15)
                        if stats(a,b,mask[ids])!=tuple(int(row[x]) for x in ['rank_changed_origins','top5_changed_origins','top10_changed_origins']):raise ValueError('Checkpoint rank mismatch')
                        seen+=1
                        if split=='test':scores[family,seed]=(a.copy(),b.copy())
        m=mask[p['test']]
        for group,seeds in GROUPS.items():
            pair={family:[ensemble_predictions([scores[family,s][i] for s in seeds],m) for i in (0,1)] for family in ('B','D')}
            for family in ('B','D','E'):
                a,b=pair[family] if family!='E' else [0.5*(pair['B'][i]+pair['D'][i]) for i in (0,1)]
                row=ensemble[(ensemble.year==year)&(ensemble.family==family)&(ensemble['group']==group)]
                if len(row)!=1:raise ValueError('Missing ensemble')
                row=row.iloc[0]
                if stats(a,b,m)!=tuple(int(row[x]) for x in ['rank_changed_origins','top5_changed_origins','top10_changed_origins']):raise ValueError('Ensemble rank mismatch')
                seen_ensemble+=1
    (doc/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(checkpoint_splits=seen,ensembles=seen_ensemble,raw_sha256=raw_hashes),indent=2)+'\n')
    print('Recomputed',seen,'checkpoint splits and',seen_ensemble,'ensemble cells')


if __name__=='__main__':main()
