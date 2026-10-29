"""Complete inference with explicit deterministic no-op contrasts.

The frozen evaluation produced all495accounts before the inherited bootstrap
rejected zero-variance differences. Preserve that protocol/source and every path.
"""
from datetime import datetime,timezone
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import run_ndx_model_hedge as study
from ndx_matched_controls import paired_inference
from research_engine import performance


def inference_with_noops(paths,contrasts,block,replicates=10000):
    """Exactly equal paths have deterministic zero differences; no epsilon fudge."""
    aliases={};distinct={};lookup={}
    for a,b in contrasts:
        for name in [a,b]:
            if name in aliases:continue
            p=np.asarray(paths[name],float)
            key=hashlib.sha256(p.tobytes()).hexdigest()
            if key not in lookup:
                lookup[key]=name;distinct[name]=p
            aliases[name]=lookup[key]
    active=[j for j,(a,b) in enumerate(contrasts) if aliases[a]!=aliases[b]]
    names=list(distinct);matrix=np.column_stack(list(distinct.values()))
    mapped=[(aliases[contrasts[j][0]],aliases[contrasts[j][1]]) for j in active]
    computed=paired_inference(matrix,names,mapped,block,replicates) if active else pd.DataFrame()
    result={}
    for j,row in zip(active,computed.to_dict('records')):
        row['strategy'],row['comparator']=contrasts[j]
        row['deterministic_noop']=False;result[j]=row
    for j,(a,b) in enumerate(contrasts):
        if j in result:continue
        result[j]=dict(strategy=a,comparator=b,block_sessions=block,replicates=replicates,
                       annual_active_mean=0.,simultaneous_active_low=0.,simultaneous_active_high=0.,
                       paired_sr_low=0.,paired_sr_high=0.,paired_cagr_low=0.,paired_cagr_high=0.,
                       deterministic_noop=True)
    return pd.DataFrame([result[j] for j in range(len(contrasts))])


def finish():
    study.verify()
    if (study.RUN/'COMPLETE.json').exists():raise ValueError('Already complete')
    metrics=pd.read_csv(study.RUN/'metrics.csv');assert len(metrics)==495
    with np.load(study.RUN/'paths.npz') as s:
        paths=dict(zip(s['ids'].astype(str),s['returns'].T.copy()));dates=s['dates'].copy()
    assert len(paths)==495 and len(dates)==1003
    stripped={key.removesuffix('/c1/b0.03'):p for key,p in paths.items() if key.endswith('/c1/b0.03')}
    contrasts=[]
    for g in study.parent.GROUPS:
        contrasts += [(s['id']+'/'+g,'baseline/'+g) for s in study.recipes() if s['kind']=='model']
        contrasts += [(s['id']+'/'+g,s['signal']+'/cash/'+g) for s in study.recipes() if s['kind']=='model' and s['mode']=='switch50']
    noops=sum(np.array_equal(stripped[a],stripped[b]) for a,b in contrasts)
    amendment=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        reason='Inherited bootstrap rejects deterministic zero differences. Raw forecasts never triggered shorting;60of120contrasts have exactly identical daily paths.',
        change='Use mathematically exact zero differences/intervals for identical paths; joint bootstrap other60contrasts with identical-path columns deduplicated. Family remains120contrasts. No resampling epsilon, trading rule, calibration, account path or forecast changes.',
        deterministic_contrasts=noops,source_sha256=study.parent.sha(__file__),
        protocol_sha256=study.parent.sha(study.RUN/'PROTOCOL.json'),
        evaluated_metrics_sha256=study.parent.sha(study.RUN/'metrics.csv'),
        evaluated_paths_sha256=study.parent.sha(study.RUN/'paths.npz'))
    target=study.RUN/'STATISTICAL_AMENDMENT.json'
    if target.exists():raise ValueError('Existing amendment; do not overwrite')
    study.parent.write(target,amendment)
    outputs=[]
    for block in [5,20,60]:
        print(f'Bootstrap block{block}:120contrasts,{noops}exact no-ops',flush=True)
        outputs.append(inference_with_noops(stripped,contrasts,block))
    pd.concat(outputs).to_csv(study.RUN/'uncertainty.csv',index=False)
    for row in metrics.itertuples():
        for field,value in performance(paths[row.id]).items():assert abs(getattr(row,field)-value)<1e-10
        assert abs(row.terminal_nav-1e6*np.prod(1+paths[row.id]))<1e-5
    with np.load(study.previous.RUN/'paths.npz') as s:old=dict(zip(s['ids'].astype(str),s['returns'].T))
    parity=[]
    for row in metrics[metrics.signal.isin(['none','qqq_ma200'])].itertuples():
        recipe='vol20/baseline' if row.signal=='none' else 'vol20/ma200/'+row.mode
        expected=old[f'{recipe}/{row.group}/c{row.cost_multiplier}/b{row.borrow_rate}']
        error=float(np.max(np.abs(paths[row.id]-expected)));assert error<1e-12
        parity.append(dict(id=row.id,max_return_error=error))
    study.verify()
    study.parent.write(study.RUN/'ACCOUNT_CHECKS.json',dict(status='passed',accounts=495,
        prior_path_parity=parity,exact_noop_contrasts=noops,
        max_self_financing_error=float(metrics.max_accounting_error.max())))
    study.DOC.mkdir(parents=True,exist_ok=True)
    for name in ['PROTOCOL.json','FREEZE.json','metrics.csv','annual.csv','targets.csv','base_rankings.csv',
                 'uncertainty.csv','signals.csv','calibration.csv','ACCOUNT_CHECKS.json','STATISTICAL_AMENDMENT.json']:
        shutil.copyfile(study.RUN/name,study.DOC/name)
    study.parent.write(study.RUN/'COMPLETE.json',dict(status='complete',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.name:study.parent.sha(p) for p in study.RUN.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    shutil.copyfile(study.RUN/'COMPLETE.json',study.DOC/'COMPLETE.json')
    print('Completed495unchangedaccounts and120contrasts',flush=True)


if __name__=='__main__':finish()
