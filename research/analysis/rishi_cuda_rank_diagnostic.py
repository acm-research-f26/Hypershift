"""All-checkpoint local CUDA score/rank diagnostics; never promotes replay gate."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ndx_reference_training import prepare,predictions
from ndx_reference_models import make_reference
from ndx_extended_hours import ExtendedModel,predict
from training_device import configure_device,device_metadata
from research_engine import target_weights
from run_robust_research import ensemble_predictions
from rishi_raw_replay import load_selected_checkpoint,SEEDS,GROUPS,PARENT,REFERENCE


def rank_changes(actual,expected,eligible):
    a=np.asarray(actual);b=np.asarray(expected);m=np.asarray(eligible,bool)
    if a.ndim!=2 or a.shape!=b.shape or a.shape!=m.shape or not np.isfinite(a[m]).all() or not np.isfinite(b[m]).all():raise ValueError('Finite aligned eligible scores required')
    changed=top5=top10=0
    for x,y,mask in zip(a,b,m):
        if not mask.any():raise ValueError('Empty eligible set')
        changed+=int(not np.array_equal(pd.Series(x[mask]).rank().to_numpy(),pd.Series(y[mask]).rank().to_numpy()))
        for k in (5,10):
            diff=not np.array_equal(target_weights(x,mask,k,'equal'),target_weights(y,mask,k,'equal'))
            if k==5:top5+=int(diff)
            else:top10+=int(diff)
    return dict(origins=len(a),rank_changed_origins=changed,top5_changed_origins=top5,top10_changed_origins=top10)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch18_cuda_rank';large=Path('results/RISHI_CUDA_RANK_DIAGNOSTIC_20261028')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    device=configure_device('cuda');env=device_metadata(device);rows=[];ensembles=[];failures=[];started=time.monotonic()
    for year in (2022,2023,2024,2025):
        p=prepare(year);mask=p['data']['mask'];n=len(p['data']['panel'].tickers);all_scores={}
        x=p['features'][...,:5].transpose(0,2,1,3).copy();extra=p['features'][...,5:].reshape(len(p['features']),n,320).copy()
        for family in ('B','D'):
            folder=(PARENT if family=='B' else REFERENCE/'dynamic_restricted')/str(year)
            for seed in SEEDS:
                checkpoint=folder/f'{seed}.pt';export=folder/f'{seed}_predictions.npz'
                try:
                    cp=load_selected_checkpoint(checkpoint)
                    model=ExtendedModel(n,[set(g) for g in cp['groups']],'after_hours_premarket') if family=='B' else make_reference('dynamic_restricted',n,p['snapshot'])
                    model.load_state_dict(cp['state_dict']);model.to(device);model.eval();outputs={};original={}
                    with np.load(export,allow_pickle=False) as saved:
                        for split in ('validation','test'):
                            ids=p[split];y=predict(model,x,extra,mask,ids,device,8) if family=='B' else predictions(model,p,ids,'dynamic_restricted',device)
                            z=saved[split].copy()
                            if y.shape!=z.shape or not np.isfinite(y).all() or not np.isfinite(z).all():raise ValueError('Nonfinite or misaligned exported forecast')
                            stats=rank_changes(y,z,mask[ids]);rows.append(dict(year=year,family=family,seed=seed,split=split,
                                max_value_error=float(np.max(np.abs(y-z))),checkpoint_sha256=digest(checkpoint),export_sha256=digest(export),**stats))
                            outputs[split]=y;original[split]=z
                    np.savez_compressed(large/f'{family}_{year}_{seed}.npz',validation=outputs['validation'],test=outputs['test'],
                        validation_ids=p['validation'],test_ids=p['test'],security_ids=np.asarray(p['data']['panel'].tickers,dtype='U'))
                    all_scores[family,seed]=(outputs['test'],original['test']);del model
                    print(year,family,seed,'complete',flush=True)
                except Exception as exc:
                    failures.append(dict(year=year,family=family,seed=seed,error=f'{type(exc).__name__}: {exc}'))
                    print(year,family,seed,'FAILED',type(exc).__name__,flush=True)
        masktest=mask[p['test']]
        for group,seeds in GROUPS.items():
            for family in ('B','D','E'):
                required=['B','D'] if family=='E' else [family]
                if any((f,s) not in all_scores for f in required for s in seeds):
                    ensembles.append(dict(year=year,group=group,family=family,status='incomplete'));continue
                pair={f:[ensemble_predictions([all_scores[f,s][i] for s in seeds],masktest) for i in (0,1)] for f in required}
                a,b=pair[family] if family!='E' else [0.5*pair['B'][i]+0.5*pair['D'][i] for i in (0,1)]
                ensembles.append(dict(year=year,group=group,family=family,status='complete',**rank_changes(a,b,masktest)))
    pd.DataFrame(rows).to_csv(out/'CHECKPOINT_SPLITS.csv',index=False)
    pd.DataFrame(ensembles).to_csv(out/'ENSEMBLES.csv',index=False)
    inputs=[Path(__file__),doc/'CUDA_ALL_RANK_DIAGNOSTIC_PROTOCOL.md',Path('research/analysis/rishi_raw_replay.py'),
        Path('latest/training/ndx_reference_training.py'),Path('research/models/ndx_extended_hours.py')]
    (out/'RESULTS.json').write_text(json.dumps(dict(expected_checkpoints=80,split_rows=len(rows),ensemble_rows=len(ensembles),failures=failures,
        device=env,seconds=time.monotonic()-started,input_sha256={str(q):digest(q) for q in inputs},
        limits='Diagnostic only: strict checkpoint gate, training prediction controls and interventions remain unresolved'),indent=2)+'\n')
    print('Completed diagnostics:',len(rows),'splits;',len(ensembles),'ensembles;',len(failures),'failures',flush=True)


if __name__=='__main__':main()
