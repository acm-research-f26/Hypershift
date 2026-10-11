"""Prefit disjoint late-2025 groups while the frozen runner fits earlier folds.

This schedules the same deterministic fit_seed_job, without evaluating returns.
Workers stop starting new fits when the main runner enters its 2025 fold.
The main runner's first ten THINK/Mixer fits provide a buffer for in-flight
Huber fits to finish; no worker writes those first ten fits or preprocessing.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pit_research_engine import load_data,fold,historical_groups
from run_robust_research import scale_inputs,fit_seed_job

RUN=Path('runs/nasdaq100_historical_monthly_20261011')


def fit_groups(kinds):
    for source,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
    plan=json.loads((RUN/'PROTOCOL.json').read_text());data=load_data()
    train,val,test=fold(data,2025);x,_,_=scale_inputs(data,train);groups=historical_groups(data,train)
    args=SimpleNamespace(device='cuda',causal_membership=True,epochs=plan['epochs'],
                         patience=plan['patience'],lr=plan['lr'],batch=8)
    folder=RUN/'2025';folder.mkdir(exist_ok=True);finished=[]
    for kind in kinds:
        for seed in plan['training_seeds']:
            if (folder/'preprocessing.npz').exists():return finished
            path=folder/f'{kind}_{seed}_predictions.npz';record_path=folder/f'{kind}_{seed}_record.json'
            if record_path.exists():continue
            print(f'Prefitting 2025 {kind} seed {seed}',flush=True)
            vp,tp,chosen,completed,parameters=fit_seed_job(data,x,train,val,test,kind,seed,groups,args,folder)
            assert np.isfinite(tp[data['mask'][test]]).all()
            np.savez_compressed(path,validation=vp,test=tp)
            record=dict(year=2025,model=kind,seed=seed,chosen_epoch=chosen,epochs_run=completed,
                        parameters=parameters,checkpoint_sha256=hashlib.sha256((folder/f'{kind}_{seed}.pt').read_bytes()).hexdigest())
            temporary=record_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(record_path)
            finished.append(dict(model=kind,seed=seed))
    return finished


def main():
    assert not (RUN/'2025/preprocessing.npz').exists(),'Main runner already at 2025; no prefit'
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(fit_groups,['think_huber']),
              executor.submit(fit_groups,['mlp','stockmixer','think_mix_huber'])]
        results=[job.result() for job in jobs]
    record=dict(training_workers=2,same_locked_fit_routine='run_robust_research.fit_seed_job',
                financial_test_metrics_computed=False,completed=results)
    (RUN/'PREFIT_WORKERS.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Disjoint prefit workers complete:',sum(map(len,results)),'fits',flush=True)


if __name__=='__main__':main()
