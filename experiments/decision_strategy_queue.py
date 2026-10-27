"""Load-balanced two-worker orchestration of the identical frozen315 jobs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from experiments.strategy_search.storage import atomic,read,sha,utc
from experiments.decision_strategy.config import STUDY,SCOPES,CATALOG,GATES,COSTS,freeze
from experiments.decision_strategy_parallel import plan,worker,receipts
from experiments.decision_strategy.runner import select,csv_gz


def tasks(output,scopes):
    if isinstance(scopes,str):raise TypeError('Pass a sequence of full authorized cohort names')
    scopes=tuple(scopes)
    if any(scope not in SCOPES for scope in scopes):raise ValueError('Unauthorized cohort label')
    result=[]
    for scope in scopes:
        for c in CATALOG:
            expected=['__'.join((scope,c['id'],gate,'cost'+format(cost,'g'))) for gate in
                (GATES if c['id'] in ('D01','D03','D04') else ('net_edge',)) for cost in COSTS]
            missing=[key for key in expected if not (output/'trials'/key/'receipt.json').exists()]
            if missing:result.append((scope,c['id'],len(missing)))
    # Scheduling only. All jobs execute; no result/risk/profit statistic enters
    # this ordering. Completed receipts and stream bytes are verified by worker.
    return sorted(result,key=lambda r:(-r[2],r[0],r[1]))


def stage(output,scopes):
    jobs=tasks(output,scopes)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,str(output),scope,[identifier],scope+'_'+identifier+'_queue') for scope,identifier,_ in jobs]
        for future in as_completed(futures):future.result()
    return receipts(output,scopes)


def main():
    output=STUDY/'decision_strategy_execution_2026-10-27_v6';freeze(output)
    binding=dict(script_sha256=sha(__file__),core_protocol_sha256=sha(output/'protocol.json'),
        reused_worker_sha256=sha(__import__('experiments.decision_strategy_parallel',fromlist=['__file__']).__file__),
        jobs=plan(),workers=2,changed_strategy_or_fit=False,later_after_development_selection=True,
        reason='Independent fixed recipes share a work queue to avoid an idle CPU while a longer cohort finishes.')
    binding=json.loads(json.dumps(binding))
    p=output/'queue_orchestration_receipt.json'
    if p.exists():
        if read(p)['binding']!=binding:raise ValueError('Changed queue orchestration')
    else:atomic(p,dict(frozen_UTC=utc(),binding=binding,initial_missing_recipe_tasks=tasks(output,SCOPES[:2])))
    dev=stage(output,SCOPES[:2]);select(dev,output)
    atomic(output/'progress.json',dict(stage='development_complete_selection_frozen',updated_UTC=utc(),completed_proxy_trials=210,budget=315))
    stage(output,(SCOPES[2],));rows=receipts(output,SCOPES)
    atomic(output/'proxy_results.json',dict(rows=rows,trials=len(rows),protocol_sha256=sha(output/'protocol.json')))
    csv_gz(output/'proxy_ledger.csv.gz',[dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],**r['metrics']) for r in rows])
    atomic(output/'progress.json',dict(stage='proxy_complete',updated_UTC=utc(),completed_proxy_trials=len(rows),budget=315))


if __name__=='__main__':main()
