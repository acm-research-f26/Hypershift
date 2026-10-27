"""Operational handoff repair: exact frozen later105, after development selection."""
import json
from pathlib import Path
from experiments.strategy_search.storage import atomic,read,sha,utc
from experiments.decision_strategy.config import STUDY,SCOPES,freeze
from experiments.decision_strategy_parallel import plan,receipts
from experiments.decision_strategy_queue import stage
from experiments.decision_strategy.runner import csv_gz


def main():
    output=STUDY/'decision_strategy_execution_2026-10-27_v7';freeze(output)
    development=receipts(output,SCOPES[:2]);selection=read(output/'selection.json')
    if len(development)!=210 or not selection['only_development_used']:raise ValueError('Development selection incomplete')
    binding=json.loads(json.dumps(dict(script_sha256=sha(__file__),protocol_sha256=sha(output/'protocol.json'),
        queue_sha256=sha(Path(__file__).with_name('decision_strategy_queue.py')),
        worker_sha256=sha(Path(__file__).with_name('decision_strategy_parallel.py')),
        selection_sha256=sha(output/'selection.json'),jobs=[j for j in plan() if j[0]==SCOPES[2]],workers=2,
        strategy_or_fit_or_selection_change=False,repair='Pass a one-element cohort tuple; reject character-path scopes.')))
    dest=output/'later_orchestration_receipt_v2.json'
    if dest.exists():
        if read(dest)['binding']!=binding:raise ValueError('Changed repaired orchestration')
    else:atomic(dest,dict(frozen_UTC=utc(),binding=binding,original_failure=str(output/'orchestration_failure_v1.json')))
    stage(output,(SCOPES[2],));rows=receipts(output,SCOPES)
    atomic(output/'proxy_results.json',dict(rows=rows,trials=len(rows),protocol_sha256=sha(output/'protocol.json')))
    csv_gz(output/'proxy_ledger.csv.gz',[dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],**r['metrics']) for r in rows])
    atomic(output/'progress.json',dict(stage='proxy_complete',updated_UTC=utc(),completed_proxy_trials=len(rows),budget=315))


if __name__=='__main__':main()
