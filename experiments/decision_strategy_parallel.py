"""Two bounded CPU workers; unchanged frozen strategy functions and trial set."""
import csv
import gzip
import io
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pickle
import numpy as np
from experiments.strategy_search.storage import atomic,read,sha,utc,clean
from experiments.decision_strategy.config import STUDY,SCOPES,CATALOG,GATES,COSTS,freeze
from experiments.decision_strategy.data import load_market
from experiments.decision_strategy.signals import features,candidates
from experiments.decision_strategy.runner import save_trial,select,csv_gz
from experiments.longer_horizon.config import proxy_settings


def plan():
    return [(scope,c['id'],gate,'cost'+format(cost,'g')) for scope in SCOPES for c in CATALOG
        for gate in (GATES if c['id'] in ('D01','D03','D04') else ('net_edge',)) for cost in COSTS]


def stream_text(rows):
    if not rows:return 'empty\n'
    stream=io.StringIO(newline='');keys=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
    for r in rows:writer.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()})
    return stream.getvalue()


def worker(output,scope,identifiers,tag):
    output=Path(output);protocol=read(output/'protocol.json')
    if not read(output/'engineering/train_streams.json')['all_accounting_reconciled']:raise ValueError('Training execution audit incomplete')
    fit_path=output/'fits/models.pkl'
    if sha(fit_path)!=read(output/'fits/manifest.json')['sha256']:raise ValueError('Changed fitted parameters')
    with fit_path.open('rb') as f:fits=pickle.load(f)
    spread=np.asarray(read(output/'training_spreads.json')['spread_bps'],float)
    m=load_market(scope,protocol,output);f=features(m);rows=[]
    for c in [c for c in CATALOG if c['id'] in identifiers]:
        for gate in (GATES if c['id'] in ('D01','D03','D04') else ('net_edge',)):
            stream=candidates(m,f,fits[c['id']],c,gate)
            dest=output/'candidates'/(scope+'__'+c['id']+'__'+gate+'.csv.gz')
            if dest.exists():
                with gzip.open(dest,'rt',encoding='utf-8',newline='') as g:old=g.read()
                if old!=stream_text(stream):raise ValueError('Candidate bytes changed during orchestration-only resume:'+dest.name)
            else:csv_gz(dest,stream)
            for cost in COSTS:
                r=save_trial(m,stream,c,gate,'cost'+format(cost,'g'),proxy_settings(cost),spread,output)
                rows.append(r)
                atomic(output/('worker_'+tag+'.json'),dict(updated_UTC=utc(),scope=scope,completed=len(rows),last_trial=r['key']))
    return [r['key'] for r in rows]


def receipts(output,scopes):
    result=[]
    for scope,identifier,gate,scenario in plan():
        if scope not in scopes:continue
        key='__'.join((scope,identifier,gate,scenario));p=Path(output)/'trials'/key/'receipt.json'
        if not p.exists():raise ValueError('Missing required frozen trial:'+key)
        result.append(read(p))
    return result


def main():
    output=STUDY/'decision_strategy_execution_2026-10-27_v6';protocol=freeze(output)
    binding=dict(script_sha256=sha(__file__),protocol_sha256=sha(output/'protocol.json'),jobs=plan(),workers=2,
        policy_change=False,learned_parameter_change=False,later_after_development_selection=True,
        source_commit='b6208f79c46efea51d38e1a554fb847f0a658c64')
    dest=output/'parallel_orchestration_receipt.json'
    if dest.exists() and read(dest)['binding']!=json.loads(json.dumps(binding)):raise ValueError('Changed orchestration')
    if not dest.exists():atomic(dest,dict(frozen_UTC=utc(),binding=binding))
    names=[c['id'] for c in CATALOG]
    # Independent cohorts use fixed first60 parameters and independent causal
    # volatility histories. There is no fitting or cross-cohort adaptation here.
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,str(output),scope,names,scope) for scope in SCOPES[:2]]
        for future in as_completed(jobs):future.result()
    dev=receipts(output,SCOPES[:2]);select(dev,output)
    atomic(output/'progress.json',dict(stage='development_complete_selection_frozen',updated_UTC=utc(),completed_proxy_trials=210,budget=315))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,str(output),SCOPES[2],names[k::2],'later_'+str(k)) for k in range(2)]
        for future in as_completed(jobs):future.result()
    rows=receipts(output,SCOPES)
    atomic(output/'proxy_results.json',dict(rows=rows,trials=len(rows),protocol_sha256=sha(output/'protocol.json')))
    csv_gz(output/'proxy_ledger.csv.gz',[dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],**r['metrics']) for r in rows])
    atomic(output/'progress.json',dict(stage='proxy_complete',updated_UTC=utc(),completed_proxy_trials=len(rows),budget=315))


if __name__=='__main__':main()
