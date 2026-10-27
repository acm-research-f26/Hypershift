"""Control conformance correction and identical-plan two-worker continuation."""
from pathlib import Path
import json
import pickle
import numpy as np
from experiments.strategy_search.storage import atomic,read,sha,utc
from experiments.decision_strategy.config import STUDY,SCOPES,CATALOG,freeze
from experiments.decision_strategy_versioning import copy_immutable
from experiments.decision_strategy.data import load_market
from experiments.decision_strategy.signals import features,fit_recipe
from experiments.decision_strategy_queue import stage,tasks
from experiments.decision_strategy_parallel import plan,receipts
from experiments.decision_strategy.runner import select,csv_gz


def prepare():
    old=STUDY/'decision_strategy_execution_2026-10-27_v6';new=STUDY/'decision_strategy_execution_2026-10-27_v7'
    code=Path(__file__).parent/'decision_strategy';prior=read(old/'protocol.json')
    anchor="    if arm=='etf_only':x[:,[0,1,4,5]]=0\n"
    addition="    if design=='D04' and arm=='stock_only':x[:,5]=0\n"
    before=(old/'source_v6/signals.py').read_text(encoding='utf-8')
    if before.count(anchor)!=1 or before.replace(anchor,anchor+addition)!=(code/'signals.py').read_text(encoding='utf-8'):raise ValueError('Signal delta exceeds stock-only restriction')
    for name,h in prior['binding']['source'].items():
        if name not in ('signals.py','config.py','runner.py') and sha(code/name)!=h:raise ValueError('Unexpected core change:'+name)
    runner=(old/'source_v6/runner.py').read_text(encoding='utf-8')
    if runner.replace("default=str(STUDY/'decision_strategy_execution_2026-10-27_v6')","default=str(STUDY/'decision_strategy_execution_2026-10-27_v7')")!=(code/'runner.py').read_text(encoding='utf-8'):raise ValueError('Unexpected runner change')
    for name in ('causal_classifications.json','training_spreads.json','protected_sessions.json','qualification_interpretation.json','model_package_references.json','cloud_destination.json','source_commit_v1.json'):
        copy_immutable(old/name,new/name)
    protocol=freeze(new);a=prior['binding'].copy();b=protocol['binding'].copy()
    for key in ('version','source','correctness_version'):a.pop(key);b.pop(key)
    if a!=b:raise ValueError('Recipe/criterion/data binding changed')
    transition=new/'version_transition_v7.json'
    if transition.exists():return new
    if (old/'selection.json').exists():raise ValueError('Expected v6 stop before first-wave selection; investigate chronology')
    affected=list((old/'trials').glob('*__D04_stock_only__*/receipt.json'))
    if affected:raise ValueError('Affected control already ran; explicit affected-outcome review required')
    for folder in ('quotes','engineering'):
        for p in (old/folder).rglob('*'):
            if p.is_file():copy_immutable(p,new/p.relative_to(old))
    fitsroot=new/'fits';fitsroot.mkdir(exist_ok=True)
    for p in (old/'fits').iterdir():
        if p.is_file() and p.name not in ('models.pkl','manifest.json','D04_stock_only.json','D04_stock_only.oof.npz'):copy_immutable(p,fitsroot/p.name)
    original=old/'fits/models.pkl'
    if sha(original)!=read(old/'fits/manifest.json')['sha256']:raise ValueError('Changed original fit')
    with original.open('rb') as stream:fits=pickle.load(stream)
    unchanged={c['id']:sha(old/'fits'/(c['id']+'.json')) for c in CATALOG if c['id']!='D04_stock_only'}
    m=load_market(SCOPES[0],protocol,new);f=features(m);spread=np.asarray(read(new/'training_spreads.json')['spread_bps'],float)
    c=next(c for c in CATALOG if c['id']=='D04_stock_only')
    fits[c['id']]=fit_recipe(m,f,c,fitsroot,spread)
    del m,f
    with (fitsroot/'models.pkl').open('wb') as stream:pickle.dump(fits,stream)
    atomic(fitsroot/'manifest.json',dict(sha256=sha(fitsroot/'models.pkl'),protocol_sha256=sha(new/'protocol.json'),
        train_ids=read(fitsroot/'D04_stock_only.json')['train_ids'],unchanged11_fits=unchanged,
        changed_control_only='D04_stock_only: remove industry breadth; first60 refit, no neural training'))
    for identifier,h in unchanged.items():
        if sha(fitsroot/(identifier+'.json'))!=h:raise ValueError('Unchanged recipe fit mutated')
    reused=[];interrupted=[]
    extra=('minute_nav.csv.gz','shadow_minute_nav.csv.gz','minute_holdings_sparse.csv.gz','shadow_minute_holdings_sparse.csv.gz','daily_evaluation.csv.gz','csv_reuse.json')
    for p in sorted((old/'trials').glob('*/receipt.json')):
        if not any(p.parent.name.startswith(s+'__') for s in SCOPES[:2]):continue
        r=read(p)
        if r['config']['id']=='D04_stock_only':raise ValueError('Affected output cannot be reused')
        root=new/'trials'/r['key'];root.mkdir(parents=True,exist_ok=True)
        for name,h in r['files'].items():
            if sha(p.parent/name)!=h:raise ValueError('Changed original tape')
            copy_immutable(p.parent/name,root/name)
        copy_immutable(p,root/'original_receipt_v6.json')
        for name in extra:
            if (p.parent/name).exists():copy_immutable(p.parent/name,root/name)
        if (p.parent/'import_equivalence.json').exists():copy_immutable(p.parent/'import_equivalence.json',root/'upstream_import_equivalence_v6.json')
        atomic(root/'import_equivalence.json',dict(proved=True,source_receipt=str(p),source_receipt_sha256=sha(p),
            reason='Exact branch delta applies only to D04 stock_only; this recipe and its fitted parameters are unchanged.',
            copied_trial_file_hashes=r['files'],no_outcome_or_risk_rule_change=True))
        r.update(root=str(root),protocol_sha256=sha(new/'protocol.json'),imported_from=str(p),original_receipt_sha256=sha(p),equivalence_sha256=sha(root/'import_equivalence.json'))
        atomic(root/'receipt.json',r);atomic(root/'status.json',dict(status='complete_reused_branch_equivalence',original=str(p)))
        for name in ('independent_execution.json','independent_attribution.json'):
            src=p.parent/name
            if not src.exists():continue
            proof=read(src)
            if proof['receipt_sha256']!=sha(p):raise ValueError('Cached audit owns a different receipt')
            proof.update(receipt_sha256=sha(root/'receipt.json'),reused_from=str(src),original_proof_sha256=sha(src),
                byte_identical_trial_data_verified=True,unchanged_recipe_and_auditor=True)
            atomic(root/name,proof)
        reused.append(r['key'])
    for p in (old/'candidates').glob('*.csv.gz'):
        if '__D04_stock_only__' not in p.name:copy_immutable(p,new/'candidates'/p.name)
    for p in (old/'trials').glob('*/status.json'):
        if not (p.parent/'receipt.json').exists():interrupted.append(dict(key=p.parent.name,status=read(p),status_sha256=sha(p)))
    for p in code.glob('*.py'):copy_immutable(p,new/'source_v7'/p.name)
    copy_immutable(code.parents[1]/'test/test_decision_strategy.py',new/'source_v7/test_decision_strategy.py')
    atomic(transition,dict(created_UTC=utc(),old_root=str(old),source_protocol_sha256=sha(old/'protocol.json'),new_protocol_sha256=sha(new/'protocol.json'),
        exact_signal_delta=[anchor,anchor+addition],affected_control_development_trials_before_fix=0,
        refit_only=['D04_stock_only'],unchanged11_fits=unchanged,reused_completed_trials=reused,interrupted_original_attempts=interrupted,
        first_wave_selection_was_not_yet_frozen=True,no_later_outcomes_opened=True,thresholds_and_qualification_unchanged=True))
    return new


def main():
    output=prepare();binding=json.loads(json.dumps(dict(script_sha256=sha(__file__),protocol_sha256=sha(output/'protocol.json'),
        queue_sha256=sha(Path(__file__).with_name('decision_strategy_queue.py')),worker_sha256=sha(Path(__file__).with_name('decision_strategy_parallel.py')),
        jobs=plan(),workers=2,later_after_development_selection=True)))
    p=output/'v7_orchestration_receipt.json'
    if p.exists():
        if read(p)['binding']!=binding:raise ValueError('Changed frozen orchestration')
    else:atomic(p,dict(frozen_UTC=utc(),binding=binding,missing_development_tasks=tasks(output,SCOPES[:2])))
    dev=stage(output,SCOPES[:2]);select(dev,output)
    atomic(output/'progress.json',dict(stage='development_complete_selection_frozen',updated_UTC=utc(),completed_proxy_trials=210,budget=315))
    stage(output,SCOPES[2]);rows=receipts(output,SCOPES)
    atomic(output/'proxy_results.json',dict(rows=rows,trials=len(rows),protocol_sha256=sha(output/'protocol.json')))
    csv_gz(output/'proxy_ledger.csv.gz',[dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],**r['metrics']) for r in rows])
    atomic(output/'progress.json',dict(stage='proxy_complete',updated_UTC=utc(),completed_proxy_trials=len(rows),budget=315))


if __name__=='__main__':main()
