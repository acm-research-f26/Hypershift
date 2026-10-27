"""Explicit v5 -> v6 single-guard correction with per-trial equivalence proof."""
import gzip
import csv
import json
import math
import os
from pathlib import Path
import shutil
from experiments.strategy_search.storage import atomic,read,sha,utc
from experiments.decision_strategy.config import STUDY,freeze


OLD="not np.isfinite(uncertainty) or gross-uncertainty<=cost['total']"
NEW="not np.isfinite([gross,uncertainty,cost['total']]).all() or gross-uncertainty<=cost['total']"


def copy_immutable(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        if shutil.disk_usage(dest.parent).free-source.stat().st_size<10_000_000_000:raise RuntimeError('10GB reserve')
        try:os.link(source,dest)
        except OSError:shutil.copyfile(source,dest)
    if sha(source)!=sha(dest):raise ValueError('Changed copied evidence:'+str(dest))


def decision_equivalence(root,gate):
    if gate!='net_edge':return dict(proved=True,reason='Changed conditional only executes for net_edge',cost_checks=0)
    count=0;nonfinite=[]
    for prefix in ('','shadow_'):
        with gzip.open(root/(prefix+'decisions.csv.gz'),'rt',encoding='utf-8') as f:
            for d in csv.DictReader(f):
                if not d.get('estimated_cost_dollars'):continue
                cost=json.loads(d['estimated_cost_dollars'])['total']
                values=[d['expected_gross_dollars'],d['uncertainty_dollars'],cost];count+=1
                if any(v in (None,'') or not math.isfinite(float(v)) for v in values):nonfinite.append([prefix,d['origin'],d['disposition']])
    return dict(proved=not nonfinite,reason='All inputs consumed at the changed guard are finite, so its Boolean outcome is identical',cost_checks=count,nonfinite=nonfinite)


def migrate():
    old=STUDY/'decision_strategy_execution_2026-10-27_v5';new=STUDY/'decision_strategy_execution_2026-10-27_v6'
    code=Path(__file__).parent/'decision_strategy';prior=read(old/'protocol.json')
    old_replay=(old/'source_v5/replay.py').read_text(encoding='utf-8')
    if old_replay.count(OLD)!=1 or old_replay.replace(OLD,NEW)!=(code/'replay.py').read_text(encoding='utf-8'):raise ValueError('Execution delta exceeds the finite-cost guard')
    for name in prior['binding']['source']:
        if name not in ('replay.py','config.py','runner.py') and sha(code/name)!=prior['binding']['source'][name]:raise ValueError('Unexpected strategy/source change')
    old_runner=(old/'source_v5/runner.py').read_text(encoding='utf-8')
    if old_runner.replace("default=str(STUDY/'decision_strategy_execution_2026-10-27_v5')","default=str(STUDY/'decision_strategy_execution_2026-10-27_v6')")!=(code/'runner.py').read_text(encoding='utf-8'):raise ValueError('Unexpected runner change')
    for name in ('causal_classifications.json','training_spreads.json','protected_sessions.json','qualification_interpretation.json','model_package_references.json','cloud_destination.json','source_commit_v1.json'):
        copy_immutable(old/name,new/name)
    current=freeze(new)
    a=prior['binding'].copy();b=current['binding'].copy()
    for key in ('version','source','correctness_version'):a.pop(key);b.pop(key)
    if a!=b:raise ValueError('Correctness correction changed protocol/data/recipe assumptions')
    for folder in ('fits','quotes','engineering'):
        for p in (old/folder).rglob('*'):
            if p.is_file() and p.name!='manifest.json':copy_immutable(p,new/p.relative_to(old))
        for p in (old/folder).rglob('manifest.json'):
            if folder=='fits':continue
            copy_immutable(p,new/p.relative_to(old))
    fit=read(old/'fits/manifest.json');fit['protocol_sha256']=sha(new/'protocol.json');fit['reused_from']=str(old/'fits/manifest.json');fit['original_manifest_sha256']=sha(old/'fits/manifest.json')
    atomic(new/'fits/manifest.json',fit)
    reused=[];rerun=[]
    for p in sorted((old/'trials').glob('*/receipt.json')):
        r=read(p);proof=decision_equivalence(p.parent,r['gate'])
        if not proof['proved']:rerun.append(dict(key=r['key'],proof=proof));continue
        root=new/'trials'/r['key']
        for name,h in r['files'].items():
            if sha(p.parent/name)!=h:raise ValueError('Changed original trial')
            copy_immutable(p.parent/name,root/name)
        atomic(root/'import_equivalence.json',dict(source_trial=str(p),source_receipt_sha256=sha(p),source_protocol_sha256=sha(old/'protocol.json'),
            correction='single finite-cost guard',unchanged_execution_stream=True,**proof))
        r.update(root=str(root),protocol_sha256=sha(new/'protocol.json'),imported_from=str(p),original_receipt_sha256=sha(p),equivalence_sha256=sha(root/'import_equivalence.json'))
        atomic(root/'receipt.json',r);atomic(root/'status.json',dict(status='complete_reused_with_exact_guard_equivalence',original=str(p)))
        reused.append(dict(key=r['key'],proof=proof))
    source=new/'source_v6'
    for p in code.glob('*.py'):copy_immutable(p,source/p.name)
    copy_immutable(code.parents[1]/'test/test_decision_strategy.py',source/'test_decision_strategy.py')
    atomic(new/'version_transition.json',dict(created_UTC=utc(),source_v5=str(old),new_protocol_sha256=sha(new/'protocol.json'),
        exact_execution_delta=[OLD,NEW],unchanged_binding_except_version_source_receipt=True,
        unchanged_fits_sha256=sha(new/'fits/models.pkl'),train_execution_audits_reused='All train audits used no_gate; the conditional is unreachable.',
        reused_trials=reused,rerun_trials=rerun,
        original_completed_trials=len(list((old/'trials').glob('*/receipt.json'))),
        interrupted_originals=[p.parent.name for p in (old/'trials').glob('*/status.json') if not (p.parent/'receipt.json').exists()],
        no_later_cohort_opened=True,no_threshold_or_recipe_change=True))
    return dict(reused=len(reused),must_rerun=len(rerun),output=str(new))


if __name__=='__main__':print(migrate())
