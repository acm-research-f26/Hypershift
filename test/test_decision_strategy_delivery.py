"""Independent evidence checks must detect plausible corrupt ledgers."""
from dataclasses import asdict
import numpy as np
import pytest
from experiments.decision_strategy.audit import fixture
from experiments.decision_strategy.replay import run
from experiments.decision_strategy.runner import csv_gz
from experiments.decision_strategy_independent import fill_audit
from experiments.strategy_validation.config import Execution
from experiments.strategy_search.storage import arrays


def recorded_fixture(tmp_path):
    m=fixture();settings=Execution()
    candidates=[dict(origin=60,stock=0,hedges=[1],side=1,prediction=1.,edge_mean=.01,edge_se=0.,
        relative=.01,horizon=30,gate='no_gate',design='D01')]
    result=run(m,candidates,settings,np.ones(1),np.full(2,2.))
    for key in ('fills','orders','decisions'):csv_gz(tmp_path/('shadow_'+key+'.csv.gz'),result[key])
    arrays(tmp_path/'minute_tapes.npz',shadow_path=result['path'],shadow_holdings=result['holdings'],
        scale=np.zeros(1),shadow_daily=np.array([[0,0,100000.,0,0,0]]),sessions=m['sessions'])
    m['open_marks']=m['open'];m['close_marks']=m['close']
    receipt=dict(root=str(tmp_path),settings=asdict(settings),mode='minute_open_proxy')
    return receipt,m,result


def test_independent_audit_accepts_reconciled_integer_fixture(tmp_path):
    r,m,result=recorded_fixture(tmp_path);proof=fill_audit(r,m,True)
    assert proof['independent_financial_and_participation_checks_passed']
    assert proof['filled_children']==len(result['fills']) and proof['fill_risk_valid']


def test_independent_audit_detects_parent_fee_tampering(tmp_path):
    r,m,result=recorded_fixture(tmp_path);result['fills'][0]['commission']+=.01
    csv_gz(tmp_path/'shadow_fills.csv.gz',result['fills'])
    with pytest.raises(ValueError,match='parent fee'):fill_audit(r,m,True)


def test_independent_audit_detects_reused_participation(tmp_path):
    r,m,result=recorded_fixture(tmp_path);m['volume'][:]=0
    with pytest.raises(ValueError,match='fill/volume/cash'):fill_audit(r,m,True)


def test_cash_comparison_has_zero_paired_difference_and_explicit_uncertainty():
    from experiments.decision_strategy_delivery import paired_uncertainty
    proof=paired_uncertainty(np.zeros(40),np.zeros(40))
    assert proof['mean_daily_return_difference']==0 and proof['CI95']==[0,0]


def test_bundle_verifies_exact_decompressed_member_bytes(tmp_path):
    from experiments.decision_strategy_artifacts import package
    from experiments.strategy_search.storage import sha
    (tmp_path/'a.txt').write_text('a'*100);(tmp_path/'b.txt').write_text('b'*100)
    manifest=package(tmp_path,limit=120)
    assert len(manifest['parts'])==2
    assert all(p['CRC_verified'] and p['member_SHA256_verified'] for p in manifest['parts'])
    assert {r['sha256'] for r in manifest['files']}=={sha(tmp_path/'a.txt'),sha(tmp_path/'b.txt')}
    repeated=package(tmp_path,limit=120)
    assert [p['sha256'] for p in repeated['parts']]==[p['sha256'] for p in manifest['parts']]


def test_queue_preserves_exact_finite_plan_and_skips_completed_jobs(tmp_path):
    from experiments.decision_strategy_queue import tasks
    from experiments.decision_strategy_parallel import plan
    from experiments.decision_strategy.config import SCOPES
    jobs=plan();pending=tasks(tmp_path,SCOPES)
    assert len(jobs)==len(set(jobs))==315 and sum(n for _,_,n in pending)==315
    scope,recipe,gate,scenario=jobs[0];p=tmp_path/'trials'/'__'.join((scope,recipe,gate,scenario))/'receipt.json'
    p.parent.mkdir(parents=True);p.write_text('{}')
    assert sum(n for _,_,n in tasks(tmp_path,SCOPES))==314


def test_cash_export_reuse_cannot_erase_a_position_or_realized_activity():
    from experiments.decision_strategy_delivery import cash_tape
    p=np.zeros((3,13));p[:,:3]=100000;held=np.zeros((3,2),int)
    assert cash_tape(p,held,100000)
    held[1,0]=1;assert not cash_tape(p,held,100000)
    held[:]=0;p[1,7]=100;assert not cash_tape(p,held,100000)


def test_single_scope_cannot_be_split_into_character_paths(tmp_path):
    from experiments.decision_strategy_queue import tasks
    from experiments.decision_strategy.config import SCOPES
    with pytest.raises(TypeError,match='sequence'):tasks(tmp_path,SCOPES[2])
    with pytest.raises(ValueError,match='Unauthorized'):tasks(tmp_path,('4',))
    assert sum(n for _,_,n in tasks(tmp_path,(SCOPES[2],)))==105
