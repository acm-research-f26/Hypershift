"""Adversarial fixtures for the decision-study execution contract."""
from dataclasses import replace
import numpy as np
import pytest
from experiments.decision_strategy.execution import Book, exposures, integer_targets, round_trip_cost
from experiments.strategy_validation.config import Execution


def book(prices=(100.,100.)):
    p=np.array(prices);s=Execution();b=Book(len(p),s,np.zeros(len(p),int))
    q={j:(v-.01,v+.01,1000,1000,1_000_000_000) for j,v in enumerate(p)}
    return b,p,q


@pytest.mark.parametrize('stress',['normal','long_first','short_first','hedge_unavailable'])
def test_sequential_fills_obey_hard_net_even_when_hedge_is_missing(stress):
    b,p,q=book();b.execute(1,1_000_000_000,{0:39,1:-39},p,np.full(2,100000),q,'entry','entry',stress)
    assert all(abs(f['post_net'])<=.0025+1e-12 for f in b.fills)
    assert all(f['post_error']<=1e-10 for f in b.fills)
    if stress=='hedge_unavailable':
        assert b.q[0]>0 and b.q[1]==0
        assert b.q[0]<=2
    else:
        np.testing.assert_array_equal(b.q,[39,-39])


def test_parent_minimum_is_charged_once_across_partial_children():
    b,p,q=book();b.execute(1,1_000_000_000,{0:10,1:-10},p,np.full(2,100000),q,'e','entry')
    assert sum(f['commission'] for f in b.fills)==pytest.approx(.7)


def test_displayed_size_and_minute_participation_are_not_reused():
    b,p,q=book();q={j:(v[0],v[1],2,2,v[4]) for j,v in q.items()}
    b.execute(1,1_000_000_000,{0:10,1:-10},p,np.full(2,100000),q,'e','entry')
    b.execute(2,2_000_000_000,{0:10,1:-10},p,np.full(2,100000),q,'e','entry')
    np.testing.assert_array_equal(b.q,[2,-2])
    # A new quote can replenish depth, but the same completed-volume budget
    # cannot be reused by a second basket in the same minute.
    b,p,q=book();b.execute(1,1_000_000_000,{0:10,1:-10},p,np.full(2,100),q,'a','entry')
    b.execute(1,1_000_000_000,{0:20,1:-20},p,np.full(2,100),q,'b','entry')
    np.testing.assert_array_equal(b.q,[1,-1])


def test_missing_stale_or_future_quote_never_fills_and_attempt_is_retained():
    for quote in [None,(99,101,10,10,-6_000_000_000),(99,101,10,10,2_000_000_000)]:
        b,p,_=book();b.execute(1,1_000_000_000,{0:1},p,np.full(2,100000),{0:quote},'e','entry')
        assert not b.fills and len(b.attempts)==1 and not b.attempts[0]['quote_valid']


def test_required_exit_can_retry_after_missing_quote():
    b,p,q=book();b.execute(1,1_000_000_000,{0:10,1:-10},p,np.full(2,100000),q,'e','entry')
    b.execute(2,2_000_000_000,{0:0,1:0},p,np.full(2,100000),{},'x','exit')
    assert b.q.any()
    q={j:(99.99,100.01,1000,1000,3_000_000_000) for j in range(2)}
    b.execute(3,3_000_000_000,{0:0,1:0},p,np.full(2,100000),q,'x','exit')
    assert not b.q.any()
    expected=100000-sum(f['shares']*f['price']+sum(f[k] for k in ('commission','exchange','clearing','regulatory')) for f in b.fills)
    assert b.cash==pytest.approx(expected)


def test_integer_basket_targets_include_all_hedges_and_skip_impossible_band():
    q=integer_targets([100,60,70,80],[1,-1,-1,-1],[1,1/3,1/3,1/3],4000,100000,np.zeros(4,int))
    assert (q!=0).all() and q[0]>0 and (q[1:]<0).all() and abs(q@np.array([100,60,70,80]))<=100
    assert not integer_targets([5000,5000],[1,-1],[1,1],4000,100000,np.zeros(2,int)).any()


def test_cost_gate_counts_every_entry_exit_and_hedge_in_dollars():
    s=replace(Execution(),exchange_taker_share=0,clearing_share=0,cat_share=0,sec_sell_fraction=0,commission_pass_through=0)
    c=round_trip_cost([20,-20],[100,100],[.02,.02],[2,2],30,s)
    assert c['fees']==pytest.approx(1.4)
    assert c['spread']==pytest.approx(.8)
    assert c['buffer']==pytest.approx(.4)
    assert c['borrow']>0


def test_risk_is_checked_after_each_fill_not_only_after_netting():
    b,p,q=book();b.execute(1,1_000_000_000,{0:39,1:-39},p,np.full(2,100000),q,'e','entry','long_first')
    held=np.zeros(2);cash=100000.
    for f in b.fills:
        held[f['stock']]+=f['shares']
        cash-=f['shares']*f['price']+sum(f[k] for k in ('commission','exchange','clearing','regulatory'))
        assert exposures(held,p,cash,b.sectors)['error']<=1e-10


def test_full_minute_replay_reconciles_and_includes_closing_exit():
    from experiments.decision_strategy.audit import fixture
    from experiments.decision_strategy.replay import run,verify
    m=fixture();c=[dict(origin=60,stock=0,hedges=[1],side=1,prediction=1,edge_mean=.01,edge_se=0,
        relative=.01,horizon=500,gate='no_gate',design='D01')]
    r=run(m,c,Execution(),np.ones(1),np.full(2,2.))
    proof=verify(m,r,Execution())
    assert proof['accounting_reconciled'] and proof['fill_risk_valid'] and proof['no_overnight']
    assert r['roundtrips'][0]['exit_reason']=='closing' and r['roundtrips'][0]['exit']==385


def test_data_guard_rejects_reserved_and_unexposed_dates_before_access():
    from experiments.decision_strategy.data import guard_dates
    protocol={'binding':{'protected':{'reserved_session_dates':['2026-10-13']},'partitions':{'a':{'x':{'dates':['2025-01-02']}}}}}
    guard_dates({'2025-01-02'},'a',protocol)
    for d in ['2026-10-13','2024-01-02']:
        with pytest.raises(ValueError,match='reserved'):guard_dates({d},'a',protocol)


def test_oof_sessions_expand_strictly_and_heldout_cannot_enter_fit():
    from experiments.decision_strategy.calibration import fold_sessions
    for a,b in fold_sessions(list(range(60))):
        assert max(a)<min(b) and not set(a)&set(b) and max(b)<60


def test_delayed_labels_cannot_cross_sessions():
    from experiments.decision_strategy.calibration import labels
    m={'open':np.array([[100,100],[100,100],[110,100],[120,100],[200,100],[200,100]],float),'sessions':np.array([0,0,0,0,1,1])}
    y=labels(m,np.array([0,1]),np.array([0,0]),np.array([[1],[1]]),1)
    assert y[0]==pytest.approx(120/110-1) and np.isnan(y[1])


def test_calibration_uncertainty_uses_session_means_and_insufficient_is_cash():
    from experiments.decision_strategy.calibration import calibrate,edge
    c=calibrate(np.ones(2000),np.full(2000,.001),np.repeat(np.arange(20),100))
    assert c['valid'];mean,se=edge(c,.001)
    assert mean==pytest.approx(.001) and se==pytest.approx(0)
    c=calibrate(np.ones(100),np.ones(100),np.zeros(100))
    assert not c['valid'] and not np.isfinite(edge(c,1)[1])


def test_trial_npz_writer_and_readback_preserve_equity_and_holdings(tmp_path):
    from experiments.decision_strategy.audit import fixture
    from experiments.decision_strategy.replay import run,verify
    from experiments.strategy_search.storage import arrays
    m=fixture();r=run(m,[],Execution(),np.ones(1),np.full(2,2.))
    path=tmp_path/'tape.npz';arrays(path,equity=r['path'],holdings=r['holdings'])
    with np.load(path) as z:
        np.testing.assert_array_equal(z['equity'],r['path'])
        np.testing.assert_array_equal(z['holdings'],r['holdings'])


def test_extreme_quantile_excludes_nontradable_zero_vol_windows():
    from experiments.decision_strategy.signals import learn_structure
    rng=np.random.default_rng(12);t,n=390,261
    eligible=np.ones((t,n),bool);eligible[:,:80]=False
    vol=np.full((t,n),.001);vol[:,:80]=0
    m=dict(sessions=np.zeros(t,int),clock=np.arange(t),industry=np.zeros(n,int),
        volume=np.full((t,n),1000.),close=np.full((t,n),100.),vol=vol,eligible_full=eligible,
        stock_etf=np.full(250,250,int))
    f=dict(ok=np.ones((t,n),bool),own5=rng.normal(0,.001,(t,n)),own15=rng.normal(0,.001,(t,n)),residual5=rng.normal(0,.001,(t,250)),
        relative=np.full((t,250),.001),peerok=np.ones((t,250),bool))
    result=learn_structure(m,f,[0])
    assert result['extreme']==pytest.approx(.001/(.001*np.sqrt(15)))


def test_idle_marks_and_missing_proxy_open_cannot_create_hidden_nav_mismatch():
    from experiments.decision_strategy.audit import fixture
    from experiments.decision_strategy.replay import run,verify
    m=fixture()
    for key in ('open','close','volume','actions','vol'):m[key]=np.tile(m[key],(1,2))
    m.update(sector=np.zeros(4,int),industry=np.zeros(4,int),symbols=['A','B','C','D'])
    m['close'][121:123,2]=130.;m['open'][122,2]=np.nan
    cs=[dict(origin=i,stock=j,hedges=[j+1],side=1,prediction=1,edge_mean=.01,edge_se=0,
        relative=.01,horizon=30,gate='no_gate',design='D01') for i,j in [(60,0),(120,2)]]
    r=run(m,cs,Execution(),np.ones(1),np.full(4,2.))
    assert verify(m,r,Execution())['accounting_reconciled']
    assert not any(f['minute']==122 and f['stock']==2 for f in r['fills'])


def test_unstartable_integer_basket_is_zero_and_known_capacity_caps_hurdle_size():
    assert not integer_targets([1000,1000],[1,-1],[1,1],4000,100000,np.zeros(2,int)).any()
    assert not integer_targets([490,200],[1,-1],[1,1],4000,100000,np.zeros(2,int)).any()
    q=integer_targets([100,100],[1,-1],[1,1],4000,100000,np.zeros(2,int),capacity=[4,7])
    np.testing.assert_array_equal(q,[4,-4])


def test_no_trade_band_is_bypassed_by_new_signal_or_required_exit():
    from experiments.decision_strategy.execution import no_trade_targets
    np.testing.assert_array_equal(no_trade_targets([10,-10],[9,-9],[20,20]),[10,-10])
    np.testing.assert_array_equal(no_trade_targets([10,-10],[9,-9],[20,20],new_signal=True),[9,-9])
    np.testing.assert_array_equal(no_trade_targets([1,-1],[0,0],[20,20],required=True),[0,0])


def test_completely_unfilled_expired_entries_release_basket_slots():
    from experiments.decision_strategy.audit import fixture
    from experiments.decision_strategy.replay import run,verify
    m=fixture();n=6;t=780
    for key in ('open','close','volume','actions','vol'):m[key]=np.tile(m[key],(2,3))
    m.update(sessions=np.repeat([0,1],390),starts=m['starts'][0]+np.arange(t)*60_000_000_000,
        sector=np.zeros(n,int),industry=np.zeros(n,int),symbols=list('ABCDEF'),meta=dict(session_ids=[0,1]),evaluate_ids=[0,1])
    cs=[dict(origin=i,stock=j,hedges=[j+1],side=1,prediction=1,edge_mean=.01,edge_se=0,
        relative=.01,horizon=30,gate='no_gate',design='D01') for i,j in [(60,0),(60,2),(60,4),(450,0)]]
    # The causal targets and prior volume permit entries, but missing arrival
    # opens cause all first-session attempts to miss. Empty slots must expire.
    m['open'][62:66]=np.nan
    r=run(m,cs,Execution(),np.ones(2),np.full(n,2.))
    assert verify(m,r,Execution())['accounting_reconciled']
    assert sum(x['exit_reason']=='unfilled_entry_expired' for x in r['roundtrips'])==3
    assert any(x['session']==1 and x['completed_roundtrip'] for x in r['roundtrips'])


def test_arithmetic_same_horizon_label_has_exact_equal_dollar_units():
    from experiments.decision_strategy.calibration import labels
    m={'open':np.array([[100,50],[100,50],[100,50],[120,55]],float),'sessions':np.zeros(4,int)}
    value=labels(m,np.array([0]),np.array([0]),np.array([[1]]),1)[0]
    cash_pnl=20*(120-100)-40*(55-50)
    assert value*2000==pytest.approx(cash_pnl)


@pytest.mark.parametrize('current',[.004,.016])
def test_D03_exit_uses_completed_industry_residual_and_execution_delay(current):
    from experiments.decision_strategy.audit import fixture
    from experiments.decision_strategy.replay import run,verify
    m=fixture();m['industry_residual15']=np.full((390,2),.01);m['industry_residual15'][70:]=current
    c=[dict(origin=60,stock=0,hedges=[1],side=-1,prediction=-1.,edge_mean=.01,edge_se=0,
        relative=.01,entry_industry_residual=.01,horizon=30,gate='no_gate',design='D03')]
    r=run(m,c,Execution(),np.ones(1),np.full(2,2.))
    assert verify(m,r,Execution())['accounting_reconciled']
    assert r['roundtrips'][0]['exit']==72 and r['roundtrips'][0]['exit_reason']=='gap_or_stop'


def test_oof_and_production_share_the_same_prespecified_direction_filter():
    from experiments.decision_strategy.signals import direction_mask
    relative=np.array([.01,.01,-.01,-.01]);score=np.array([-.1,.1,.1,-.1])
    np.testing.assert_array_equal(direction_mask({'design':'D03','arm':'stall'},relative,score),[True,False,True,False])
    np.testing.assert_array_equal(direction_mask({'design':'D03','arm':'momentum'},relative,score),[False,True,False,True])
