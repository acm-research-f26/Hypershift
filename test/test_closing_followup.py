"""Regression checks for paired liquidation, order expiry and causal filters."""
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('numba')
from experiments.strategy_validation.config import Execution
from experiments.strategy_validation.quotes import QUOTE_DTYPE
from experiments.closing_followup.engine import replay,balanced_batch,limits
from experiments.closing_followup.runner import verify
from experiments.closing_followup.signals import build_stock,fit,build_etf
from experiments.closing_followup.config import CATALOG


def example():
    t,n=90,6;prices=np.full((t,n),100.)
    start=np.datetime64('2026-09-25T18:30','ns').astype(np.int64)
    m=dict(open=prices.copy(),close=prices.copy(),volume=np.full((t,n),1e6),
        sessions=np.zeros(t,np.int64),starts=start+np.arange(t)*60_000_000_000,
        actions=np.zeros((t,n),bool),sector=np.zeros(n,np.int64),symbols=list('ABCDEF'),
        meta={'session_ids':[0]},evaluate_ids=[0],scope='initial_validation',
        vol=np.full((t,n),.001),eligible_full=np.ones((t,n),bool))
    m['completed']=m['starts']+60_000_000_000
    base=np.zeros_like(prices);base[60:,:2]=[.04,-.04]
    quotes=np.zeros((t,n),dtype=QUOTE_DTYPE);quotes['bid']=99.99;quotes['ask']=100.01
    quotes['bid_size']=quotes['ask_size']=1000;quotes['timestamp']=m['starts'][:,None]-1_000_000
    return m,base,quotes


def test_asymmetric_closing_stays_neutral_and_flattens():
    m,base,quotes=example();quotes['ask_size'][-5:-3,1]=1
    path,orders,failures=replay(m,base,np.ones(1),Execution(),quotes)
    assert failures==0 and path[-1,3]==0
    assert abs(path[:,4]).max()<=.0025
    assert path[-5,3]>0
    assert verify(m,path,orders,Execution(),quotes)['passed']


def test_unavailable_entry_expires_after_three_attempts():
    m,base,quotes=example();quotes['bid'][62:65]=np.nan
    path,orders,failures=replay(m,base,np.ones(1),Execution(),quotes)
    assert failures==0 and not orders[:,5].any()
    assert set(orders[:,0])=={62,63,64}
    assert path[-1,1]==100000


def test_unavailable_closing_is_still_invalid():
    m,base,quotes=example();quotes['bid'][-5:]=np.nan
    path,orders,failures=replay(m,base,np.ones(1),Execution(),quotes)
    assert failures>0 and path[-1,16]>0


def test_higher_cost_for_identical_flat_market_fills_reduces_nav():
    m,base,quotes=example();s=Execution()
    a,ao,_=replay(m,base,np.ones(1),s,quotes)
    b,bo,_=replay(m,base,np.ones(1),replace(s,adverse_bps=.5),quotes)
    np.testing.assert_array_equal(ao[:,5],bo[:,5])
    assert b[-1,1]<a[-1,1]


def test_future_data_cannot_change_past_fills():
    m,base,quotes=example();a,ao,_=replay(m,base,np.ones(1),Execution(),quotes)
    m['open'][75:]*=2;m['close'][75:]*=2;quotes['bid'][75:]*=2;quotes['ask'][75:]*=2
    b,bo,_=replay(m,base,np.ones(1),Execution(),quotes)
    np.testing.assert_array_equal(a[:75],b[:75])
    np.testing.assert_array_equal(ao[ao[:,0]<75],bo[bo[:,0]<75])


def test_filter_cannot_add_non_candidates_or_use_future_volume():
    m,base,quotes=example();m['close'][59]=[103,97,101,99,102,98];m['close'][60]=[104,96,102,98,103,97]
    fitted=dict(sd=np.full(6,.001),liquid=np.ones(6,bool),early_volume=np.full(6,4e7),beta=np.array([.01,0.]))
    config=next(c for c in CATALOG if c['id']=='participation_volume')
    a,decisions=build_stock(m,base,quotes,fitted,config,Execution())
    assert a[60,0]>0 and a[60,1]<0 and not a[:,2:].any()
    m['volume'][61:]*=1000;m['close'][61:]*=10
    b,_=build_stock(m,base,quotes,fitted,config,Execution())
    np.testing.assert_array_equal(a,b)


def test_cost_hurdle_rejects_unfunded_edge_and_future_quote():
    m,base,quotes=example();m['close'][59]=[103,97,101,99,102,98];m['close'][60]=[104,96,102,98,103,97]
    fitted=dict(sd=np.full(6,.001),liquid=np.ones(6,bool),early_volume=np.full(6,4e7),beta=np.array([0.,0.]))
    config=next(c for c in CATALOG if c['id']=='concentrated_3_cost')
    a,_=build_stock(m,base,quotes,fitted,config,Execution());assert not a.any()
    fitted['beta'][0]=.01
    b,_=build_stock(m,base,quotes,fitted,config,Execution());assert b.any()
    quotes['timestamp'][61]=m['completed'][60]+1
    c,_=build_stock(m,base,quotes,fitted,config,Execution());assert not c.any()


def test_regression_fit_excludes_later_sessions(tmp_path):
    m,base,_=example();n=6;days=61;length=len(base)
    for key in ['open','close','volume','vol','eligible_full']:
        m[key]=np.tile(m[key],(days,1))
    m['sessions']=np.repeat(np.arange(days),length);m['meta']={'session_ids':list(range(days))}
    base=np.tile(base,(days,1));reference=tmp_path/'reference';reference.mkdir()
    np.savez(reference/'fit.npz',S10_60_0_None_0_0_sd=np.full(n,.001))
    a=fit(m,base,reference,tmp_path/'a')
    m['volume'][-length:]*=1000;m['open'][-length:]*=10;m['close'][-length:]*=20
    b=fit(m,base,reference,tmp_path/'b')
    for k in a:np.testing.assert_array_equal(a[k],b[k])


def test_market_etf_pair_is_dollar_neutral_and_causal():
    m,_,_=example();m['close'][59,:2]=[101.,102.];m['close'][60,:2]=[102.,104.]
    config=next(c for c in CATALOG if c['id']=='etf_market_pair')
    a,decisions=build_etf(m,config)
    assert a[60,0]==-.04 and a[60,1]==.04 and a[60].sum()==0
    m['close'][61:]=1
    b,_=build_etf(m,config);np.testing.assert_array_equal(a,b)


def test_selection_resume_preserves_frozen_development_decision(tmp_path):
    from experiments.closing_followup.runner import select
    rows=[]
    for c in CATALOG:
        for scope,labels in [('earlier_OOF',['a','b','c']),('initial_validation',['d','e'])]:
            for scenario in ['primary','adverse_0p5']:
                rows.append(dict(config=c,scope=scope,scenario=scenario,
                    metrics=dict(accounting_valid=True,coverage_passed=True,total_return=.001,active_sessions=20),
                    blocks={k:dict(Sharpe=1.1) for k in labels}))
    a=select(rows,tmp_path);b=select(rows,tmp_path)
    assert a==b and len(a['shortlist'])==len(CATALOG)


def test_etf_bar_cache_tampering_is_rejected(tmp_path):
    from experiments.closing_followup.data import load_etfs
    from experiments.closing_followup.config import ETFS
    from experiments.strategy_search.storage import atomic,read
    m,_,_=example();stamp=str(np.datetime64(int(m['starts'][0]),'ns').astype('datetime64[D]'))
    p=tmp_path/'etf_cache'/'bars'/(stamp+'.json')
    atomic(p,dict(bars={s:[dict(t=str(np.datetime64(int(m['starts'][i]),'ns'))+'Z',o=100.,c=100.,v=1000.) for i in range(len(m['starts']))] for s in ETFS}))
    load_etfs(m,tmp_path)
    d=read(p);d['bars']['SPY'][0]['o']=101.;atomic(p,d)
    with pytest.raises(ValueError,match='Changed cached ETF bars'):load_etfs(m,tmp_path)
