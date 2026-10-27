"""Causality, training isolation, mandatory exit and selection safeguards."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('numba')
from experiments.longer_horizon.signals import sampled_features,build,fit
from experiments.longer_horizon.engine import replay
from experiments.longer_horizon.runner import select
from experiments.longer_horizon.config import CATALOG,proxy_settings
from experiments.longer_horizon.auction import closing_print,decisions
from experiments.closing_followup.runner import verify
from experiments.strategy_search.storage import atomic,sha,read
from experiments.strategy_validation.config import Execution
from test.test_closing_followup import example


def feature_market(days=1):
    rng=np.random.default_rng(77);t=390*days;n=6
    raw=rng.normal(0,.0005,(t,n));raw[56:61,0]+=.01;raw[56:61,1]-=.01
    m=dict(raw=raw,valid=np.ones((t,n),bool),sessions=np.repeat(np.arange(days),390),
        meta={'session_ids':list(range(days))},open=np.full((t,n),100.),vol=np.full((t,n),.001),
        eligible_full=np.ones((t,n),bool),industry=np.array([0,0,0,1,1,1]),sector=np.zeros(n,int),
        completed=np.arange(t,dtype=np.int64)*60_000_000_000,symbols=list('ABCDEF'))
    w=np.zeros((n,n));w[2,0]=1;w[3,1]=1
    f=dict(sd=np.full(n,.001),direct=w,two_hop=w.copy(),shuffled_two_hop=w[::-1].copy())
    return m,f


def test_directed_follower_is_bought_after_completed_leader_shock():
    m,f=feature_market();features=sampled_features(m,f)
    base,chosen=build(m,features,dict(family='direct',hold=60))
    assert chosen[0]['origin']==60
    assert base[60,2]>0 and base[60,3]<0 and not base[:,[0,1,4,5]].any()
    assert not base[:60].any() and not base[120:].any()


def test_future_bars_cannot_change_existing_decision():
    m,f=feature_market();a,da=build(m,sampled_features(m,f),dict(family='direct',hold=120))
    m['raw'][61:]*=1000;m['vol'][61:]*=100;m['eligible_full'][61:]=False
    b,db=build(m,sampled_features(m,f),dict(family='direct',hold=120))
    np.testing.assert_array_equal(a,b);assert da==db


def test_normalizer_fit_never_reads_evaluation_outcomes(tmp_path):
    m,f=feature_market(61);g=tmp_path/'graph';g.mkdir()
    atomic(g/'network_fit.json',dict(details={'training_sessions':list(range(60))}))
    np.savez(g/'networks.npz',**{'5_direct':f['direct'],'5_chain':f['two_hop'],'5_shuffled_chain':f['shuffled_two_hop']})
    a=fit(m,g,tmp_path/'a');m['raw'][-390:]*=1e5
    b=fit(m,g,tmp_path/'b')
    for k in a:np.testing.assert_array_equal(a[k],b[k])


def test_mandatory_scheduled_exit_retries_after_three_missing_quotes():
    m,base,q=example();base[72:]=0.;q['bid'][74:79]=np.nan
    path,orders,failed=replay(m,base,np.ones(1),Execution(),q)
    assert failed==0 and not path[-1,3]
    assert (orders[(orders[:,0]==79),5]!=0).any()
    assert not (orders[(orders[:,0]>=80),5]!=0).any()
    assert verify(m,path,orders,Execution(),q)['passed']


def test_missing_discretionary_entry_still_expires():
    m,base,q=example();q['bid'][62:65]=np.nan
    path,orders,failed=replay(m,base,np.ones(1),Execution(),q)
    assert failed==0 and set(orders[:,0])=={62,63,64} and not orders[:,5].any()


def test_quote_age_sensitivity_cannot_change_primary_default():
    m,base,q=example();q['timestamp']=m['starts'][:,None]-10_000_000_000
    _,orders,_=replay(m,base,np.ones(1),Execution(),q)
    _,older,_=replay(m,base,np.ones(1),replace(Execution(),quote_max_age_seconds=60),q)
    assert not orders[:,5].any() and older[:,5].any() and Execution().quote_max_age_seconds==5


def test_development_selection_ignores_later_scores_and_resumes(tmp_path):
    rows=[]
    for c in CATALOG:
        for scope,labels in [('earlier_OOF',['a','b','c']),('initial_validation',['d','e'])]:
            rows.append(dict(config=c,scope=scope,scenario='cost1',mode='minute_proxy',
                metrics=dict(accounting_valid=True,total_return=.001,active_sessions=20),
                blocks={label:dict(Sharpe=1.1) for label in labels}))
    a=select(rows,tmp_path)
    rows.append(dict(config=CATALOG[-1],scope='later_development64',scenario='cost1',mode='minute_proxy',metrics={'Sharpe':999}))
    b=select(rows,tmp_path)
    assert a==b and a['shortlist']==['direct_60','self_reversal_60']


def test_auction_tie_disagreement_is_not_silently_resolved():
    records=[dict(d='2025-04-01',c=[dict(c='6',p=100,s=100,t='a'),dict(c='6',p=101,s=100,t='b')])]
    assert closing_print(records,'2025-04-01') is None
    records[0]['c'][1]['s']=99
    assert closing_print(records,'2025-04-01')['p']==100


def test_auction_price_availability_cannot_select_a_stock(tmp_path):
    m,f=feature_market();m['starts']=np.datetime64('2025-04-01T13:30','ns').astype(np.int64)+m['completed']
    m['close']=m['open'].copy();m['close'][59]=[103,97,102,98,101,99]
    p=tmp_path/'auctions'/'2025-04-01.json';atomic(p,dict(auctions={}))
    atomic(p.parent/'manifest.json',dict(files={p.name:sha(p)}))
    a=decisions(m,'early_momentum_auction',tmp_path)
    atomic(p,dict(auctions={'A':[dict(d='2025-04-01',c=[dict(c='6',p=900,s=100,t='2025-04-01T20:00:00Z')])]}))
    atomic(p.parent/'manifest.json',dict(files={p.name:sha(p)}))
    b=decisions(m,'early_momentum_auction',tmp_path)
    assert a[0]['weights']==b[0]['weights'] and a[0]['selected']==b[0]['selected']


def test_quote_cache_detects_changed_source_snapshot(tmp_path):
    from experiments.longer_horizon.data import QuoteCache
    m,_,_=example();old=tmp_path/'old'/'quotes'/m['scope'];atomic(old/'manifest.json',dict(files={}))
    root=tmp_path/'new'/'quotes'/m['scope'];p=root/(str(m['starts'][0])+'_A.json')
    atomic(p,dict(arrival_ns=int(m['starts'][0]),symbol='A',quote=None))
    atomic(root/'manifest.json',dict(files={p.name:sha(p)}))
    d=read(p);d['quote']=[99,100,10,10,int(m['starts'][0])];atomic(p,d)
    with pytest.raises(ValueError,match='Changed execution quote cache'):QuoteCache(m,tmp_path/'new',tmp_path/'old'/'validated','.env')


def test_zero_execution_proxy_retains_short_borrow():
    s=proxy_settings(0)
    assert s.borrow==.03 and s.commission_free and s.adverse_bps==0
    assert not any([s.exchange_taker_share,s.clearing_share,s.cat_share,s.sec_sell_fraction,s.finra_sell_share])
