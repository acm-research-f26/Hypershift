"""D09 fixed-stream engineering experiment, independent of alpha selection."""
from pathlib import Path
import numpy as np
from experiments.strategy_validation.config import Execution
from experiments.longer_horizon.engine import replay as legacy
from experiments.strategy_search.storage import arrays,atomic,utc
from .replay import run,verify


def fixture():
    t=390;n=2;start=int(np.datetime64('2025-01-02T14:30','ns').astype(np.int64))
    return dict(open=np.full((t,n),100.),close=np.full((t,n),100.),volume=np.full((t,n),100000.),
        sessions=np.zeros(t,int),starts=start+np.arange(t)*60_000_000_000,completed=start+(np.arange(t)+1)*60_000_000_000,
        sector=np.zeros(n,int),industry=np.zeros(n,int),symbols=['A','B'],actions=np.zeros((t,n),bool),
        vol=np.full((t,n),.001),meta=dict(session_ids=[0]),evaluate_ids=[0],scope='fixture')


def fixed_stream(output):
    output=Path(output);m=fixture();s=Execution();base=np.zeros_like(m['open']);base[60:90]=[.0399,-.0399]
    old,orders,failures=legacy(m,base,np.ones(1),s,None)
    q=np.zeros(2);cash=s.capital;old_max=0.;old_events=0
    for o in orders:
        if not o[5]:continue
        j=int(o[1]);q[j]+=o[5];cash-=o[5]*o[7]+sum(o[10:14]);nav=cash+q@np.full(2,100.)
        net=abs(q.sum()*100/nav);old_max=max(old_max,net);old_events+=int(net>.0025)
    c=[dict(origin=60,stock=0,hedges=[1],side=1,prediction=.001,edge_mean=.001,edge_se=0,
        relative=.01,horizon=30,gate='no_gate',design='D01')]
    new=run(m,c,s,np.ones(1),np.full(2,2.));proof=verify(m,new,s)
    arrays(output/'fixed_stream.npz',legacy_path=old,legacy_orders=orders,new_path=new['path'],new_holdings=new['holdings'])
    atomic(output/'fixed_stream.json',dict(created_UTC=utc(),candidate_origin=60,legacy_reported_failures=failures,
        legacy_reconstructed_sequential_max_net=old_max,legacy_sequential_hard_net_events=old_events,
        new_sequential_max_net=max((abs(x['post_net']) for x in new['fills']),default=0),
        new_verification=proof,new_orders=new['orders'],new_fills=new['fills'],
        note='Identical economic pair signal and clocks; sizing and sequencing are the engineering treatment, not optimized alpha. Legacy combined-batch risk can hide temporary exposure.'))
    return proof
