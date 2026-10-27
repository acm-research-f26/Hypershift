"""Future observations and inactive issuers cannot alter fitted preprocessing."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pit_research_engine import fold,historical_groups
from run_robust_research import scale_inputs


class TimingTests(unittest.TestCase):
    def test_future_features_and_inactive_nodes_do_not_change_scaling(self):
        rng=np.random.default_rng(7)
        d=dict(x=rng.normal(size=(20,16,4,5)).astype('float32'),mask=np.ones((20,4),bool))
        d['mask'][:10,3]=False;train=np.arange(10)
        _,mean,std=scale_inputs(d,train)
        changed=dict(d,x=d['x'].copy())
        changed['x'][10:]=1e12;changed['x'][:10,:,3]=np.nan
        _,other_mean,other_std=scale_inputs(changed,train)
        np.testing.assert_array_equal(mean,other_mean);np.testing.assert_array_equal(std,other_std)

    def test_future_returns_and_inactive_nodes_do_not_change_graph(self):
        rng=np.random.default_rng(19)
        d=dict(panel=SimpleNamespace(tickers=['A','B','FUTURE']),origin=np.arange(0,100,5),
               returns=rng.normal(size=(100,3)),membership=np.ones((100,3),bool),tradable=np.ones((100,3),bool))
        d['membership'][:60,2]=False;train=np.arange(12)
        original=historical_groups(d,train)
        changed=dict(d,returns=d['returns'].copy())
        changed['returns'][56:]=1e9;changed['returns'][:56,2]=np.nan
        self.assertEqual(original,historical_groups(changed,train))
        self.assertEqual(original[2],{2})

    def test_fold_outcomes_mature_before_validation_and_test_signals(self):
        dates=pd.bdate_range('2018-01-01','2025-12-31')
        origin=np.arange(260,len(dates)-6,5)
        d=dict(panel=SimpleNamespace(dates=dates),origin=origin,entry=origin+1,exit=origin+6)
        for year in [2022,2023,2024,2025]:
            train,val,test=fold(d,year)
            self.assertLess(dates[d['exit'][train]].max(),dates[d['origin'][val]].min())
            self.assertLess(dates[d['exit'][val]].max(),dates[d['origin'][test]].min())
            self.assertFalse(set(train)&set(val));self.assertFalse(set(val)&set(test))


if __name__=='__main__':unittest.main()
