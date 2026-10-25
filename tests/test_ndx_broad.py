"""Shared-model generalization, actual-feature causality and risk-account parity."""
import copy
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from ndx_broad_models import SharedNetwork,prepare,normalize,splits,stability
from ndx_policy_engine import policy_backtest,policy_replay
from pit_research_engine import load_data,continuous_backtest
from run_ndx_alpaca_training import feature_data
from test_pit_research_engine import fixture


class SharedModelTests(unittest.TestCase):
    def test_permutation_equivariance_and_variable_security_count(self):
        torch.manual_seed(7); model = SharedNetwork(8).eval()
        x = torch.randn(2,7,8); mask = torch.ones(2,7,dtype=torch.bool)
        permutation = torch.tensor([6,2,4,1,0,5,3])
        with torch.no_grad():
            original = model(x,mask)
            changed = model(x[:,permutation],mask[:,permutation])
            np.testing.assert_allclose(changed,original[:,permutation],atol=1e-6,rtol=0)
            self.assertEqual(tuple(model(torch.randn(2,11,8),torch.ones(2,11,dtype=torch.bool)).shape),(2,11))

    def test_inactive_nan_nodes_do_not_influence_eligible_outputs(self):
        torch.manual_seed(19); model = SharedNetwork(8).eval()
        x = torch.randn(2,5,8); mask = torch.ones(2,5,dtype=torch.bool); mask[:,-1]=False
        with torch.no_grad():
            first = model(x,mask)
            x[:,-1]=float('nan'); second = model(x,mask)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(second[:,-1],np.zeros(2))

    def test_future_and_inactive_observations_do_not_change_normalization(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(20,4,6))
        mask = np.ones((20,4),bool); mask[:,3]=False; x[:,:,5]=np.nan
        first = normalize(x,mask,np.arange(10))
        x[10:]=1e10; x[:,3]=1e20
        second = normalize(x,mask,np.arange(10))
        np.testing.assert_array_equal(first[1],second[1]); np.testing.assert_array_equal(first[2],second[2])
        self.assertTrue(np.isfinite(first[0]).all())


class RiskAccountTests(unittest.TestCase):
    def test_full_exposure_is_identical_to_frozen_parent(self):
        data = fixture(); data['marks'][3:,0]=125.
        ids = np.arange(5); scores = np.tile([2.,1.,0.],(5,1))
        _,before,_ = continuous_backtest(data,ids,scores,k=1,cost_multiplier=2.)
        _,after,_ = policy_backtest(data,ids,scores,k=1,cost_multiplier=2.,exposure=np.ones(5))
        np.testing.assert_array_equal(before.net,after.net)

    def test_half_exposure_has_known_half_sized_price_gain(self):
        data = fixture(); data['marks'][3:,0]=125.
        ids = np.arange(5); scores = np.tile([2.,1.,0.],(5,1))
        metric,_,_ = policy_backtest(data,ids,scores,k=1,cost_multiplier=0.,exposure=np.full(5,.5))
        self.assertAlmostEqual(metric['terminal_nav'],1125000.)

    def test_dynamic_exposure_reconciles_independent_account(self):
        data = fixture(); data['marks'][3:,0]=125.; data['marks'][12:,1]=80.
        ids = np.arange(5); scores = np.tile([2.,1.,0.],(5,1)); exposure = np.array([1.,.5,0.,1.,.5])
        _,daily,ledger = policy_backtest(data,ids,scores,k=1,cost_multiplier=4.,exposure=exposure)
        independent = policy_replay(data,ids,scores,1,4.,exposure=exposure)
        np.testing.assert_allclose(independent,1e6*np.cumprod(1+daily.net),atol=1e-6,rtol=1e-12)
        self.assertGreaterEqual(ledger.cash.min(),-1e-5)


class RealFeatureCausalityTests(unittest.TestCase):
    def test_future_input_perturbation_cannot_change_past_features_or_mature_labels(self):
        base = load_data(); root = Path(base['manifest']['root'])
        dv = pd.read_csv(root/'dollar_volume.csv',index_col=0).to_numpy().copy()
        q = pd.read_csv(root/'etf_adj_close.csv',index_col=0)['QQQ'].to_numpy().copy()
        intraday = feature_data()[0]
        first = prepare(base=base,dollar_volume=dv,benchmark=q,intraday=intraday)
        changed = copy.deepcopy(base); stop = 1400
        changed['marks'][stop+1:]*=np.linspace(2,10,len(base['panel'].tickers))
        changed['panel'].close[stop+1:]*=np.linspace(2,10,len(base['panel'].tickers))
        dv[stop+1:]*=100.; q[stop+1:]*=.5; intraday[stop+1:]*=100.
        second = prepare(base=changed,dollar_volume=dv,benchmark=q,intraday=intraday,verify_parent_mask=False)
        known = first['origin']<=stop
        for view in ['daily','intraday']:
            np.testing.assert_array_equal(first['views'][view][known],second['views'][view][known])
        matured = first['exit']<=stop
        for target in ['raw','residual']:
            np.testing.assert_array_equal(first['targets'][target][matured],second['targets'][target][matured])
        for year in [2022,2023,2024,2025]:
            train,val,test = splits(first,year)
            self.assertGreater(len(train),300)
            self.assertLess(first['exit'][train].max(),first['origin'][val].min())
            self.assertLess(first['exit'][val].max(),first['origin'][test].min())


if __name__=='__main__':
    unittest.main()
