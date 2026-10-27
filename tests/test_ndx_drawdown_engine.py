import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pit_research_engine import continuous_backtest
from ndx_drawdown_engine import issuer_weights, capped_weights, risk_inputs, exposure_series, simulate


class DrawdownEngineTests(unittest.TestCase):
    def test_issuer_selection_is_invariant_to_share_class_order_and_ties(self):
        names = ['GOOG','GOOGL','A','B','C','D']
        score = np.array([1.,1.,.9,.8,.7,.6])
        mask = np.ones(6,dtype=bool)
        actual = issuer_weights(score,mask,names,3)
        np.testing.assert_allclose(actual,[1/6,1/6,1/3,1/3,0,0])
        permutation = np.array([3,1,5,0,2,4])
        other = issuer_weights(score[permutation],mask,[names[j] for j in permutation],3)
        np.testing.assert_allclose(other,actual[permutation])

    def test_weight_cap_redistributes_excess_without_leverage(self):
        weights = capped_weights([100,1,1,1,1],.3)
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertLessEqual(weights.max(),.3+1e-12)
        np.testing.assert_allclose(weights,[.3,.175,.175,.175,.175])

    def test_volatility_estimates_cannot_see_future_returns(self):
        rng = np.random.default_rng(19)
        returns = rng.normal(0,.02,(120,5))
        weights = np.full_like(returns,.2)
        first = risk_inputs({'returns':returns},weights)
        altered = returns.copy(); altered[81:] = 10
        second = risk_inputs({'returns':altered},weights)
        for key in first:
            np.testing.assert_array_equal(first[key][:81],second[key][:81])
        spec = dict(vol='basket',window=20,target=.2)
        market = {'qqq':np.ones(120)}
        np.testing.assert_array_equal(exposure_series(spec,market,first)[:81],exposure_series(spec,market,second)[:81])

    def test_frozen_orders_costs_settlement_and_cash_match_parent(self):
        dates = pd.bdate_range('2022-01-03',periods=14)
        rng = np.random.default_rng(7)
        marks = 20*np.exp(np.cumsum(rng.normal(.002,.03,(14,4)),axis=0))
        tradable = np.ones_like(marks,dtype=bool);tradable[8:,0]=False
        data = dict(panel=SimpleNamespace(dates=dates,tickers=['A','B','C','D']),
                    marks=marks,tradable=tradable,adv=np.full_like(marks,4e6),
                    origin=np.array([0,5,10]),entry=np.array([1,6,11]),
                    events=[dict(date=str(dates[8].date()),security_id='A',stock_fraction_at_completion=.5)],
                    mask=np.array([[1,1,1,1],[1,1,1,1],[0,1,1,1]],dtype=bool))
        scores=np.array([[4.,3.,2.,1.],[3.,4.,2.,1.],[0.,1.,4.,3.]])
        ids=np.arange(3);weekly=np.array([[.5,.5,0,0],[.5,.5,0,0],[0,0,.5,.5]])
        expected,curve,ledger=continuous_backtest(data,ids,scores,2,1.)
        for cadence in ['weekly','daily']:
            actual,path,detail=simulate(data,ids,weekly,np.ones(14),cadence,detailed=True)
            np.testing.assert_allclose(path,curve.net,atol=1e-14,rtol=1e-14)
            np.testing.assert_allclose(detail.nav,ledger.nav,atol=1e-8)
            for key in ['sharpe','maximum_drawdown','cost_dollars','turnover_annual']:
                self.assertAlmostEqual(actual[key],expected[key],places=9)

    def test_exposure_signal_fills_next_close_and_does_not_rebalance_early(self):
        dates=pd.bdate_range('2022-01-03',periods=5)
        marks=np.array([[100.],[100.],[120.],[120.],[120.]])
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A']),marks=marks,
                  tradable=np.ones((5,1),dtype=bool),adv=np.full((5,1),1e9),
                  origin=np.array([0]),entry=np.array([1]),events=[])
        exposure=np.array([1.,0.,0.,0.,0.])
        _,path,detail=simulate(data,np.array([0]),np.ones((1,1)),exposure,'daily',detailed=True)
        self.assertGreater(path[1],.19) # Rise occurs before the exit fill, not after.
        self.assertAlmostEqual(detail.iloc[1].equity_exposure,0.)
        self.assertEqual(detail.iloc[2].traded,0.)


if __name__ == '__main__':
    unittest.main()
