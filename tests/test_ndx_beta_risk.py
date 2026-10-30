import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ndx_beta_risk import trailing_beta,risk_targets


class BetaRiskTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(7);qret=rng.normal(0,.01,260)
        q=100*np.cumprod(1+qret)
        returns=np.column_stack((.5*qret,1.5*qret))
        data=dict(returns=returns,origin=np.array([150,200]),panel=SimpleNamespace(dates=pd.bdate_range('2020-01-01',periods=260)))
        market=dict(qqq=q,qqqvol20=np.full(260,.2));w=np.array([[.4,.6],[.4,.6]])
        return data,market,w

    def test_beta_and_target_neutrality_use_known_past_basket(self):
        data,market,w=self.fixture();beta,count=trailing_beta(data,np.array([0,1]),w,market['qqq'],60)
        np.testing.assert_allclose(beta,1.1,rtol=0,atol=1e-12);np.testing.assert_array_equal(count,60)
        target,_=risk_targets(data,np.array([0,1]),w,market,'beta60')
        np.testing.assert_allclose(target[:,:-1].sum(1),1/2.1)
        np.testing.assert_allclose(target[:,:-1]@np.array([.5,1.5])+target[:,-1],0,atol=1e-12)

    def test_future_prices_and_returns_cannot_change_prior_decision(self):
        data,market,w=self.fixture();a,_=risk_targets(data,np.array([0,1]),w,market,'beta60')
        data['returns'][151:]*=3;market['qqq'][151:]*=2
        b,_=risk_targets(data,np.array([0,1]),w,market,'beta60')
        np.testing.assert_array_equal(a[0],b[0])

    def test_short_volatility_budget_and_missing_history_take_priority(self):
        data,market,w=self.fixture();market['qqqvol20'][:]=.8
        target,_=risk_targets(data,np.array([0,1]),w,market,'beta60')
        np.testing.assert_allclose(target[:,-1],-.25)
        self.assertLessEqual(np.abs(target).sum(1).max(),1.)
        data['returns'][:]=np.nan
        beta,count=trailing_beta(data,np.array([0,1]),w,market['qqq'],120)
        np.testing.assert_array_equal(beta,1.);np.testing.assert_array_equal(count,0)


if __name__=='__main__':unittest.main()
