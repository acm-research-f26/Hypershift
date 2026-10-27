import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pit_research_engine import continuous_backtest


def fixture():
    dates=pd.bdate_range('2021-12-31',periods=27)
    origins=np.arange(0,25,5)
    return dict(panel=SimpleNamespace(dates=dates,tickers=['OLD','OTHER','FUTURE']),
                origin=origins,entry=origins+1,exit=np.minimum(origins+6,26),
                marks=np.tile([100.,100.,np.nan],(27,1)),adv=np.full((27,3),1e9),
                mask=np.tile([True,True,False],(5,1)),
                membership=np.tile([True,True,False],(27,1)),
                tradable=np.tile([True,True,False],(27,1)),events=[])


class PortfolioTests(unittest.TestCase):
    def test_flat_no_cost_preserves_cash_and_nav_with_nan_future_node(self):
        data=fixture();scores=np.tile([2.,1.,0.],(5,1))
        m,d,a=continuous_backtest(data,np.arange(5),scores,k=1,cost_multiplier=0.)
        np.testing.assert_allclose(d.net,0.,atol=1e-12)
        self.assertAlmostEqual(m['terminal_nav'],1e6)
        self.assertAlmostEqual(a.cash.iloc[-1],1e6)
        self.assertEqual(a.holdings.iloc[-1],0.)

    def test_acquisition_credits_cash_and_does_not_forward_fill_stock(self):
        data=fixture();date=data['panel'].dates[3]
        data['marks'][3:,0]=125.;data['tradable'][3:,0]=False
        data['mask'][1:,0]=False
        data['events']=[dict(security_id='OLD',date=str(date.date()),stock_fraction_at_completion=0.)]
        scores=np.tile([2.,1.,0.],(5,1))
        m,d,a=continuous_backtest(data,np.arange(5),scores,k=1,cost_multiplier=0.)
        self.assertAlmostEqual(m['terminal_nav'],1.25e6)
        self.assertAlmostEqual(a.settled_dollars.sum(),1.25e6)
        self.assertAlmostEqual(a.loc[3-1,'cash'],1.25e6)
        self.assertAlmostEqual(a.loc[3-1,'holdings'],0.)

    def test_unpriced_held_stock_without_action_fails(self):
        data=fixture();data['tradable'][3,0]=False
        with self.assertRaisesRegex(ValueError,'Missing held price'):
            continuous_backtest(data,np.arange(5),np.tile([2.,1.,0.],(5,1)),k=1,cost_multiplier=0.)

    def test_transaction_costs_reduce_nav_without_negative_cash(self):
        data=fixture();scores=np.tile([2.,1.,0.],(5,1))
        m,d,a=continuous_backtest(data,np.arange(5),scores,k=1)
        self.assertLess(m['terminal_nav'],1e6)
        self.assertTrue((a.cash>=-1e-6).all())
        self.assertAlmostEqual(1e6-m['terminal_nav'],a.fee.sum(),places=6)


if __name__=='__main__':unittest.main()
