import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rishi_account_trace import traced_backtest


class AccountTraceTests(unittest.TestCase):
    def test_trace_reconstructs_shares_cash_fees_and_return_contributions(self):
        dates=pd.bdate_range('2022-01-03',periods=5)
        marks=np.array([[100.,100.],[102.,99.],[101.,98.],[103.,101.],[98.,104.]])
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A','B']),marks=marks,
                  tradable=np.ones((5,2),bool),adv=np.full((5,2),1e9),events=[],
                  origin=np.array([0]),entry=np.array([1]),mask=np.ones((1,2),bool))
        m,c,l,trace=traced_backtest(data,np.array([0]),np.array([[2.,1.]]),1)
        np.testing.assert_allclose(trace['net_contribution'].sum(1),c.net,atol=1e-12)
        self.assertTrue(np.all(trace['units'][-1]==0))
        self.assertGreater(len(trace['orders']),0)
        self.assertLess(trace['cash_reconciliation_error'],1e-6)
        self.assertLess(trace['share_reconciliation_error'],1e-8)

    def test_settlement_and_missing_fill_reconcile(self):
        dates=pd.bdate_range('2022-01-03',periods=6)
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A','B']),
            marks=np.array([[100.,100.],[102.,99.],[101.,98.],[103.,101.],[103.,104.],[103.,105.]]),
            tradable=np.ones((6,2),bool),adv=np.full((6,2),1e9),
            events=[dict(date=str(dates[3].date()),security_id='A',stock_fraction_at_completion=.5)],
            origin=np.array([0]),entry=np.array([1]),mask=np.ones((1,2),bool))
        data['tradable'][3:,0]=False
        _,c,_,trace=traced_backtest(data,np.array([0]),np.array([[2.,1.]]),1)
        self.assertEqual(sum(x['kind']=='settlement' for x in trace['orders']),1)
        np.testing.assert_allclose(trace['net_contribution'].sum(1),c.net,atol=1e-12)
        data['events']=[];data['tradable'][1,0]=False
        _,c,_,trace=traced_backtest(data,np.array([0]),np.array([[2.,1.]]),1)
        self.assertEqual(len(trace['orders']),0)
        np.testing.assert_allclose(c.net,0)


if __name__=='__main__':unittest.main()
