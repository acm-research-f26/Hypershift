import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rishi_leader_trace import trace_risk,trace_etf


class LeaderTraceTests(unittest.TestCase):
    def test_settlement_and_unavailable_new_fill_reconcile(self):
        dates=pd.bdate_range('2022-01-03',periods=7);marks=np.full((7,2),100.)
        tradable=np.ones_like(marks,bool);tradable[3:,0]=False;tradable[4,1]=False
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A','B']),marks=marks,tradable=tradable,
            adv=np.full_like(marks,1e9),origin=np.array([0,3]),entry=np.array([1,4]),
            events=[dict(date=str(dates[3].date()),security_id='A',stock_fraction_at_completion=.5)])
        _,path,_,trace=trace_risk(data,np.arange(2),np.eye(2),np.ones(7))
        np.testing.assert_allclose(trace['net_contribution'].sum(1),path,atol=1e-12)
        self.assertEqual(sum(o['kind']=='settlement' for o in trace['fills']),1)
        self.assertGreater(sum(o['status']=='missing_fill' for o in trace['fills']),0)
        self.assertEqual(trace['units'][-1].sum(),0)

    def test_submitted_share_order_records_binding_cap(self):
        dates=pd.bdate_range('2022-01-03',periods=4);marks=np.full((4,1),100.)
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A']),marks=marks,
            tradable=np.ones_like(marks,bool),adv=np.full_like(marks,1e4),events=[],
            origin=np.array([0]),entry=np.array([1]))
        _,_,_,trace=trace_risk(data,np.array([0]),np.ones((1,1)),np.ones(4))
        first=trace['submitted'][0]
        self.assertEqual(first['desired_delta'],10000.)
        self.assertEqual(first['frozen_delta'],1.)
        self.assertTrue(first['cap_binding'])
        self.assertEqual(trace['units'][0,0],1.)

    def test_etf_first_and_terminal_fees_roll_into_positions_and_cash(self):
        with tempfile.TemporaryDirectory() as folder:
            dates=pd.bdate_range('2021-12-01','2022-01-06');prices=np.linspace(100,110,len(dates))
            pd.DataFrame({'SPY':prices,'QQQ':prices},index=dates).to_csv(Path(folder)/'etf_adj_close.csv')
            pd.DataFrame({'SPY':np.full(len(dates),1e9),'QQQ':np.full(len(dates),1e9)},index=dates).to_csv(Path(folder)/'etf_dollar_volume.csv')
            data=dict(panel=SimpleNamespace(dates=dates),manifest={'root':folder})
            _,curve,_,trace=trace_etf(data,'SPY')
            np.testing.assert_allclose(trace['net_contribution'].sum(1),curve.net,atol=1e-12)
            self.assertEqual(len(trace['fills']),2)
            self.assertEqual(trace['units'][-1,0],0)
            self.assertGreater(trace['fees'][0,0],0)
            self.assertGreater(trace['fees'][-1,0],0)


if __name__=='__main__':unittest.main()
