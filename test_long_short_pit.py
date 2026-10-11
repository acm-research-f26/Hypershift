import unittest
import numpy as np
from long_short_pit import signed_backtest, signed_weights
from audit_ndx_long_short import replay
from pit_research_engine import continuous_backtest
from test_pit_research_engine import fixture


class SignedAccountTests(unittest.TestCase):
    def compare(self,data,k=1,long=.5,short=.5,borrow=0.,cost=0.,ids=None):
        ids=np.array([0]) if ids is None else np.asarray(ids)
        scores=np.tile([2.,1.,np.nan],(len(ids),1))
        metrics,daily,account=signed_backtest(data,ids,scores,k,long,short,borrow,cost)
        independent=replay(data,ids,scores,k,long,short,borrow,cost)
        np.testing.assert_allclose(account[['nav','cash','fee','borrow_fee']],independent,atol=1e-7,rtol=1e-12)
        return metrics,daily,account

    def test_short_drop_profit_and_rise_loss(self):
        for price,expected in [(90.,1.05e6),(125.,.875e6)]:
            data=fixture(); data['marks'][2:,1]=price
            result,_,_=self.compare(data)
            self.assertAlmostEqual(result['terminal_nav'],expected)

    def test_equal_moves_cancel_market_exposure(self):
        data=fixture(); data['marks'][2:,:2]=125.
        result,_,_=self.compare(data)
        self.assertAlmostEqual(result['terminal_nav'],1e6)

    def test_calendar_borrow_includes_weekends(self):
        data=fixture(); result,_,account=self.compare(data,borrow=.03)
        elapsed=(data['panel'].dates[-1]-data['panel'].dates[1]).days
        expected=500000*.03*elapsed/365
        self.assertAlmostEqual(result['borrow_dollars'],expected)
        self.assertAlmostEqual(result['terminal_nav'],1e6-expected)
        self.assertEqual(account.short_dollars.iloc[-1],0.)

    def test_signed_merger_obligation_and_positive_stock_fee(self):
        data=fixture(); data['marks'][3:,1]=125.; data['tradable'][3:,1]=False
        data['events']=[dict(security_id='OTHER',date=str(data['panel'].dates[3].date()),stock_fraction_at_completion=.6)]
        result,_,account=self.compare(data,cost=1.)
        self.assertLess(result['terminal_nav'],875000)
        self.assertLess(account.settled_dollars.sum(),0.)
        self.assertGreater(account.fee.iloc[2],0.)

    def test_unchanged_long_only_execution(self):
        data=fixture(); data['marks'][2:,0]=110.; data['marks'][7:,1]=90.
        ids=np.arange(5); scores=np.tile([2.,1.,0.],(5,1))
        _,daily,_=signed_backtest(data,ids,scores,1,1,0,0,1)
        _,original,_=continuous_backtest(data,ids,scores,1,1)
        np.testing.assert_allclose(daily.net,original.net,atol=1e-12)

    def test_inactive_nan_and_boundary_tie_netting(self):
        result=signed_weights(np.array([3.,2.,2.,0.,np.nan]),np.array([1,1,1,1,0],bool),2,.5,.5)
        np.testing.assert_allclose(result,[.25,0,0,-.25,0])
        self.assertEqual(result.sum(),0.)

    def test_insufficient_universe_remains_cash_and_missing_short_price_fails(self):
        np.testing.assert_array_equal(signed_weights(np.array([2.,1.]),np.ones(2,bool),2,.5,.5),[0,0])
        data=fixture(); data['tradable'][3,1]=False
        with self.assertRaisesRegex(ValueError,'Missing held price'):
            self.compare(data)

    def test_orders_fixed_before_next_close_price_gap(self):
        data=fixture(); data['marks'][1:,1]=125.
        _,_,account=self.compare(data)
        self.assertEqual(account.short_dollars.iloc[0],625000)
        self.assertEqual(account.long_dollars.iloc[0],500000)
        self.assertEqual(account.nav.iloc[0],1e6)


if __name__=='__main__':
    unittest.main()
