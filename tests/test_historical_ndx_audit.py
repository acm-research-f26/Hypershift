"""Check the independent auditor against known accounts before reading results."""
import unittest
import numpy as np
from audit_historical_ndx import replay
from pit_research_engine import continuous_backtest
from test_pit_research_engine import fixture


class IndependentLedgerTests(unittest.TestCase):
    def compare(self, data, scores, k=1, cost=0., basket=False):
        ids=np.arange(5)
        _, daily, _=continuous_backtest(data,ids,scores,k,cost,basket=basket)
        independent=replay(data,ids,scores,k,cost,basket)
        np.testing.assert_allclose(independent,1e6*np.cumprod(1+daily.net),rtol=1e-12,atol=1e-7)
        return independent

    def test_cash_merger_known_terminal_value(self):
        data=fixture();day=3
        data['marks'][day:,0]=125.;data['tradable'][day:,0]=False
        data['mask'][1:,0]=False
        data['events']=[dict(security_id='OLD',date=str(data['panel'].dates[day].date()),stock_fraction_at_completion=0.)]
        result=self.compare(data,np.tile([2.,1.,0.],(5,1)))
        self.assertAlmostEqual(result[-1],1.25e6)

    def test_ties_split_capital_and_holdings_rebalance_after_price_moves(self):
        data=fixture();data['marks'][2:,0]=110.;data['marks'][7:,1]=90.
        result=self.compare(data,np.zeros((5,3)),cost=1.)
        self.assertTrue(np.isfinite(result).all())

    def test_stock_merger_fees_and_next_fill(self):
        data=fixture();day=3
        data['marks'][day:,0]=125.;data['tradable'][day:,0]=False
        data['mask'][1:,0]=False
        data['events']=[dict(security_id='OLD',date=str(data['panel'].dates[day].date()),stock_fraction_at_completion=.6)]
        result=self.compare(data,np.tile([2.,1.,0.],(5,1)),cost=4.)
        self.assertLess(result[-1],1.25e6)

    def test_liquidity_caps_and_monthly_basket(self):
        data=fixture();data['adv'][:]=2e6
        data['marks'][10:,0]=80.;data['marks'][12:,1]=130.
        self.compare(data,np.tile([2.,1.,0.],(5,1)),cost=2.)
        self.compare(data,np.zeros((5,3)),cost=2.,basket=True)


if __name__=='__main__':unittest.main()
