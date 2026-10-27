import unittest
import numpy as np
from test_pit_research_engine import fixture
from ndx_drawdown_engine import simulate
from ndx_weak_market_short import targets, weak_signals, simulate_signed


class WeakMarketShortTests(unittest.TestCase):
    def test_signals_and_targets_cannot_see_future(self):
        p = 100+np.sin(np.arange(400)/10)
        q = p.copy(); q[301:] = 10000
        for key in weak_signals(p):
            np.testing.assert_array_equal(weak_signals(p)[key][:301], weak_signals(q)[key][:301])
        w = np.full((2, 2), .5)
        out = targets(w, np.array([.7,.7]), [0,1], [True, False], [.3,.3], 'hedge50')
        np.testing.assert_allclose(out, [[.35,.35,-.3],[.35,.35,0]])
        out = targets(w, [.7,.7], [0,1], [True,False], [.4,.4], 'switch100')
        np.testing.assert_allclose(out, [[0,0,-.5],[.35,.35,0]])

    def test_no_short_parity_with_frozen_long_engine(self):
        data = fixture(); data['marks'][2:,0] = 120
        w = np.tile([.3,.3,0.], (5,1)); ids = np.arange(5)
        _, expected, _ = simulate(data, ids, w/.6, np.full(27,.6))
        _, actual, _ = simulate_signed(data, ids, w)
        np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_short_profit_loss_borrow_weekends_and_terminal_cover(self):
        for price in [90.,125.]:
            data = fixture(); data['marks'][2:,1] = price
            w = np.array([[0.,-.5,0.]])
            m, _, ledger = simulate_signed(data, [0], w, borrow_rate=.03, multiplier=0., detailed=True)
            days1 = (data['panel'].dates[2]-data['panel'].dates[1]).days
            remaining = (data['panel'].dates[-1]-data['panel'].dates[2]).days
            borrow = .03*(500000*days1+5000*price*remaining)/365
            self.assertAlmostEqual(m['terminal_nav'], 1e6+5000*(100-price)-borrow)
            self.assertAlmostEqual(m['borrow_dollars'], borrow)
            self.assertEqual(ledger.short.iloc[-1], 0.)

    def test_signal_fills_following_close_and_orders_are_fixed(self):
        data = fixture(); data['marks'][1:,1] = 120.; data['marks'][2:,1] = 90.
        _, path, ledger = simulate_signed(data, [0], np.array([[0.,-.5,0.]]), 0.,0.,detailed=True)
        self.assertEqual(ledger.short.iloc[0], 600000.)
        self.assertEqual(path[0], 0.)
        self.assertAlmostEqual(path[1], .15)

    def test_signed_settlement_fees_and_missing_marks(self):
        data = fixture(); data['marks'][3:,1] = 125.; data['tradable'][3:,1] = False
        data['events'] = [dict(date=str(data['panel'].dates[3].date()),security_id='OTHER',stock_fraction_at_completion=.6)]
        m, _, ledger = simulate_signed(data, [0], np.array([[0.,-.5,0.]]), 0.,1.,detailed=True)
        self.assertLess(m['terminal_nav'],875000.)
        self.assertGreater(ledger.fee.iloc[2],0.)
        data['events'] = []
        with self.assertRaisesRegex(ValueError,'Unpriced'):
            simulate_signed(data,[0],np.array([[0.,-.5,0.]]))

    def test_cover_cash_scaling_keeps_borrow_funding(self):
        data = fixture()
        w = np.tile([1.,0.,0.],(5,1)); w[0] = [0.,-.25,0.]
        m,_,ledger = simulate_signed(data,np.arange(5),w,.10,4.,detailed=True)
        self.assertGreaterEqual(m['minimum_cash'],0.)
        self.assertGreater(ledger.cash.iloc[5],0.)


if __name__ == '__main__':
    unittest.main()
