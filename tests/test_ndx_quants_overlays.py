import copy
from types import SimpleNamespace
import unittest
import numpy as np

from ndx_drawdown_engine import allocations, daily_allocation, risk_inputs, exposure_series
from ndx_retention import hedged_targets
from ndx_quants_overlays import equal_risk_weights, market_ranks, portfolio_targets, size_issuers


class QuantsOverlayTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(43)
        returns = rng.normal(0, np.linspace(.008, .025, 7), (320, 7))
        data = dict(returns=returns, marks=100*np.cumprod(1+returns, axis=0),
                    tradable=np.ones((320, 7), bool), mask=np.ones((2, 7), bool),
                    origin=np.array([270, 300]), panel=SimpleNamespace(
                        tickers=['GOOG', 'GOOGL', 'A', 'B', 'C', 'D', 'E'], dates=np.arange(320)))
        market = dict(qqq=np.full(320, 100.), ma200=np.full(320, 99.), qqqvol20=np.full(320, .2))
        score = np.tile(np.arange(7), (2, 1)).astype(float)
        return data, market, score

    def test_erc_solves_diagonal_and_correlated_risk_budgets(self):
        np.testing.assert_allclose(equal_risk_weights(np.diag([1., 4., 9.])), np.array([1., .5, 1/3])/ (1+.5+1/3))
        c = np.array([[1., .2, -.1], [.2, 2., .5], [-.1, .5, 3.]])
        w = equal_risk_weights(c); rc = w*(c@w)
        np.testing.assert_allclose(rc/rc.sum(), 1/3, atol=1e-9)

    def test_baseline_fast_risk_matches_inherited_implementation(self):
        data, market, score = self.fixture(); ids = np.arange(2)
        stock = allocations(data, ids, score, 'issuer5')
        risk = risk_inputs(data, daily_allocation(data, ids, stock))
        long = exposure_series(dict(vol='basket', window=20, target=.20), market, risk)[data['origin'][ids]]
        expected = hedged_targets(stock, long, market['qqqvol20'][data['origin'][ids]], .75)
        np.testing.assert_array_equal(portfolio_targets(data, ids, score, market, {}), expected)

    def test_future_market_changes_cannot_change_known_target_or_signal(self):
        data, market, score = self.fixture(); ids = np.arange(2)
        for sizing in ('invvol', 'erc'):
            a = portfolio_targets(data, ids, score, market, dict(sizing=sizing, off=.5))
            r = market_ranks(data, ids, 252)
            changed = copy.deepcopy(data); changed['returns'][271:] *= 5; changed['marks'][271:] *= 2
            other = copy.deepcopy(market); other['qqq'][271:] *= .2
            b = portfolio_targets(changed, ids, score, other, dict(sizing=sizing, off=.5))
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(r[0], market_ranks(changed, ids, 252)[0])

    def test_cap_is_per_issuer_and_unsupported_momentum_is_neutral(self):
        data, market, score = self.fixture(); ids = np.arange(2)
        stock = np.tile([.1, .1, .2, .2, .2, .2, 0], (2, 1))
        for method in ('invvol', 'erc'):
            w = size_issuers(data, ids, stock, method)
            self.assertLessEqual(w[:, :2].sum(1).max(), .30+1e-12)
            self.assertLessEqual(w[:, 2:].max(), .30+1e-12)
            np.testing.assert_allclose(w.sum(1), 1.)
        data['tradable'][18, 0] = False
        self.assertEqual(market_ranks(data, ids, 252)[0, 0], .5)
        market['qqq'][:] = 50
        np.testing.assert_array_equal(portfolio_targets(data, ids, score, market, dict(off=0)), 0.)


if __name__ == '__main__':
    unittest.main()
