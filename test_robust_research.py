import unittest
import numpy as np
import torch
from research_engine import prepare_panel, fold_indices, backtest as baseline_backtest
from robust_engine import backtest, allocation
from robust_models import robust_objective, make_model, CausalMix, EXPERIMENTS
from test_research import fixture


class RobustTests(unittest.TestCase):
    def test_baseline_accounting_unchanged(self):
        d = prepare_panel(fixture())
        _, _, ids = fold_indices(d, 2017)
        score = np.random.default_rng(42).normal(size=(len(ids), 12))
        for method in ['equal', 'confidence', 'long_short', 'buy_hold', 'equal_universe']:
            a, x, _ = backtest(d, ids, score, 5, method)
            b, y, _ = baseline_backtest(d, ids, score, 5, method)
            np.testing.assert_array_equal(x.net, y.net)
            self.assertEqual(a, b)

    def test_huber_limits_tail_gradient(self):
        x = torch.tensor([[.1, 100.]], requires_grad=True)
        loss = robust_objective(x, torch.zeros_like(x), torch.ones_like(x), 'think_huber')
        loss.backward()
        np.testing.assert_allclose(x.grad, [[.1, 1.]], atol=1e-6)

    def test_masked_outlier_cannot_change_loss(self):
        x = torch.tensor([[1., 2.]])
        m = torch.tensor([[1., 0.]])
        a = robust_objective(x, torch.zeros_like(x), m, 'stockmixer_huber')
        b = robust_objective(x, torch.tensor([[0., 1e6]]), m, 'stockmixer_huber')
        self.assertEqual(a.item(), b.item())

    def test_causal_temporal_mixer(self):
        torch.manual_seed(1)
        model = CausalMix(16)
        x = torch.randn(2, 5, 16)
        y = x.clone(); y[..., 8:] += 100
        torch.testing.assert_close(model(x)[..., :8], model(y)[..., :8])

    def test_portfolio_risk_and_buffer_rules(self):
        score = np.arange(20.)
        mask = np.ones(20, bool); mask[-1] = False
        ret = np.random.default_rng(1).normal(size=(60, 20))*.01
        old = np.zeros(20); old[9:14] = .2
        for method in ['inverse_vol', 'buffer_equal', 'buffer_inverse_vol']:
            w = allocation(score, mask, 5, method, ret, old)
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertEqual(w[-1], 0.)
            self.assertTrue((w >= 0).all())
            self.assertLessEqual(w.max(), .4+1e-10)
        buffered = allocation(score, mask, 5, 'buffer_equal', ret, old)
        np.testing.assert_allclose(buffered, old)

    def test_daily_timing_and_masking(self):
        p = fixture(); d = prepare_panel(p, horizon=1)
        train, val, test = fold_indices(d, 2017)
        self.assertTrue((d['exit']-d['entry'] == 1).all())
        self.assertTrue((np.diff(d['origin']) == 1).all())
        self.assertLess(d['exit'][train[-1]], d['origin'][val[0]])
        self.assertLess(d['exit'][val[-1]], d['origin'][test[0]])
        before = d['x'][test[0]].copy()
        p.close[d['origin'][test[0]]+1:] *= 10
        np.testing.assert_array_equal(before, prepare_panel(p, horizon=1)['x'][test[0]])
        result, path, periods = backtest(d, test, np.zeros_like(d['y'][test]), 5, 'inverse_vol')
        self.assertEqual(len(path), len(test))
        np.testing.assert_array_equal(periods, path.net)

    def test_new_models_finite_gradients(self):
        x = torch.randn(2, 16, 12, 5)*.1
        mask = torch.ones(2, 12); mask[:, -1] = 0
        for kind in list(EXPERIMENTS)[9:]:
            model = make_model(kind, 12, [{i} for i in range(12)])
            score, aux = model(x, mask)
            loss = robust_objective(score, torch.randn_like(score), mask, kind, aux)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for param in model.parameters():
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all())


if __name__ == '__main__':
    unittest.main()
