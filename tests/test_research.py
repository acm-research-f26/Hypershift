import unittest
import numpy as np
import pandas as pd
import torch
from research_engine import (Panel, prepare_panel, fold_indices, ranking_metrics, target_weights,
                             backtest, performance, training_groups)
from research_models import RankModel, VARIANTS, objective
from run_architecture_research import scale_inputs, choose_policy, ensemble_predictions


def fixture():
    rng = np.random.default_rng(913)
    dates = pd.bdate_range('2012-01-02', periods=1800)
    close = 100*np.cumprod(1+rng.normal(.0002, .01, (len(dates), 12)), axis=0)
    return Panel('synthetic_test_only', dates, [str(i) for i in range(12)], close,
                 np.full_like(close, 1e8), close[:, 0], [])


class ResearchTests(unittest.TestCase):
    def test_ineligible_nan_scores_cannot_contaminate_weights(self):
        mask = np.array([True, False, True, True])
        missing = np.array([1., np.nan, 2., 3.])
        finite = np.array([1., 1000., 2., 3.])
        for method in ['equal', 'confidence', 'long_short']:
            weights = target_weights(missing, mask, 1, method)
            self.assertTrue(np.isfinite(weights).all())
            self.assertEqual(weights[1], 0.)
            np.testing.assert_allclose(weights, target_weights(finite, mask, 1, method))

    def test_ensemble_ignores_ineligible_predictions(self):
        eligible = np.array([[True, False, True, True]])
        predictions = [np.array([[1., 100., 2., 3.]]), np.array([[3., -100., 2., 1.]])]
        before = ensemble_predictions(predictions, eligible)
        predictions[0][0, 1] = -999.
        predictions[1][0, 1] = 999.
        np.testing.assert_array_equal(before, ensemble_predictions(predictions, eligible))
        np.testing.assert_allclose(before[eligible], 2/3)

    def test_future_features_eligibility_and_scaling_invariant(self):
        panel = fixture()
        a = prepare_panel(panel)
        train, val, test = fold_indices(a, 2017)
        cut = a['origin'][test[0]]+1
        panel.close[cut:] *= np.arange(1, 13)
        panel.volume[cut:] *= 100
        b = prepare_panel(panel)
        ids = np.flatnonzero(a['origin'] < cut)
        np.testing.assert_array_equal(a['mask'][ids], b['mask'][ids])
        np.testing.assert_allclose(a['x'][ids], b['x'][ids])
        _, ma, sa = scale_inputs(a, train)
        _, mb, sb = scale_inputs(b, train)
        np.testing.assert_allclose(ma, mb)
        np.testing.assert_allclose(sa, sb)
        self.assertEqual(training_groups(a, train), training_groups(b, train))

    def test_delayed_labels_and_purging(self):
        d = prepare_panel(fixture())
        i = 50
        t = d['origin'][i]
        np.testing.assert_allclose(d['y'][i], d['panel'].close[t+6]/d['panel'].close[t+1]-1, rtol=1e-6)
        train, val, test = fold_indices(d, 2017)
        self.assertLess(d['exit'][train[-1]], d['origin'][val[0]])
        self.assertLess(d['exit'][val[-1]], d['origin'][test[0]])

    def test_missing_outcome_does_not_change_eligibility(self):
        p = fixture()
        a = prepare_panel(p)
        i = 30
        p.close[a['exit'][i], 0] = np.nan
        b = prepare_panel(p)
        np.testing.assert_array_equal(a['mask'][i], b['mask'][i])
        self.assertTrue(b['mask'][i, 0])
        self.assertFalse(b['labelmask'][i, 0])

    def test_rank_oracle_reverse_and_ties(self):
        y = np.array([[-.04, -.03, -.02, -.01]])
        mask = np.ones_like(y, dtype=bool)
        oracle, _ = ranking_metrics(y, y, mask, mask, [1, 3])
        reverse, _ = ranking_metrics(-y, y, mask, mask, [1, 3])
        self.assertAlmostEqual(oracle['IC'], 1)
        self.assertAlmostEqual(oracle['RankIC'], 1)
        self.assertAlmostEqual(oracle['NDCG@3'], 1)
        self.assertEqual(reverse['NDCG@1'], 0)
        a, _ = ranking_metrics(np.zeros_like(y), y, mask, mask, [3])
        b, _ = ranking_metrics(np.zeros_like(y), y[:, ::-1], mask, mask, [3])
        self.assertAlmostEqual(a['NDCG@3'], b['NDCG@3'])
        np.testing.assert_allclose(target_weights(np.zeros(4), mask[0], 2, 'equal'), [.25]*4)

    def test_turnover_costs_and_daily_risk(self):
        d = prepare_panel(fixture())
        ids = np.arange(20, 35)
        scores = np.tile(np.arange(12), (len(ids), 1))
        zero, zd, _ = backtest(d, ids, scores, 3, 'equal', cost_multiplier=0)
        cost, cd, _ = backtest(d, ids, scores, 3, 'equal', cost_multiplier=2)
        self.assertEqual(len(cd), len(ids)*5)
        self.assertGreater(cost['cost_dollars'], 0)
        self.assertLess(cost['net_cumulative_return'], zero['net_cumulative_return'])
        self.assertGreater(zero['turnover_mean'], 2/len(ids)) # Drift creates extra trading.
        self.assertAlmostEqual(zero['gross_cumulative_return'], zero['net_cumulative_return'])
        expected = np.prod(1+cd.net)-1
        self.assertAlmostEqual(expected, cost['net_cumulative_return'])
        self.assertLessEqual(performance([-.1, .1])['maximum_drawdown'], -.1+1e-10)

    def test_liquidity_cap(self):
        d = prepare_panel(fixture())
        d['adv'][:] = 10000
        metrics, _, _ = backtest(d, np.arange(20, 23), np.ones((3, 12)), 3, 'equal')
        self.assertGreater(metrics['constrained_orders'], 0)
        self.assertLess(metrics['turnover_mean'], .01)

    def test_constant_return_accounting_matches_closed_form(self):
        panel = fixture()
        panel.close = 100*1.001**np.arange(len(panel.dates))[:,None]*np.ones((1,12))
        d = prepare_panel(panel)
        metric, daily, periods = backtest(d, np.arange(20,24), np.zeros((4,12)), 3, 'equal', cost_multiplier=0)
        np.testing.assert_allclose(daily.net, .001, atol=1e-12)
        np.testing.assert_allclose(periods, 1.001**5-1, atol=1e-12)
        self.assertAlmostEqual(metric['net_cumulative_return'], 1.001**20-1)

    def test_ranking_uses_full_eligible_universe(self):
        y = np.array([[.01,.02,.03,.04,.05,.06]])
        pred = np.array([[6.,5.,4.,3.,2.,1.]])
        eligible = np.ones_like(y,dtype=bool)
        metric, daily = ranking_metrics(pred,y,eligible,eligible,[1,3])
        self.assertAlmostEqual(metric['IC'],-1.)
        self.assertAlmostEqual(metric['RankIC'],-1.)
        self.assertEqual(int(daily.eligible.iloc[0]),6)

    def test_policy_selection_cannot_see_test_outcomes(self):
        panel = fixture()
        a = prepare_panel(panel)
        _, val, test = fold_indices(a,2017)
        pred = np.random.default_rng(87).normal(size=(len(val),12))
        before = choose_policy(a,val,[pred],[3,5,8])
        cutoff = a['origin'][test[0]]
        panel.close[cutoff:] *= np.linspace(.01,100,12)
        panel.volume[cutoff:] *= 100
        b = prepare_panel(panel)
        after = choose_policy(b,val,[pred],[3,5,8])
        self.assertEqual(before,after)

    def test_architectures_gradients_and_masked_nodes(self):
        torch.set_num_threads(2)
        groups = [{i, (i+1)%12} for i in range(12)]
        for kind in VARIANTS:
            torch.manual_seed(7)
            model = RankModel(kind, 12, groups)
            x = torch.randn(2, 16, 12, 5)*.2
            mask = torch.ones(2, 12)
            mask[:, -1] = 0
            score, aux = model(x, mask)
            changed = x.clone()
            changed[:, :, -1] = 1e4
            other, _ = model(changed, mask)
            torch.testing.assert_close(score, other)
            loss = objective(score, torch.randn(2, 12), mask, kind, aux, torch.ones(2, 12))
            loss.backward()
            self.assertTrue(torch.isfinite(score).all(), kind)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), kind)


if __name__ == '__main__':
    unittest.main()
