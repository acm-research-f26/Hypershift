import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from ndx_rank_retraining import (neutralize_stock_initialization, rank_objective,
    split_validation, rank_statistics, percentile_predictions, epoch_choices)


class RankRetrainingTests(unittest.TestCase):
    def test_stock_specific_preferences_are_neutralized(self):
        model = nn.Module(); model.core = nn.Module()
        model.core.market = nn.Sequential(nn.Linear(4, 2), nn.GELU(), nn.Linear(2, 4))
        neutralize_stock_initialization(model, 'think')
        np.testing.assert_array_equal(model.core.market(torch.ones(3, 4)).detach(), 0.)
        self.assertTrue(model.core.market[2].weight.requires_grad)

    def test_loss_ignores_unobserved_nan_and_common_return_shift(self):
        score = torch.tensor([[.1, .2, .3, float('nan')]], requires_grad=True)
        actual = torch.tensor([[1., 2., 4., float('nan')]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        value = rank_objective(score, actual, mask)
        shifted = rank_objective(score, actual+100, mask)
        self.assertAlmostEqual(float(value.detach()), float(shifted.detach()), places=5)
        value.backward(); self.assertTrue(torch.isfinite(score.grad).all())
        self.assertEqual(float(score.grad[0, 3]), 0.)

    def test_loss_rewards_correct_order_and_is_seed_permutation_invariant(self):
        actual = torch.tensor([[-1., 0., 1.]])
        good = torch.tensor([[-1., 0., 1.]]); mask = torch.ones_like(good, dtype=torch.bool)
        self.assertLess(float(rank_objective(good, actual, mask)), float(rank_objective(-good, actual, mask)))
        self.assertAlmostEqual(float(rank_objective(good, actual, mask)),
                               float(rank_objective(good[:, [2,0,1]], actual[:, [2,0,1]], mask)), places=6)

    def test_validation_boundary_purges_crossing_labels(self):
        data = dict(origin=np.arange(50)*5, exit=np.arange(50)*5+6)
        selection, audit = split_validation(data, np.arange(50))
        self.assertLess(data['exit'][selection].max(), data['origin'][audit].min())
        self.assertEqual(len(selection), 24)

    def test_quality_and_percentiles_are_tie_aware(self):
        p = np.tile([1., 2., 2., 4.], (24, 1)); mask = np.ones_like(p, bool)
        stats = rank_statistics(p, p, mask)
        self.assertAlmostEqual(stats['rankic'], 1.); self.assertAlmostEqual(stats['ndcg10'], 1.)
        np.testing.assert_allclose(percentile_predictions(p, mask)[0], [.25, .625, .625, 1.])

    def test_epoch_selectors_use_separate_fixed_criteria(self):
        history = [dict(epoch=1, selection_loss=.1, selection_criterion=.2),
                   dict(epoch=2, selection_loss=.2, selection_criterion=.4),
                   dict(epoch=3, selection_loss=.1, selection_criterion=.3)]
        choice = epoch_choices(history)
        self.assertEqual(choice['loss'], [1]); self.assertEqual(choice['rank'], [2])
        self.assertEqual(choice['snapshots'], [2,3,1]); self.assertEqual(choice['initial'], [0])


if __name__ == '__main__': unittest.main()
