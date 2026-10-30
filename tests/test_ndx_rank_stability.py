import unittest
import numpy as np

from ndx_rank_stability import (percentile_cube, validation_diagnostics,
    validation_weights, blended_scores, allocate)


class RankStabilityTests(unittest.TestCase):
    def test_percentiles_honor_eligibility_and_ties(self):
        p = np.array([[[1, 2, 2, np.nan]], [[3, 1, 1, np.nan]]])
        actual = percentile_cube(p, [[1, 1, 1, 0]])
        np.testing.assert_allclose(actual[0, 0], [1/3, 2.5/3, 2.5/3, 0])
        np.testing.assert_allclose(actual[1, 0], [1, .5, .5, 0])

    def test_validation_unknown_labels_never_influence_quality(self):
        p = np.tile(np.arange(4.), (2, 24, 1)); mask = np.ones((24, 4), bool)
        observed = mask.copy(); observed[:, -1] = False
        actual = np.tile(np.arange(4.), (24, 1)); ranks = percentile_cube(p, mask)
        before = validation_diagnostics(ranks, actual, mask, observed)
        actual[:, -1] = np.nan
        after = validation_diagnostics(ranks, actual, mask, observed)
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])

    def test_weights_are_shrunk_and_seed_permutation_equivariant(self):
        rng = np.random.default_rng(7); ic = rng.normal(0, .2, (5, 48))
        permutation = [3, 0, 4, 1, 2]
        for method in ('equal', 'validation_ic', 'validation_stable', 'stable_consensus'):
            w = validation_weights(ic, method)
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertTrue(np.all(w >= .15)); self.assertTrue(np.all(w <= .4))
            np.testing.assert_allclose(validation_weights(ic[permutation], method), w[permutation])

    def test_stable_quality_penalizes_time_varying_evidence(self):
        ic = np.array([[.2]*24+[-.2]*24, [.01]*48, [.01]*48])
        weights = validation_weights(ic, 'validation_stable')
        self.assertLess(weights[0], weights[1])
        np.testing.assert_allclose(validation_weights(np.zeros((5, 48)), 'validation_stable'), .2)

    def test_consensus_penalty_prefers_agreement_at_same_mean(self):
        cube = np.array([[[.9, .5]], [[.1, .5]]])
        mean = blended_scores(cube, cube, [.5, .5], [.5, .5])
        score = blended_scores(cube, cube, [.5, .5], [.5, .5], .25)
        np.testing.assert_allclose(mean, [[.5, .5]])
        self.assertLess(score[0, 0], score[0, 1])

    def test_soft_allocation_conserves_weights_and_excludes_ineligible_names(self):
        p = np.arange(15., dtype=float)[None, :]/15; mask = np.ones_like(p, bool); mask[:, -1] = False
        w = allocate(p, mask, 'soft10')
        self.assertAlmostEqual(w.sum(), 1.); self.assertEqual(np.count_nonzero(w), 10)
        self.assertEqual(w[0, -1], 0.); self.assertGreater(w[0, -2], w[0, -11])

    def test_future_predictions_cannot_affect_earlier_allocations(self):
        rng = np.random.default_rng(19); p = rng.uniform(size=(12, 15)); mask = np.ones_like(p, bool)
        for kind in ('retain5', 'retain10', 'soft10'):
            old = allocate(p, mask, kind); changed = p.copy(); changed[8:] = 100
            np.testing.assert_array_equal(allocate(changed, mask, kind)[:8], old[:8])

    def test_bad_evidence_is_rejected(self):
        with self.assertRaises(ValueError): validation_weights(np.zeros((5, 4)), 'validation_ic')
        with self.assertRaises(ValueError): validation_weights(np.full((5, 24), np.nan), 'validation_ic')
        with self.assertRaises(ValueError): allocate(np.ones((1, 2)), np.ones((1, 2), bool), 'unknown')


if __name__ == '__main__': unittest.main()
