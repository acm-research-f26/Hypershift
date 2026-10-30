import unittest

import numpy as np

from ndx_retention import retention_targets, hedged_targets


class RetentionTests(unittest.TestCase):
    def test_bonus_reduces_marginal_turnover_but_does_not_lock_holdings(self):
        scores = np.array([[.90, .80, .70], [.79, .80, .70], [.70, .80, .90]])
        actual = retention_targets(scores, np.ones_like(scores, bool), k=1)
        np.testing.assert_array_equal(actual, [[1, 0, 0], [1, 0, 0], [0, 0, 1]])

    def test_ineligible_previous_target_is_removed(self):
        scores = np.array([[.9, .8, .7], [.99, .8, .7]])
        mask = np.array([[1, 1, 1], [0, 1, 1]], bool)
        np.testing.assert_array_equal(retention_targets(scores, mask, 1),
                                      [[1, 0, 0], [0, 1, 0]])

    def test_boundary_ties_receive_fractional_retention_bonus(self):
        scores = np.array([[.9, .8, .8], [.8, .81, .80]])
        mask = np.array([[1, 1, 1], [1, 1, 0]], bool)
        np.testing.assert_allclose(retention_targets(scores, mask, 2),
                                   [[.5, .25, .25], [.5, .5, 0]])

    def test_future_changes_cannot_change_prior_targets(self):
        rng = np.random.default_rng(7)
        scores = rng.uniform(size=(12, 8)); mask = np.ones_like(scores, bool)
        expected = retention_targets(scores, mask)
        scores[8:] = 100.; mask[8:, :3] = False
        np.testing.assert_array_equal(retention_targets(scores, mask)[:8],
                                      expected[:8])

    def test_state_continues_across_separately_processed_years(self):
        # A year boundary must not reset the preceding decision's bonus.
        scores = np.array([[.90, .80], [.79, .80]])
        mask = np.ones_like(scores, bool)
        np.testing.assert_array_equal(retention_targets(scores, mask, 1)[1], [1, 0])
        np.testing.assert_array_equal(retention_targets(scores[1:], mask[1:], 1)[0], [0, 1])

    def test_short_is_limited_by_cap_budget_and_volatility(self):
        weights = np.array([[.5, .5], [1, 0], [0, 1]])
        actual = hedged_targets(weights, [.2, .9, .2], [.2, .2, .8], .5)
        np.testing.assert_allclose(actual, [[.1, .1, -.5], [.9, 0, -.1], [0, .2, -.25]])
        self.assertTrue(np.all(np.abs(actual).sum(axis=1) <= 1+1e-12))
        np.testing.assert_allclose(hedged_targets(weights, [.2, .9, .2], [.2]*3, 0)[:, -1], 0)

    def test_invalid_inputs_fail_before_accounting(self):
        for scores, mask in [([[np.nan]], [[1]]), ([[1, 2]], [[1]]), ([[1]], [[0]])]:
            with self.assertRaises(ValueError): retention_targets(scores, mask)
        for exposure, volatility, cap in [([1.1], [.2], .5), ([.5], [-1], .5), ([.5], [.2], 1.1)]:
            with self.assertRaises(ValueError): hedged_targets([[1.]], exposure, volatility, cap)


if __name__ == '__main__':
    unittest.main()
