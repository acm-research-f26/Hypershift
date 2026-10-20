import json
import unittest
import numpy as np
from run_ndx_seed_expansion import CONFIG,bootstrap_draws
from run_robust_research import ensemble_predictions


class SeedExpansionTests(unittest.TestCase):
    def test_fixed_seed_inventory_unique_and_disjoint(self):
        plan=json.loads(CONFIG.read_text());self.assertEqual(len(plan['new_seeds']),20)
        seeds=plan['original_seeds']+plan['new_seeds'];self.assertEqual(len(set(seeds)),25)
        self.assertEqual(len(plan['test_years'])*len(plan['new_seeds']),80)

    def test_resampling_is_reproducible_and_uses_all_seed_positions(self):
        plan=json.loads(CONFIG.read_text());a=bootstrap_draws(plan);b=bootstrap_draws(plan)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(128,25))
        self.assertEqual(set(a.ravel()),set(range(25)))
        self.assertGreater(int(np.sum([len(set(row))<25 for row in a])),120)

    def test_weighted_rank_aggregation_matches_direct_resampled_ensemble(self):
        rng=np.random.default_rng(7);p=rng.normal(size=(25,8,14));mask=rng.random((8,14))>.1
        draw=rng.integers(0,25,25);ranks=np.stack([ensemble_predictions([v],mask) for v in p])
        direct=ensemble_predictions(p[draw],mask);fast=ranks[draw].mean(0)
        np.testing.assert_array_equal(direct,fast)


if __name__=='__main__':unittest.main()
