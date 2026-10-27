import unittest
from itertools import combinations, permutations
import numpy as np
import pandas as pd
from hyperbolicity_audit import delta_metric, euclidean_distances, graph_audit, shortest_paths, group_adjacency
from download_hourly import full_hour_mask


class AuditTests(unittest.TestCase):
    def test_four_point_matches_independent_gromov_product(self):
        points = np.random.default_rng(31).normal(size=(8,3))
        d = euclidean_distances(points)
        expected = 0.0
        for w,x,y,z in permutations(range(8),4):
            g = lambda a,b: (d[w,a]+d[w,b]-d[a,b])/2
            expected = max(expected,min(g(x,y),g(y,z))-g(x,z))
        result = delta_metric(d)
        self.assertAlmostEqual(result['delta'],expected)
        self.assertEqual(result['quadruples_evaluated'],70)
        self.assertAlmostEqual(result['witness']['delta'],expected)

    def test_known_metrics_and_disconnected_graph(self):
        path = np.abs(np.arange(6)[:,None]-np.arange(6)[None,:])
        self.assertEqual(delta_metric(path)['delta'],0)
        clique = np.ones((6,6))-np.eye(6)
        self.assertEqual(delta_metric(clique)['delta'],0)
        a = np.zeros((4,4),bool)
        for i in range(4):
            a[i,(i+1)%4] = a[(i+1)%4,i] = True
        self.assertEqual(delta_metric(shortest_paths(a))['delta'],1)
        disjoint = np.zeros((6,6),bool); disjoint[:4,:4]=a
        result = graph_audit(disjoint)
        self.assertIsNone(result['global_delta'])
        self.assertEqual(result['max_component_lower_bound'],1)

    def test_sample_is_lower_bound_and_bound_certificate(self):
        d = euclidean_distances(np.random.default_rng(4).normal(size=(12,4)))
        exact, sample = delta_metric(d), delta_metric(d,exact_limit=0,samples=50)
        self.assertFalse(sample['exact'])
        self.assertIsNone(sample['delta'])
        self.assertLessEqual(sample['lower_bound'],exact['delta']+1e-12)
        self.assertGreaterEqual(sample['upper_bound'],exact['delta']-1e-12)
        circle = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]])
        self.assertTrue(delta_metric(circle,exact_limit=0,samples=20)['exact'])

    def test_scale_normalization_and_s_membership(self):
        d = euclidean_distances(np.random.default_rng(1).normal(size=(7,3)))
        first, second = delta_metric(d), delta_metric(d*17)
        self.assertAlmostEqual(first['relative_lower_bound'],second['relative_lower_bound'])
        groups = [{0,1},{0,1},{1,2}]
        a = group_adjacency(3,groups,2)
        self.assertTrue(a[0,1]); self.assertFalse(a[1,2])

    def test_hourly_completed_bars_and_partial_final_bar(self):
        index = pd.DatetimeIndex(['2026-09-28 13:30Z','2026-09-28 14:30Z','2026-09-28 19:30Z'])
        mask = full_hour_mask(index,pd.Timestamp('2026-09-28 15:00Z'))
        np.testing.assert_array_equal(mask,[True,False,False])
        mask = full_hour_mask(index,pd.Timestamp('2026-09-29 01:00Z'))
        np.testing.assert_array_equal(mask,[True,True,False])


if __name__ == '__main__':
    unittest.main()
