import unittest
import numpy as np
from rishi_cuda_rank_diagnostic import rank_changes


class CudaRankDiagnosticTests(unittest.TestCase):
    def test_changes_inside_tail_are_distinct_from_changes_at_boundary(self):
        eligible=np.ones((1,12),dtype=bool)
        original=np.arange(12,dtype=float)[None,:]
        swapped=original.copy();swapped[0,10],swapped[0,11]=swapped[0,11],swapped[0,10]
        d=rank_changes(swapped,original,eligible)
        self.assertEqual(d['rank_changed_origins'],1)
        self.assertEqual(d['top5_changed_origins'],0)
        changed=original.copy();changed[0,0]=20
        self.assertEqual(rank_changes(changed,original,eligible)['top5_changed_origins'],1)

    def test_ties_and_inactive_garbage(self):
        eligible=np.array([[True,True,False]])
        a=np.array([[1.,1.,float('nan')]])
        b=np.array([[1.,1.,99.]])
        d=rank_changes(a,b,eligible)
        self.assertEqual(d['rank_changed_origins'],0)
        self.assertEqual(d['top5_changed_origins'],0)
        with self.assertRaises(ValueError):rank_changes(np.array([[float('nan'),1.,0.]]),b,eligible)
