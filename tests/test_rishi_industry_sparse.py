import unittest
import numpy as np
from rishi_industry_sparse import sparse_control,priorities


class SparseIndustryTests(unittest.TestCase):
    def test_group_weight_multisets_and_unknown_are_preserved(self):
        w=np.array([.3,.2,0.,.25,.25,0.,0.]);eligible=np.array([1,1,1,1,1,1,0],bool)
        industry=np.array([6,6,6,np.nan,np.nan,np.nan,6]);priority=np.array([.9,.8,.1,.7,.8,.2,0])
        got=sparse_control(w,eligible,industry,priority)
        np.testing.assert_array_equal(got,[0,.2,.3,.25,0,.25,0])

    def test_priorities_reproducible_and_invalid_weights_rejected(self):
        np.testing.assert_array_equal(priorities(2024,0,5),priorities(2024,0,5))
        self.assertFalse(np.array_equal(priorities(2024,0,5),priorities(2024,1,5)))
        with self.assertRaises(ValueError):sparse_control(np.array([.5,.5]),np.array([1,0],bool),np.array([6.,6.]),np.array([.1,.2]))
