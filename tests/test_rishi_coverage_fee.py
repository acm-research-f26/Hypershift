import unittest
import numpy as np
from rishi_coverage_fee import components


class FeeTests(unittest.TestCase):
    def test_fee_normalized_by_prior_nav(self):
        result=components([990000,1000000],[-.01,1000000/990000-1],[100,200],[10000,20000])
        np.testing.assert_allclose(result['fee'],[100/1e6,200/990000])
        np.testing.assert_allclose(result['gross']-result['fee'],result['net'])

    def test_reject_bad_ledger(self):
        with self.assertRaises(ValueError):components([1e6,1e6],[0,.1],[0,0],[0,0])
        with self.assertRaises(ValueError):components([1e6,1e6],[0,0],[-1,0],[0,0])
