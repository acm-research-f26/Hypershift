import unittest
import numpy as np
from rishi_industry_matched import matched_control


class IndustryMatchedTests(unittest.TestCase):
    def test_preserves_each_group_including_unknown_and_zero(self):
        original=np.array([.4,.1,.5,0.,0.,0.]);eligible=np.array([1,1,1,1,1,0],bool)
        industry=np.array([6.,6.,np.nan,np.nan,10.,6.])
        out=matched_control(original,eligible,industry)
        np.testing.assert_allclose(out,[.25,.25,.25,.25,0,0])

    def test_rejects_ineligible_holdings_or_bad_weights(self):
        with self.assertRaises(ValueError):matched_control(np.array([.5,.5]),np.array([1,0],bool),np.array([6.,np.nan]))
        with self.assertRaises(ValueError):matched_control(np.array([.4,.4]),np.array([1,1],bool),np.array([6.,6.]))
        with self.assertRaises(ValueError):matched_control(np.array([1.000009]),np.array([1],bool),np.array([6.]))
