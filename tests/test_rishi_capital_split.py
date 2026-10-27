import unittest
import numpy as np
from rishi_capital_split import pooled_returns


class CapitalSplitTests(unittest.TestCase):
    def test_sums_dollar_nav_and_includes_initial_fees(self):
        np.testing.assert_allclose(pooled_returns(np.array([49.,55.]),np.array([50.,40.]),100.),[-.01,95/99-1])

    def test_keeps_capital_drift_between_sleeves(self):
        left=np.array([50.,100.,50.]);right=np.array([50.,50.,100.])
        np.testing.assert_allclose(pooled_returns(left,right,100.),[0.,.5,0.])


if __name__=='__main__':unittest.main()
