import unittest
import numpy as np
from rishi_coverage_changes import change_correlations


class CoverageChangeTests(unittest.TestCase):
    def test_rank_correlation_and_liquidity_residual(self):
        x=np.arange(30.);y=x.copy();z=np.tile([0.,1.,2.],10)
        raw,adjusted,status=change_correlations(x,y,z)
        self.assertEqual(status,'ok');self.assertAlmostEqual(raw,1);self.assertAlmostEqual(adjusted,1)

    def test_undefined_cases(self):
        x=np.arange(30.)
        self.assertEqual(change_correlations(x[:10],x[:10],x[:10])[2],'too_few')
        self.assertEqual(change_correlations(x,np.ones(30),x)[2],'constant_score_or_coverage')
        self.assertEqual(change_correlations(x,x,np.ones(30))[2],'constant_liquidity')
        self.assertEqual(change_correlations(x,x,x)[2],'constant_residual')
