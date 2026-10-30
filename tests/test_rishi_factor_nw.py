import unittest
import numpy as np
from rishi_factor_nw import fit_hac


class FactorNWTests(unittest.TestCase):
    def test_intercept_only_hac_matches_hand_computed_bartlett_cases(self):
        x=np.ones((4,1));y=np.array([1.,-1.,1.,-1.])
        self.assertAlmostEqual(fit_hac(y,x,0)['covariance'][0,0],1/3)
        self.assertAlmostEqual(fit_hac(y,x,1)['covariance'][0,0],1/12)
        y=np.array([1.,1.,-1.,-1.])
        self.assertAlmostEqual(fit_hac(y,x,1)['covariance'][0,0],5/12)

    def test_exact_linear_fit_recovers_intercept_and_slope(self):
        z=np.arange(-2.,3.);x=np.column_stack([np.ones(5),z])
        result=fit_hac(2+3*z,x,1)
        np.testing.assert_allclose(result['beta'],[2,3],atol=1e-12)
        np.testing.assert_allclose(result['covariance'],0,atol=1e-24)

    def test_singular_factor_design_is_rejected(self):
        with self.assertRaises(ValueError):fit_hac(np.arange(6.),np.ones((6,2)),1)


if __name__=='__main__':unittest.main()
