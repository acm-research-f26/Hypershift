import unittest
import numpy as np
from rishi_cash_yield import sidecar


class CashYieldTests(unittest.TestCase):
    def test_prior_cash_compounding_and_first_day(self):
        interest,balance,wealth,returns=sidecar([1e6,1e6,1e6],[500000,400000,0],[.1,.01,.02])
        np.testing.assert_allclose(interest,[0,5000,8100])
        np.testing.assert_allclose(balance,[0,5000,13100])
        np.testing.assert_allclose(wealth,[1e6,1005000,1013100])
        np.testing.assert_allclose(returns,[0,.005,1013100/1005000-1])

    def test_zero_yield_and_bad_cash(self):
        nav=np.array([1e6,990000,1010000.]);cash=np.array([5e5,4e5,2e5])
        interest,balance,wealth,_=sidecar(nav,cash,np.zeros(3))
        np.testing.assert_array_equal(wealth,nav)
        self.assertTrue((interest==0).all() and (balance==0).all())
        with self.assertRaises(ValueError):sidecar(nav,[-1,0,0],np.zeros(3))
