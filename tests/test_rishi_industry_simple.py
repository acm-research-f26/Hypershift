import unittest
import numpy as np
from rishi_industry_simple import signal_control


class IndustrySimpleTests(unittest.TestCase):
    def test_score_order_tie_break_and_group_multisets(self):
        w=np.array([.3,.2,0.,.5,0.,0.]);eligible=np.array([1,1,1,1,1,0],bool)
        industry=np.array([6.,6.,6.,np.nan,np.nan,6.]);score=np.array([1.,2.,2.,0.,1.,100.]);priority=np.array([.5,.7,.1,.4,.2,.0])
        out=signal_control(w,eligible,industry,score,priority)
        np.testing.assert_array_equal(out,[0,.2,.3,0,.5,0])

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):signal_control(np.array([1.,0]),np.array([1,1],bool),np.array([6.,6.]),np.array([1.,np.nan]),np.array([.2,.1]))
        with self.assertRaises(ValueError):signal_control(np.array([.5,.5]),np.array([1,0],bool),np.array([6.,6.]),np.array([1.,2.]),np.array([.2,.1]))
