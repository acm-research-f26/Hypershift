import unittest
import numpy as np
from rishi_historical_blend import blend

class BlendTests(unittest.TestCase):
    def test_endpoints_midpoint_and_ineligible(self):
        b=np.array([[.2,.8,np.nan]]); d=np.array([[.6,.4,np.nan]]); mask=np.array([[True,True,False]])
        np.testing.assert_allclose(blend(b,d,mask,0)[mask],b[mask])
        np.testing.assert_allclose(blend(b,d,mask,1)[mask],d[mask])
        np.testing.assert_allclose(blend(b,d,mask,.5)[mask],[.4,.6])
        self.assertTrue(np.isnan(blend(b,d,mask,.5)[0,2]))
    def test_shape_and_bad_active_values_rejected(self):
        with self.assertRaises(ValueError): blend(np.ones((1,2)),np.ones((2,1)),np.ones((1,2),bool),.5)
        with self.assertRaises(ValueError): blend(np.array([[np.nan]]),np.ones((1,1)),np.ones((1,1),bool),.5)
        with self.assertRaises(ValueError): blend(np.ones((1,1)),np.ones((1,1)),np.ones((1,1),bool),1.1)

if __name__=='__main__': unittest.main()
