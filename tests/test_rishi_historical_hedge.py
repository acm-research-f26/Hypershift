import unittest
import numpy as np
from rishi_historical_hedge import positions
class HedgeTests(unittest.TestCase):
 def test_cap_cash_and_vol(self):
  w=np.array([[.5,.5],[1.,0.],[0.,1.]]);long=np.array([.5,.9,.2]);vol=np.array([.4,.4,.8]);p=positions(w,long,vol,.5);np.testing.assert_allclose(p[:,-1],[-.5,-.1,-.25]);self.assertTrue(np.all(np.abs(p).sum(1)<=1+1e-12))
 def test_zero_and_bad(self):
  np.testing.assert_allclose(positions(np.array([[1.]]),np.array([.5]),np.array([.2]),0),[[.5,0.]])
  with self.assertRaises(ValueError):positions(np.array([[1.]]),np.array([.5]),np.array([.2]),-1)
if __name__=='__main__':unittest.main()
