import unittest
import numpy as np
from rishi_historical_risk import smooth
class RiskTests(unittest.TestCase):
 def test_recurrence_and_future(self):
  x=np.array([.4,.8,.0,1.]);y=smooth(x);np.testing.assert_allclose(y,[.4,.5,.375,.53125]);x[3]=.2;np.testing.assert_allclose(smooth(x)[:3],y[:3])
 def test_invalid(self):
  with self.assertRaises(ValueError):smooth(np.array([.5,np.nan]))
if __name__=='__main__':unittest.main()
