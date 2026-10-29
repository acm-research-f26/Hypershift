import unittest
import numpy as np
from rishi_historical_selection import targets

class SelectionTests(unittest.TestCase):
 def test_retention_ties_mask_and_no_lookahead(self):
  s=np.array([[.9,.8,.8],[.8,.81,.80],[.2,.3,.4]]);m=np.array([[1,1,1],[1,1,0],[1,1,1]],bool)
  w=targets(s,m,2,'retention');np.testing.assert_allclose(w[0],[.5,.25,.25]);np.testing.assert_allclose(w[1],[.5,.5,0]);self.assertAlmostEqual(w.sum(1).min(),1)
  s[2]=[.9,.9,.9];np.testing.assert_allclose(targets(s,m,2,'retention')[:2],w[:2])
 def test_confidence_and_equal_ties(self):
  s=np.array([[.9,.9,.1]]);m=np.ones_like(s,bool)
  for mode in ('equal','confidence','retention'):np.testing.assert_allclose(targets(s,m,1,mode),[[.5,.5,0]])

if __name__=='__main__':unittest.main()
