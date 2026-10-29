import unittest
import numpy as np
from rishi_historical_objective_evaluation import average_ranks
class EvaluationTests(unittest.TestCase):
 def test_ties_ensemble_and_mask(self):
  a=np.array([[[2.,2.,100.]],[[1.,3.,-100.]]]);mask=np.array([[1,1,0]],bool);r=average_ranks(a,mask);np.testing.assert_allclose(r[0,:2],[.625,.875]);self.assertTrue(np.isnan(r[0,2]))
 def test_invalid(self):
  with self.assertRaises(ValueError):average_ranks(np.array([[[np.nan]]]),np.ones((1,1),bool))
if __name__=='__main__':unittest.main()

