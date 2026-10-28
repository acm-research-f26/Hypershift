import unittest
import numpy as np
from rishi_logmap_precision import scalar_map,CAP


class ScalarLogMapTests(unittest.TestCase):
    def test_scalar_values_symmetry_and_cap(self):
        x=np.array([0.,.5,-.5,CAP+.000001,-CAP-.000001])
        y=scalar_map(x)
        np.testing.assert_allclose(y,[0.,np.arctanh(.5),-np.arctanh(.5),np.arctanh(CAP),-np.arctanh(CAP)],rtol=0,atol=1e-12)
        self.assertEqual(scalar_map([CAP])[0],scalar_map([1.])[0])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):scalar_map([np.nan])
