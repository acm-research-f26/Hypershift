import unittest
import numpy as np
from rishi_extended_association import rank_pair,correlation


class AssociationTests(unittest.TestCase):
    def test_finite_common_support_and_ties(self):
        s=np.arange(12,dtype=float);f=s.copy();f[0]=np.nan;f[1]=f[2]
        mask=np.ones(12,bool);mask[-1]=False
        support,x,y=rank_pair(s,f,mask)
        self.assertEqual(support.sum(),10)
        self.assertFalse(support[0]);self.assertFalse(support[-1])
        self.assertEqual(y[0],y[1]);self.assertGreater(correlation(x,y),.99)

    def test_undefined_and_bad_axes(self):
        self.assertTrue(np.isnan(correlation(np.arange(9),np.arange(9))))
        self.assertTrue(np.isnan(correlation(np.ones(10),np.arange(10))))
        with self.assertRaises(ValueError):rank_pair([1,2],[1],[True,True])
