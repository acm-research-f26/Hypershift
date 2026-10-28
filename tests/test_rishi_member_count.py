import unittest
import numpy as np
from rishi_member_count import toy_input


class MemberCountTests(unittest.TestCase):
    def test_masks_variants_and_replication(self):
        x,m=toy_input(0,4,'replicated_peer','nan')
        self.assertEqual(int(m.sum()),4);self.assertTrue(np.isnan(x[~m]).all())
        np.testing.assert_array_equal(x[1],x[2]);np.testing.assert_array_equal(x[1],x[3])
        y,n=toy_input(0,4,'replicated_peer','garbage');np.testing.assert_array_equal(y[m],x[m]);self.assertTrue((y[~n]==12345).all())
        diverse,_=toy_input(0,4,'diverse','zero');self.assertFalse(np.array_equal(diverse[1],diverse[2]));np.testing.assert_array_equal(x[0],diverse[0])

    def test_reproducible_and_reject_invalid_count(self):
        a,_=toy_input(2,6,'diverse','zero');b,_=toy_input(2,6,'diverse','zero');np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):toy_input(0,9,'diverse','zero')
