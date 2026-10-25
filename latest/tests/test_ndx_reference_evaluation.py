import unittest
import numpy as np
from report_ndx_reference_networks import family,blend_scores


class ReferenceEvaluationTests(unittest.TestCase):
    def test_fixed_family_includes_blend_vs_standalone(self):
        rows=family('ten_seed');self.assertEqual(len(rows),23);self.assertEqual(len({r[0] for r in rows}),23)
        self.assertEqual(sum('blend minus standalone' in r[0] for r in rows),3)

    def test_blend_is_always_equal_and_does_not_use_targets(self):
        a=np.array([[0.,1.,.2]]);b=np.array([[1.,0.,.4]])
        np.testing.assert_allclose(blend_scores(a,b),np.array([[.5,.5,.3]]),atol=1e-15,rtol=0)


if __name__=='__main__':unittest.main()
