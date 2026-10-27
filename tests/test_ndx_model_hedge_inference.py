import unittest
import numpy as np
from finalize_ndx_model_hedge import inference_with_noops


class NoopInferenceTests(unittest.TestCase):
    def test_exact_noops_are_zero_and_other_contrasts_still_bootstrap(self):
        rng=np.random.default_rng(7)
        baseline=rng.normal(.001,.01,80);other=baseline+rng.normal(.0002,.003,80)
        paths={'a':baseline,'clone':baseline.copy(),'b':other}
        result=inference_with_noops(paths,[('clone','a'),('b','a')],20,100)
        self.assertTrue(result.deterministic_noop.iloc[0])
        self.assertEqual(result.simultaneous_active_low.iloc[0],0.)
        self.assertEqual(result.paired_sr_high.iloc[0],0.)
        self.assertFalse(result.deterministic_noop.iloc[1])
        self.assertAlmostEqual(result.annual_active_mean.iloc[1],np.mean(other-baseline)*252)


if __name__=='__main__':unittest.main()
