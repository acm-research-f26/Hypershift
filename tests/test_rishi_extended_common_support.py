import unittest
import numpy as np
import pandas as pd
from rishi_extended_common_support import common_correlations


class CommonSupportTests(unittest.TestCase):
    def test_inner_join_reranks_and_preserves_ties(self):
        cov=pd.DataFrame(dict(security_id=list(range(12)),score_rank=np.arange(12),feature_rank=np.arange(12)))
        ret=pd.DataFrame(dict(security_id=list(range(1,12)),score_rank=np.arange(1,12)*3,feature_rank=-np.arange(1,12)))
        n,c,r=common_correlations(cov,ret)
        self.assertEqual(n,11);self.assertAlmostEqual(c,1);self.assertAlmostEqual(r,-1)
        ret.loc[0,'score_rank']=ret.loc[1,'score_rank']
        with self.assertRaises(ValueError):common_correlations(cov,ret)

    def test_duplicate_or_small_support(self):
        a=pd.DataFrame(dict(security_id=list(range(9)),score_rank=np.arange(9),feature_rank=np.arange(9)))
        n,c,r=common_correlations(a,a)
        self.assertEqual(n,9);self.assertTrue(np.isnan(c) and np.isnan(r))
        with self.assertRaises(ValueError):common_correlations(pd.concat([a,a.iloc[[0]]]),a)
