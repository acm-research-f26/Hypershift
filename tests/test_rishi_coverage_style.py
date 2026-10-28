import unittest
import numpy as np
from rishi_coverage_style import conditional_correlation


class CoverageStyleTests(unittest.TestCase):
    def test_rank_and_industry_partial_correlation(self):
        rng=np.random.default_rng(5);n=40
        groups=np.repeat([1,2],n//2);adv=np.exp(rng.normal(size=n))
        score=rng.normal(size=n);coverage=.6*score+rng.normal(size=n)
        raw,status=conditional_correlation(score,coverage,adv,groups,'raw')
        self.assertEqual(status,'ok');self.assertGreater(raw,0)
        value,status=conditional_correlation(score,coverage,adv,groups,'both')
        self.assertEqual(status,'ok');self.assertTrue(-1<=value<=1)

    def test_small_one_group_and_singular(self):
        x=np.arange(30,dtype=float);industry=np.ones(30)
        self.assertEqual(conditional_correlation(x[:10],x[:10],x[:10]+1,industry[:10],'raw')[1],'too_few')
        self.assertEqual(conditional_correlation(x,x,x+1,industry,'industry')[1],'one_group')
        self.assertEqual(conditional_correlation(x,x,np.ones(30),industry,'adv')[1],'singular')
