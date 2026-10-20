import unittest
import numpy as np
from compare_ndx_strategies_20261020 import bootstrap_sharpe,raw_rankic,model_label,RF


class ComparisonTests(unittest.TestCase):
    def test_model_labels_distinguish_ungated_corrections_from_broad_models(self):
        self.assertIn('ungated',model_label('ungated__intraday_window__net_utility'))
        self.assertEqual(model_label('daily__raw__ridge'),'Ridge (daily, raw)')

    def test_block_moments_exactly_match_materialized_daily_samples(self):
        rng=np.random.default_rng(7);a=rng.normal(.001,.02,(103,4));seed=42
        for block in [1,5,20,60]:
            actual=bootstrap_sharpe(a,block=block,repeats=137,seed=seed)
            generator=np.random.default_rng(seed);expected=[]
            for first in range(0,137,100):
                count=min(100,137-first);starts=generator.integers(0,len(a),(count,int(np.ceil(len(a)/block))))
                indices=((starts[:,:,None]+np.arange(block))%len(a)).reshape(count,-1)[:,:len(a)]
                draw=a[indices]
                expected.append(np.sqrt(252)*(draw.mean(1)-((1+RF)**(1/252)-1))/draw.std(1,ddof=1))
            np.testing.assert_allclose(actual,np.concatenate(expected),atol=1e-12,rtol=1e-12)

    def test_identical_and_positive_scaled_excess_paths_have_identical_sr_samples(self):
        rng=np.random.default_rng(19);a=rng.normal(.001,.02,103);daily=(1+RF)**(1/252)-1
        values=np.column_stack([a,a,daily+3*(a-daily)])
        result=bootstrap_sharpe(values,repeats=200)
        np.testing.assert_allclose(result[:,0],result[:,1],atol=1e-12)
        np.testing.assert_allclose(result[:,0],result[:,2],atol=1e-12)

    def test_rankic_uses_cross_section_observed_labels_and_constant_is_undefined(self):
        p=np.array([[1.,2.,3.,1000.],[3.,2.,1.,1000.]])
        y=np.array([[1.,2.,3.,-1000.],[1.,2.,3.,-1000.]])
        mask=np.array([[1,1,1,0],[1,1,1,0]],bool)
        self.assertAlmostEqual(raw_rankic(p,y,mask),0.)
        self.assertIsNone(raw_rankic(np.ones_like(p),y,mask))


if __name__=='__main__':unittest.main()
