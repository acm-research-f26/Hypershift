import unittest
import numpy as np
from report_ndx_seed_expansion import seed_time_intervals


class SeedTimeTests(unittest.TestCase):
    def test_zero_differences_are_zero_and_model_variation_is_not_discarded(self):
        n=40;zero=np.zeros(n);paths={};boot={}
        for policy in ['full','trend_half']:
            paths[policy,'all_twenty_five']=zero;paths[policy,'original_five']=zero;boot[policy]=np.zeros((4,n))
        paths['full','SPY']=zero;paths['full','QQQ']=zero
        first=seed_time_intervals(boot,paths,blocks=[5],repeats=200)
        self.assertTrue((first.pointwise_low==0).all());self.assertTrue((first.pointwise_high==0).all())
        for policy in boot:boot[policy][:2]=.01;boot[policy][2:]=-.01
        second=seed_time_intervals(boot,paths,blocks=[5],repeats=200)
        self.assertTrue((second.pointwise_low<0).all());self.assertTrue((second.pointwise_high>0).all())

    def test_reproducible_and_paired_control_cancels_shared_market_path(self):
        rng=np.random.default_rng(7);q=rng.normal(0,.01,40);paths={};boot={}
        for policy in ['full','trend_half']:
            paths[policy,'all_twenty_five']=q+.001;paths[policy,'original_five']=q;boot[policy]=np.tile(q+.001,(4,1))
        paths['full','SPY']=q;paths['full','QQQ']=q
        a=seed_time_intervals(boot,paths,blocks=[5],repeats=200);b=seed_time_intervals(boot,paths,blocks=[5],repeats=200)
        np.testing.assert_allclose(a.pointwise_low,.252,atol=1e-12);np.testing.assert_array_equal(a.pointwise_low,b.pointwise_low)


if __name__=='__main__':unittest.main()
