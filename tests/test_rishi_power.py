import unittest
import numpy as np
from rishi_mechanism_power import stationary_draws,holm_rejections,upper_p,planning_tape,sample_statistics


class PowerTests(unittest.TestCase):
    def test_stationary_draws_reproduce_and_keep_contiguous_blocks(self):
        a=stationary_draws(7,8,12,float('inf'),np.random.default_rng(7))
        b=stationary_draws(7,8,12,float('inf'),np.random.default_rng(7))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,1:],(a[:,:-1]+1)%7)

    def test_holm_uses_two_contrast_family_and_restores_order(self):
        p=np.array([[.04,.01],[.01,.04],[.03,.04],[.001,.2]])
        np.testing.assert_array_equal(holm_rejections(p),[[1,1],[1,1],[0,0],[1,0]])

    def test_empirical_upper_p_includes_ties_and_plus_one(self):
        np.testing.assert_allclose(upper_p(np.array([1.,2.,2.,3.]),np.array([0.,2.,4.])),[1.,.8,.2])

    def test_planning_calibration_and_paired_identity(self):
        rng=np.random.default_rng(9);x=rng.normal(0,.01,100);raw=np.column_stack([x,x,2*x])
        tape=planning_tape(raw)
        rf=(1.03**(1/252)-1)
        np.testing.assert_allclose((tape.mean(0)-rf)*np.sqrt(252)/tape.std(0,ddof=1),.5)
        stats=sample_statistics(tape,60,10,5,np.random.default_rng(4))
        np.testing.assert_allclose(stats['mean'][:,0],0,atol=1e-12)
        np.testing.assert_allclose(stats['sr'][:,0],0,atol=1e-12)
        self.assertTrue((stats['left_sd_ratio']>0).all())


if __name__=='__main__':unittest.main()
