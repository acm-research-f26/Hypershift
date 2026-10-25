"""Check paired inference on analytically known contrasts and family scope."""
import unittest
import numpy as np

from report_ndx_ranking import joint_intervals, contrast_family


class RankingInferenceTests(unittest.TestCase):
    def test_identical_paired_paths_have_exactly_zero_intervals(self):
        values = np.sin(np.arange(120))
        paths = {('a','x'):values,('b','x'):values.copy()}
        frame = joint_intervals(paths,[('identical',('a','x'),('b','x'))],[1,4,12],repeats=200)
        for field in ['estimate','paired_low','paired_high','simultaneous_low','simultaneous_high']:
            np.testing.assert_array_equal(frame[field],np.zeros(3))

    def test_constant_difference_recovers_known_annual_active_return(self):
        paths = {('a','x'):np.full(160,.001),('b','x'):np.zeros(160)}
        frame = joint_intervals(paths,[('constant',('a','x'),('b','x'))],[5,20,60],repeats=200,annualizer=252.)
        for field in ['estimate','paired_low','paired_high','simultaneous_low','simultaneous_high']:
            np.testing.assert_allclose(frame[field],.252,atol=1e-12,rtol=0)

    def test_contrast_families_have_only_declared_cells(self):
        keys = {(view,kind):None for view in ['daily_control','intraday_augmented']
                for kind in ['raw_mse','residual_mse','residual_rank']}
        keys.update({('control','SPY'):None,('control','QQQ'):None})
        rank,financial = contrast_family(keys),contrast_family(keys,True)
        self.assertEqual(len(rank),7)
        self.assertEqual(len(financial),19)
        self.assertEqual(sum(b[0]=='control' for _,_,b in financial),12)
        self.assertEqual(len(set(name for name,_,_ in financial)),19)


if __name__=='__main__':
    unittest.main()
