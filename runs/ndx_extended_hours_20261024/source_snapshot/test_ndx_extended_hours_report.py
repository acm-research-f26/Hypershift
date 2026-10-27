import unittest
from unittest.mock import patch
import numpy as np

from prepare_ndx_extended_hours import summarize,FIELDS
from report_ndx_extended_hours import reference_summary,forecast_maps
from test_ndx_extended_hours import bars,calendar


class IndependentFeatureTests(unittest.TestCase):
    def test_scalar_reference_matches_early_close_ah_and_premarket(self):
        fixtures=[('after_hours','2024-11-29',__import__('pandas').concat([
            bars('2024-11-29','12:45',1),bars('2024-11-29','13:00',28)],ignore_index=True)),
            ('premarket','2024-12-02',bars('2024-12-02','04:00',24))]
        for segment,date,frame in fixtures:
            expected=summarize(frame,calendar(),segment).iloc[0]
            values,count=reference_summary(frame,calendar(),date,segment)
            np.testing.assert_allclose(values,expected[FIELDS].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
            self.assertEqual(count,expected.observed)

    def test_empty_or_sparse_sessions_never_invent_a_return(self):
        frame=bars('2024-12-02','04:00',3)
        values,count=reference_summary(frame,calendar(),'2024-12-02','premarket')
        self.assertEqual(count,3);self.assertTrue(np.isnan(values[:7]).all())
        frame.volume=0
        values,count=reference_summary(frame,calendar(),'2024-12-02','premarket')
        np.testing.assert_allclose(values[7:],[0,1,1]);self.assertEqual(count,0)


class EnsembleTests(unittest.TestCase):
    def test_fixed_seed_groups_and_test_outcome_independence(self):
        seeds=[7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941]
        plan=dict(seeds=seeds,test_years=[2022],views=['daily','after_hours','after_hours_premarket'])
        data=dict(mask=np.ones((2,4),bool),y=np.zeros((2,4)))
        class Saved:
            def __init__(self,path):self.seed=int(path.name.split('_')[0])
            def __enter__(self):return dict(test=np.tile(np.arange(4) if self.seed in seeds[:5] else -np.arange(4),(2,1)))
            def __exit__(self,*args):return False
        with patch('report_ndx_extended_hours.fold',return_value=(np.array([],int),np.array([],int),np.array([0,1]))),patch('report_ndx_extended_hours.np.load',side_effect=Saved):
            _,_,before=forecast_maps(data,plan)
            data['y'][:]=1e20
            _,_,after=forecast_maps(data,plan)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_allclose(before['daily','ten_seed'],.625)
        np.testing.assert_allclose(before['daily','original_five'][0],[.25,.5,.75,1.])


if __name__=='__main__':unittest.main()
