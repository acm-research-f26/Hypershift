import unittest
import numpy as np
from rishi_saved_industry import percentile,industry_masses


class SavedIndustryTests(unittest.TestCase):
    def test_percentile_respects_eligibility_and_average_ties(self):
        result=percentile(np.array([1.,1.,3.,999.]),np.array([1,1,1,0],bool))
        np.testing.assert_allclose(result[:3],[.5,.5,1.]);self.assertTrue(np.isnan(result[3]))

    def test_unknown_industry_mass_is_retained(self):
        result=industry_masses(np.array([.3,.2,.5]),np.array([1.,np.nan,1.]))
        self.assertAlmostEqual(result['1'],.8);self.assertAlmostEqual(result['unclassified'],.2)
        self.assertAlmostEqual(sum(result.values()),1.)
