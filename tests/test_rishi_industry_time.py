import unittest
import numpy as np
from rishi_industry_time import frozen_labels
from rishi_saved_industry import industry_masses


class IndustryTimeTests(unittest.TestCase):
    def test_new_id_is_not_given_a_future_classification(self):
        value=frozen_labels(['A','B','C'],{'A':6,'B':float('nan')})
        self.assertEqual(value[0],6);self.assertTrue(np.isnan(value[1]));self.assertTrue(np.isnan(value[2]))

    def test_unclassified_mass_is_never_lost(self):
        self.assertAlmostEqual(industry_masses(np.array([.4,.6]),np.array([6.,np.nan]))['unclassified'],.6)
