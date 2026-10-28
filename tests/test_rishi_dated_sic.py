import unittest
import numpy as np
from rishi_dated_sic import match_sic, between_fraction


class DatedSicTests(unittest.TestCase):
    def test_filing_must_precede_day_and_segment_must_cover_day(self):
        mapping={'X':{'segments':[['2020-01-01','2023-12-31',11]]}}
        sic={'11':[1234,'2022-01-03']}
        self.assertEqual(match_sic('X','2022-01-03',mapping,sic)[0],'not_yet_filed')
        self.assertEqual(match_sic('X','2022-01-04',mapping,sic),('matched',11,1234,'2022-01-03'))
        self.assertEqual(match_sic('X','2024-01-01',mapping,sic)[0],'no_covering_segment')

    def test_overlapping_segments_are_never_guessed(self):
        mapping={'X':{'segments':[['2020-01-01','2022-12-31',11],['2022-01-01','2023-12-31',12]]}}
        self.assertEqual(match_sic('X','2022-02-01',mapping,{})[0],'ambiguous_segments')

    def test_between_and_within_partition(self):
        f,between,within,total=between_fraction(np.array([0.,2.,4.,6.]),np.array([1,1,2,2]))
        self.assertAlmostEqual(f,.8)
        self.assertAlmostEqual(between+within,total)
        self.assertTrue(np.isnan(between_fraction(np.ones(3),np.array([1,1,2]))[0]))


if __name__=='__main__':unittest.main()
