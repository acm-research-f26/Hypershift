import unittest
import numpy as np
from rishi_industry_accounting import active_snapshot,partition


class IndustryAccountingTests(unittest.TestCase):
    def test_first_signal_day_uses_prior_snapshot(self):
        dates=np.array(['2021-12-31','2023-01-06'],dtype='datetime64[D]')
        self.assertEqual(active_snapshot(dates,np.datetime64('2023-01-06')) ,0)
        self.assertEqual(active_snapshot(dates,np.datetime64('2023-01-09')) ,1)
        self.assertEqual(active_snapshot(dates,np.datetime64('2021-12-31')) ,-1)

    def test_fees_and_unclassified_partition_exactly(self):
        x=np.array([[1.,2.,-1.],[3.,-.5,2.]])
        groups=np.array([[0,1,2],[2,1,0]])
        out=partition(x,groups)
        np.testing.assert_allclose(out,[[1,2,-1],[2,-.5,3]])
        np.testing.assert_allclose(out.sum(1),x.sum(1))
