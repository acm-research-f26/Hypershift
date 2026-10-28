import unittest
import numpy as np
from rishi_simple_trace import paired_contributions,aggregate_issuers


class SimpleTraceTests(unittest.TestCase):
    def test_paired_contributions_use_each_accounts_normalized_returns(self):
        a=np.array([[.01,.02],[-.01,0.]])
        b=np.array([[.02,0.],[0.,.01]])
        n=paired_contributions(a,b)
        np.testing.assert_allclose(n,[-2.52,1.26])
        self.assertAlmostEqual(n.sum(),252*(a.sum(1)-b.sum(1)).mean())

    def test_issuer_aggregation_adds_share_classes(self):
        result=aggregate_issuers(['GOOG','GOOGL','A'],[1.,2.,-4.],{'GOOG':'ALPHABET','GOOGL':'ALPHABET'})
        self.assertEqual(result,{'ALPHABET':3.,'A':-4.})
