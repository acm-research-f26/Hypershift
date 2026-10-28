import unittest
import numpy as np
from rishi_training_issuer_exclusion import selected_issuers


class TrainingIssuerTests(unittest.TestCase):
    def test_issuer_dedup_support_and_ties(self):
        tickers=['A1','A2','B','C','D','E'];issuers=['A','A','B','C','D','E']
        scores=np.array([.9,.8,.9,.7,1.,.6]);eligible=np.array([1,1,1,1,1,1],bool);counts=np.array([30,40,31,35,29,45])
        result=selected_issuers(tickers,issuers,scores,eligible,counts)
        self.assertEqual([x[0] for x in result],['A','B','C'])
        self.assertEqual(result[0][2],'A1')

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):selected_issuers(['A'],['A'],[1.],[True],[40])
        with self.assertRaises(ValueError):selected_issuers(['A','B','C'],['A','B','C'],[np.nan,1.,2.],[True]*3,[40]*3)
