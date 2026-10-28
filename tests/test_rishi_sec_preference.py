import unittest
import pandas as pd
from rishi_sec_preference import checked_join


class SecPreferenceTests(unittest.TestCase):
    def test_exact_keys_and_missing_status_are_retained(self):
        keys={'year':[2024,2024],'signal_date':['2024-01-05']*2,'security_id':['A','B']}
        scores=pd.DataFrame(dict(**keys,hlabel_percentile=[.9,.1]))
        receipts=pd.DataFrame(dict(**keys,status=['matched','no_candidate'],sic=[3571,None]))
        joined=checked_join(scores,receipts)
        self.assertEqual(joined.status.tolist(),['matched','no_candidate'])
        self.assertTrue(pd.isna(joined.loc[1,'sic']))

    def test_missing_or_duplicate_receipt_fails(self):
        scores=pd.DataFrame({'year':[2024],'signal_date':['2024-01-05'],'security_id':['A']})
        with self.assertRaises(ValueError):checked_join(scores,pd.concat([scores,scores]))
        with self.assertRaises(ValueError):checked_join(scores,scores.iloc[:0])
