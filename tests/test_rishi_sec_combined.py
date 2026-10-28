import unittest
import pandas as pd
from rishi_sec_combined import combine


class SecCombinedTests(unittest.TestCase):
    def test_additions_do_not_override_existing_matches(self):
        base=pd.DataFrame({'year':[2024,2024,2024],'signal_date':['2024-01-05']*3,'security_id':['A','B','C'],'status':['matched','no_candidate','no_candidate'],'sic':[1111,None,None]})
        extra=pd.DataFrame({'year':[2024,2024],'signal_date':['2024-01-05']*2,'security_id':['B','C'],'status':['matched','no_current_candidate'],'sic':[2222,None]})
        z=combine(base,extra)
        self.assertEqual(z.source.tolist(),['original','supplement','unresolved'])
        self.assertEqual(z.combined_sic.iloc[:2].tolist(),[1111,2222])
        self.assertTrue(pd.isna(z.combined_sic.iloc[2]))

    def test_duplicates_and_unexpected_supplement_rejected(self):
        base=pd.DataFrame({'year':[2024],'signal_date':['2024-01-05'],'security_id':['A'],'status':['matched'],'sic':[1111]})
        with self.assertRaises(ValueError):combine(base,pd.concat([base,base]))
        with self.assertRaises(ValueError):combine(base,base)
