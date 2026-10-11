import unittest
import numpy as np
import pandas as pd
from historical_nasdaq_universe import rank_months,membership_on_signals


class HistoricalUniverseTests(unittest.TestCase):
    def setUp(self):
        self.calendar=pd.bdate_range('2021-12-01','2022-03-01')
        self.rows=pd.DataFrame([dict(formation_date=date,company_id=company,security_id=security,
            ticker=security,exchange='XNAS',security_type='CS',company_market_cap_usd=cap,
            adv20_usd=adv,cap_available_date=date,listing_available_date=date,
            identity_available_date=date,source='synthetic test')
            for date,companies in [('2021-12-31',[('A','A1',100,20),('A','A2',100,10),('B','B1',90,15),('C','C1',80,10)]),
                                   ('2022-01-31',[('A','A1',100,20),('A','A2',100,10),('B','B1',90,15),('C','C1',1000,10)])]
            for company,security,cap,adv in companies])
        self.coverage=pd.DataFrame([dict(formation_date=date,expected_security_count=4,
            historical_listing_complete=True,missing_company_caps=0,
            publication_timing_reviewed=True,issuer_identity_reviewed=True)
            for date in ['2021-12-31','2022-01-31']])

    def rank(self,rows=None,coverage=None):
        return rank_months(self.rows if rows is None else rows,self.coverage if coverage is None else coverage,
                          self.calendar,['2022-01','2022-02'],count=2)

    def test_future_winner_not_used_in_prior_month(self):
        members=self.rank()
        self.assertEqual(members.query('month=="2022-01"').security_id.tolist(),['A1','B1'])
        self.assertEqual(members.query('month=="2022-02"').security_id.tolist(),['C1','A1'])
        changed=self.rows.copy()
        changed.loc[(changed.formation_date=='2022-01-31')&(changed.company_id=='C'),'company_market_cap_usd']=1
        pd.testing.assert_frame_equal(members.query('month=="2022-01"'),self.rank(changed).query('month=="2022-01"'))

    def test_duplicate_classes_are_not_duplicate_companies(self):
        self.assertFalse(self.rank().groupby('month').company_id.apply(lambda x:x.duplicated().any()).any())
        changed=self.rows.copy();changed.loc[0,'company_market_cap_usd']=110
        with self.assertRaisesRegex(ValueError,'reconciled COMPANY'):self.rank(changed)

    def test_unknown_filing_fact_rejected(self):
        changed=self.rows.copy();changed.loc[0,'cap_available_date']='2022-01-10'
        with self.assertRaisesRegex(ValueError,'future cap_available_date'):self.rank(changed)

    def test_missing_candidate_cannot_silently_change_ranking(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):self.rank(self.rows.drop(index=0))
        changed=self.rows.copy();changed.loc[3,'company_market_cap_usd']=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid'):self.rank(changed)

    def test_dates_and_identity_gate_before_next_close_orders(self):
        members=self.rank();ids=['A1','A2','B1','C1']
        mask=membership_on_signals(members,ids,['2022-01-03','2022-01-31'],['2022-01-04','2022-02-01'])
        np.testing.assert_array_equal(mask,[[1,0,1,0],[1,0,0,1]])
        with self.assertRaisesRegex(ValueError,'Unknown-at-signal'):
            membership_on_signals(members,ids,['2022-01-28'],['2022-02-01'])
        with self.assertRaisesRegex(ValueError,'missing from panel'):
            membership_on_signals(members,ids[:-1],['2022-01-31'],['2022-02-01'])

    def test_unverified_inventory_and_missing_month_rejected(self):
        changed=self.coverage.copy();changed.loc[0,'historical_listing_complete']=False
        with self.assertRaisesRegex(ValueError,'not verified'):self.rank(coverage=changed)
        with self.assertRaisesRegex(ValueError,'Missing complete'):self.rank(coverage=self.coverage.iloc[:1])


if __name__=='__main__':unittest.main()
