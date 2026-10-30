import unittest
import numpy as np
from rishi_traded_boundary import partition_account


class TradedBoundaryTests(unittest.TestCase):
    def test_old_position_profit_new_purchase_fee_and_exit_fee_are_separate(self):
        days=np.array([1,2]);decisions={1:np.array([0,5]),2:np.array([5,2])}
        units=np.array([[1.,0.],[0.,1.]]);marks=np.array([[100.,100.],[110.,100.]])
        gross=np.array([[0.,0.],[10.,0.]]);fees=np.array([[1.,0.],[2.,1.]])
        ops=[dict(day=1,security_id='A',kind='fill',fee=1.),dict(day=2,security_id='A',kind='fill',fee=2.),dict(day=2,security_id='B',kind='fill',fee=1.)]
        result=partition_account(days,decisions,units,marks,np.array([1000.,999.]),np.array([999.,1006.]),gross,fees,ops,['A','B'])
        self.assertAlmostEqual(result['gross'][1,0],10/999)
        self.assertAlmostEqual(result['cost'][1,4],2/999)
        self.assertAlmostEqual(result['cost'][1,2],1/999)
        self.assertAlmostEqual(result['exposure'][1,2],100/1006)

    def test_terminal_and_settlement_fees_use_prior_holding_category(self):
        args=dict(days=np.array([1,2]),decisions={1:np.array([1]),2:np.array([3])},
            units=np.array([[1.],[0.]]),marks=np.array([[100.],[110.]]),old=np.array([1000.,1000.]),nav=np.array([1000.,1009.]),
            gross=np.array([[0.],[10.]]),fees=np.array([[0.],[1.]]),security_ids=['A'])
        for kind in ['terminal','settlement']:
            result=partition_account(**args,operations=[dict(day=2,security_id='A',kind=kind,fee=1.)])
            self.assertAlmostEqual(result['cost'][1,1],.001)
            self.assertEqual(result['cost'][1,3],0)


if __name__=='__main__':unittest.main()
