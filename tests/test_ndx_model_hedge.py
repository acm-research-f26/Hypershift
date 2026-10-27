import unittest
import numpy as np
from ndx_model_hedge import validation_bias,forecast_signals


class ModelHedgeTests(unittest.TestCase):
    def test_bias_is_equal_date_and_uses_valid_labels_only(self):
        p=np.array([[2.,100.,np.nan],[4.,4.,np.nan]])
        y=np.zeros((2,3));m=np.array([[1,0,0],[1,1,0]],bool)
        self.assertEqual(validation_bias(p,y,m),3.)
        p[~m]=1e9
        self.assertEqual(validation_bias(p,y,m),3.)

    def test_negative_basket_and_market_can_disagree(self):
        p=np.array([[-2.,-2.,5.,np.nan]])
        m=np.array([[1,1,1,0]],bool);w=np.array([[0.,0.,1.,0.]])
        f,d=forecast_signals([p,p],m,w,[0,0])
        self.assertFalse(f['raw_mean'][0]);self.assertFalse(f['raw_basket'][0])
        self.assertTrue(f['raw_breadth60'][0]);self.assertAlmostEqual(d['raw_basket_pp'][0],5.)

    def test_calibration_is_family_specific_and_zero_boundary_is_clear(self):
        a=np.array([[1.,1.],[0.,0.]]);b=np.array([[-2.,-2.],[0.,0.]])
        f,_=forecast_signals([a,b],np.ones((2,2),bool),np.full((2,2),.5),[2.,0.])
        self.assertTrue(f['raw_mean'][0]);self.assertFalse(f['raw_agreement'][0])
        self.assertTrue(f['cal_agreement'][0]);self.assertFalse(f['raw_mean'][1])

    def test_future_predictions_cannot_change_earlier_signals(self):
        rng=np.random.default_rng(7);p=rng.normal(size=(40,5))
        m=np.ones_like(p,bool);w=np.full_like(p,.2)
        first,_=forecast_signals([p,p],m,w,[.1,.2])
        q=p.copy();q[21:]=1000
        second,_=forecast_signals([q,q],m,w,[.1,.2])
        for name in first:np.testing.assert_array_equal(first[name][:21],second[name][:21])


if __name__=='__main__':unittest.main()
