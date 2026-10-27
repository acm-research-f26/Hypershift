import unittest
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import calibrate_fixed_exposure,fee_view,calendar_attribution


class MechanismRiskTests(unittest.TestCase):
    def test_calibration_uses_only_training_windows(self):
        rng=np.random.default_rng(42);returns=rng.normal(0,.02,(150,2));origins=np.arange(20,100)
        first=calibrate_fixed_exposure(returns,origins,np.array([.5,.5]))
        changed=returns.copy();changed[100:]=99.
        second=calibrate_fixed_exposure(changed,origins,np.array([.5,.5]))
        self.assertEqual(first,second);self.assertFalse(first['fallback'])
        self.assertLessEqual(first['exposure'],1)

    def test_new_entry_insufficient_history_has_declared_fallback(self):
        r=np.ones((60,2))*.01;r[:50,1]=np.nan
        result=calibrate_fixed_exposure(r,np.arange(20,60),np.array([.5,.5]))
        self.assertTrue(result['fallback']);self.assertEqual(result['risk'],.3)
        self.assertAlmostEqual(result['exposure'],2/3)

    def test_flat_fee_view_restores_engine_after_exception(self):
        original=engine.cost_rate
        with self.assertRaisesRegex(RuntimeError,'test'):
            with fee_view('flat20') as multiplier:
                np.testing.assert_allclose(engine.cost_rate(np.array([-100.,50.]),np.array([1e6,1e7]),multiplier),.002)
                raise RuntimeError('test')
        self.assertIs(engine.cost_rate,original)
        with fee_view('zero') as multiplier:self.assertEqual(float(engine.cost_rate(100.,1e6,multiplier)),0)

    def test_calendar_return_is_exact_and_log_growth_adds(self):
        dates=pd.to_datetime(['2022-12-29','2022-12-30','2023-01-03'])
        result=calendar_attribution(np.array([.01,.01,.02]),dates)
        self.assertAlmostEqual(result.iloc[0].calendar_return,1.01**2-1)
        self.assertAlmostEqual(result.log_growth.sum(),2*np.log1p(.01)+np.log1p(.02))


if __name__=='__main__':unittest.main()
