import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import calibrate_fixed_exposure,fee_view,calendar_attribution


class MechanismRiskTests(unittest.TestCase):
    def test_path_axes_load_without_pickle(self):
        from rishi_mechanism_risk import write_paths
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'paths.npz'
            write_paths(path,{'A':np.array([.01,.02])},pd.date_range('2022-01-03',periods=2))
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(saved['dates'].tolist(),['2022-01-03','2022-01-04'])
                self.assertEqual(saved['ids'].tolist(),['A'])
                np.testing.assert_allclose(saved['returns'][:,0],[.01,.02])

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

    def test_vol_target_cannot_prevent_a_gap_already_held(self):
        dates=pd.bdate_range('2022-01-03',periods=35);marks=np.full((35,1),100.);marks[20:]=50.
        returns=np.zeros_like(marks);returns[20]=-.5
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A']),marks=marks,returns=returns,
            tradable=np.ones_like(marks,bool),adv=np.full_like(marks,1e9),events=[],
            origin=np.arange(0,31,5),entry=np.arange(1,32,5))
        weights=np.ones((7,1));risk=engine.risk_inputs(data,engine.daily_allocation(data,np.arange(7),weights))
        exposure=engine.exposure_series(dict(vol='basket',window=20,target=.2),{'qqq':np.ones(35)},risk)
        _,path,detail=engine.simulate(data,np.arange(7),weights,exposure,detailed=True)
        self.assertAlmostEqual(exposure[15],2/3) # Source .30 warmup risk, weekly decision.
        self.assertAlmostEqual(exposure[19],1.) # No weekly order at this day.
        self.assertAlmostEqual(path[19],-1/3,places=5)
        self.assertGreater(detail.equity_exposure.iloc[19],.49)
        self.assertLess(detail.equity_exposure.iloc[20],.2)

    def test_new_selected_stock_missing_fill_cancels_without_reranking(self):
        dates=pd.bdate_range('2022-01-03',periods=10);marks=np.full((10,2),100.)
        tradable=np.ones_like(marks,bool);tradable[6,1]=False
        data=dict(panel=SimpleNamespace(dates=dates,tickers=['A','B']),marks=marks,tradable=tradable,
            adv=np.full_like(marks,1e9),events=[],origin=np.array([0,5]),entry=np.array([1,6]))
        _,_,detail=engine.simulate(data,np.array([0,1]),np.array([[1.,0.],[0.,1.]]),np.ones(10),detailed=True)
        self.assertGreater(detail.equity_exposure.iloc[4],.99)
        self.assertEqual(detail.equity_exposure.iloc[5],0)
        self.assertEqual(detail.equity_exposure.iloc[6],0)


if __name__=='__main__':unittest.main()
