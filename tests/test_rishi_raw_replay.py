import unittest
import numpy as np
import torch
import tempfile
import pandas as pd
from pathlib import Path
from rishi_raw_replay import persistent_mean, neutralize_missing, fixed_ridge,load_selected_checkpoint,forecast_diagnostics,check_replayed_account


class RawReplayControlTests(unittest.TestCase):
    def test_persistent_mean_uses_mask_counts_and_training_median(self):
        pred=np.array([[2.,999.,4.],[4.,999.,8.],[6.,10.,12.]])
        mask=np.array([[1,0,1],[1,0,1],[1,1,0]],bool)
        score,count=persistent_mean(pred,mask,minimum=2)
        np.testing.assert_array_equal(count,[3,1,2])
        np.testing.assert_allclose(score,[4.,5.,6.])

    def test_no_qualifying_training_ids_is_unavailable(self):
        with self.assertRaises(ValueError):persistent_mean(np.zeros((2,3)),np.ones((2,3),bool),minimum=3)

    def test_missing_comparator_is_neutral_without_changing_eligibility(self):
        x=np.array([[np.nan,1.,3.,999.],[np.nan,np.nan,np.nan,999.]])
        mask=np.array([[1,1,1,0],[1,1,1,0]],bool)
        y,missing=neutralize_missing(x,mask)
        np.testing.assert_allclose(y,[[2.,1.,3.,0.],[0.,0.,0.,0.]])
        self.assertEqual(missing,4)

    def test_ridge_uses_only_observed_training_rows(self):
        x=np.arange(12,dtype=float).reshape(6,2)
        y=3*x[:,0]+1
        mask=np.array([1,1,1,1,0,0],bool)
        a=fixed_ridge(x,y,mask)
        y[~mask]=1e12
        b=fixed_ridge(x,y,mask)
        np.testing.assert_allclose(a,b)
        self.assertEqual(a.shape,(3,))

    def test_checkpoint_preserves_numpy_integer_group_metadata(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            path=Path(folder)/'selected.pt'
            torch.save({'groups':[[np.int64(1),np.int64(2)]],'state_dict':{'weight':torch.tensor([3.])}},path)
            loaded=load_selected_checkpoint(path)
            self.assertEqual(loaded['groups'],[[1,2]])
            torch.testing.assert_close(loaded['state_dict']['weight'],torch.tensor([3.]))

    def test_rank_diagnostic_finds_boundary_changes(self):
        expected=np.arange(12,dtype=float)[None,:]
        actual=expected.copy();actual[:,[6,7]]=actual[:,[7,6]]
        result=forecast_diagnostics(actual,expected,np.ones_like(actual,bool))
        self.assertEqual(result['top5_changed_origins'],1)
        self.assertEqual(result['top10_changed_origins'],0)
        self.assertLess(result['mean_rank_correlation'],1.)

    def test_account_gate_checks_nav_and_metrics_beyond_daily_return(self):
        curve=pd.DataFrame({'date':['2022-01-03','2022-01-04'],'net':[0.,.01]})
        ledger=pd.DataFrame({'date':curve.date,'nav':[100.,101.],'cash':[0.,0.]})
        result=check_replayed_account({'sharpe':.5},curve,ledger,{'sharpe':.5},curve,ledger,100.)
        self.assertEqual(result['max_relative_nav_error'],0.)
        with self.assertRaises(AssertionError):
            check_replayed_account({'sharpe':.6},curve,ledger,{'sharpe':.5},curve,ledger,100.)
        bad=ledger.copy();bad['nav']+=1.
        with self.assertRaises(AssertionError):
            check_replayed_account({'sharpe':.5},curve,bad,{'sharpe':.5},curve,ledger,100.)


if __name__=='__main__':unittest.main()
