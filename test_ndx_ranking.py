"""Economic target timing, causal betas, ranking gradients and checkpoint selection."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import ndx_ranking_objectives as module


class RankingTests(unittest.TestCase):
    def test_beta_recovers_paired_ols_and_shrinks_toward_one(self):
        q = np.sin(np.arange(180))*.01
        r = np.column_stack([.001+1.6*q, -.002+.4*q])
        r[:80,1] = np.nan
        beta,support = module.lagged_betas(r,q)
        np.testing.assert_allclose(beta[-1],[1.3,.7],atol=1e-12)
        self.assertTrue(support[-1].all())
        self.assertFalse(support[100,1])
        self.assertEqual(beta[100,1],1.)

    def test_future_returns_cannot_change_past_betas(self):
        rng = np.random.default_rng(19)
        r,q = rng.normal(0,.01,(240,3)),rng.normal(0,.01,240)
        first = module.lagged_betas(r,q)
        r[120:]=1000; q[120:]=-1000
        second = module.lagged_betas(r,q)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a[:120],b[:120])

    def test_beta_clip_and_no_variance_fallback(self):
        q = np.sin(np.arange(140))*.01
        b,_ = module.lagged_betas(np.column_stack([10*q,-10*q]),q)
        np.testing.assert_array_equal(b[-1],[2.,0.])
        b,s = module.lagged_betas(np.ones((140,1)),np.zeros(140))
        np.testing.assert_array_equal(b,np.ones_like(b))
        self.assertFalse(s.any())

    def test_residual_uses_matching_entry_exit_benchmark(self):
        base = Path(__file__).resolve().parent/'tmp'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(base.resolve()))
            dates = pd.date_range('2020-01-01',periods=6)
            q = np.array([100.,110.,121.,100.,110.,120.])
            pd.DataFrame({'QQQ':q},index=dates).to_csv(root/'etf_adj_close.csv')
            data = dict(manifest={'root':str(root)},panel=SimpleNamespace(dates=dates),
                returns=np.zeros((6,2)),origin=np.array([1,2]),entry=np.array([2,3]),exit=np.array([4,5]),
                y=np.array([[.1,-.2],[.3,-.1]]))
            target = module.target_data(data)
            expected = np.array([q[4]/q[2]-1,q[5]/q[3]-1])
            np.testing.assert_allclose(target['benchmark'],expected)
            np.testing.assert_allclose(target['residual'],100*(data['y']-expected[:,None]))

    def test_rankic_handles_ties_constants_and_missing_labels(self):
        p = np.array([[1,1,3,999],[4,4,4,999]],dtype=float)
        y = np.array([[2,2,5,-999],[1,2,3,-999]],dtype=float)
        mask = np.array([[1,1,1,0],[1,1,1,0]],dtype=bool)
        np.testing.assert_allclose(module.rank_ic(p,y,mask),[1.,0.])

    def test_pairwise_loss_rewards_correct_order_and_masks_inactive_gradient(self):
        y = torch.tensor([[1.,-1.,1000.]])
        mask = torch.tensor([[True,True,False]])
        good = torch.tensor([[1.,-1.,-1e6]],requires_grad=True)
        wrong = torch.tensor([[-1.,1.,1e6]])
        goodloss = module.training_loss(good,y,mask,'residual_rank')
        self.assertLess(float(goodloss.detach()),float(module.training_loss(wrong,y,mask,'residual_rank')))
        goodloss.backward()
        self.assertEqual(float(good.grad[0,2]),0.)
        unranked = module.training_loss(good,y,mask,'residual_mse')
        self.assertAlmostEqual(float(unranked.detach()),0.)
        changed = y.clone(); changed[0,2]=-1000.
        self.assertAlmostEqual(float(module.training_loss(good,changed,mask,'residual_rank').detach()),float(goodloss.detach()))

    def test_restore_earliest_best_ranking_checkpoint_recomputes_exports(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-1.))
            def forward(self,x,eligible):
                return x[:,0,:,0]*self.weight,None
        x = np.zeros((3,1,3,13),dtype='float32')
        x[:,0,:,0] = [1,2,3]
        labels = np.tile(np.array([1,2,3],dtype='float32'),(3,1))
        data = dict(panel=SimpleNamespace(tickers=['a','b','c']),mask=np.ones((3,3),bool),labelmask=np.ones((3,3),bool))
        plan = dict(lr=.2,weight_decay=0.,epochs=5,patience=2,batch=1,ranking_weight=1.,ranking_temperature_pp=1.,gradient_norm_cap=1.)
        base = Path(__file__).resolve().parent/'tmp'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            folder = Path(temporary).resolve()
            self.assertTrue(folder.is_relative_to(base.resolve()))
            with patch.object(module,'make',return_value=Tiny()),patch.object(module,'configure_device',return_value=torch.device('cpu')):
                vp,tp,record = module.fit(data,x,dict(raw=labels,residual=-labels),np.array([0]),np.array([1]),np.array([2]),
                                          [{0},{1},{2}],'raw_mse',7,plan,folder)
            self.assertEqual(record['chosen_epoch'],1)
            self.assertEqual(record['epochs_run'],3)
            np.testing.assert_allclose(vp,[[-.8,-1.6,-2.4]],atol=1e-6)
            np.testing.assert_array_equal(vp,tp)


if __name__=='__main__':
    unittest.main()
