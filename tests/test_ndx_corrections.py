import unittest
import numpy as np
import torch

from ndx_corrections import Correction,correction_loss,choose,accepted,proxy,partitions,prepared,normalized
from run_ndx_corrections import candidate_score


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x=torch.randn(4,14,6);self.core=torch.randn(4,14,requires_grad=True)
        self.mask=torch.ones(4,14,dtype=torch.bool);self.mask[:,-1]=False

    def test_zero_head_exactly_preserves_core_and_core_gets_no_gradient(self):
        model=Correction(6);score,delta,_=model(self.x,self.core,self.mask)
        torch.testing.assert_close(score[self.mask],self.core[self.mask],rtol=0,atol=0)
        self.assertEqual(float(delta.detach().abs().sum()),0.)
        score.sum().backward();self.assertIsNone(self.core.grad)

    def test_inactive_nan_and_permutation_invariance(self):
        model=Correction(6);torch.nn.init.normal_(model.last.weight)
        first=model(self.x,self.core,self.mask)[0]
        self.x[:,-1]=float('nan');self.core=self.core.detach();self.core[:,-1]=float('nan')
        second,delta,_=model(self.x,self.core,self.mask)
        torch.testing.assert_close(first,second)
        torch.testing.assert_close(delta.sum(1),torch.zeros(4),atol=1e-6,rtol=0)
        order=torch.randperm(14)
        third=model(self.x[:,order],self.core[:,order],self.mask[:,order])[0]
        torch.testing.assert_close(second[:,order],third,atol=1e-6,rtol=0)

    def test_loss_prefers_correct_order(self):
        target=torch.arange(14).float()[None,:].repeat(4,1)
        zero=torch.zeros_like(target);scale=torch.ones(4,1)
        good=correction_loss(target,zero,scale,target,self.mask,self.mask,target,'top_rank')
        bad=correction_loss(-target,zero,scale,target,self.mask,self.mask,target,'top_rank')
        self.assertLess(good,bad)

    def test_utility_excludes_incomplete_return_origins(self):
        model=Correction(6);score,delta,scale=model(self.x,self.core,self.mask)
        observed=self.mask.clone();observed[1,2]=False
        active=torch.randn(4,14)
        first=correction_loss(score,delta,scale,active,self.mask,observed,active,'net_utility')
        active[1]=1e20
        second=correction_loss(score,delta,scale,active,self.mask,observed,active,'net_utility')
        torch.testing.assert_close(first,second)
        first.backward();self.assertTrue(torch.isfinite(model.last.weight.grad).all())

    def test_no_observed_utility_batch_is_finite_differentiable(self):
        model=Correction(6);score,delta,scale=model(self.x,self.core,self.mask)
        loss=correction_loss(score,delta,scale,self.core.detach(),self.mask,torch.zeros_like(self.mask),self.core.detach(),'net_utility')
        self.assertTrue(torch.isfinite(loss));loss.backward()


class GateTests(unittest.TestCase):
    def test_zero_always_available_and_ties_preserve_first(self):
        self.assertIsNone(choose([1],[dict(score=-1)])['candidate'])
        self.assertIsNone(choose([1],[dict(score=1e-8)])['candidate'])
        self.assertEqual(choose([1,2],[dict(score=1),dict(score=1)])['candidate'],1)

    def test_acceptance_is_accept_or_reject_never_second_choice(self):
        candidate=dict(cell='test')
        for n,mean,wins,expected in [(11,1,11,False),(12,-1,12,False),(12,1,5,False),(12,1,6,True)]:
            self.assertEqual(accepted(dict(observations=n,mean=mean,wins=wins),candidate),expected)
        self.assertFalse(accepted(dict(observations=20,mean=1,wins=20),None))

    def test_boundary_ties_match_account_and_missing_selected_label_is_omitted(self):
        score=np.ones((3,14));mask=np.ones_like(score,dtype=bool);y=np.ones_like(score);observed=mask.copy()
        observed[1,0]=False
        stat=proxy(score,score,mask,observed,y)
        self.assertEqual(stat['observations'],2);self.assertEqual(stat['omitted'],1);self.assertEqual(stat['mean'],0)

    def test_zero_shrink_is_exact_core_ensemble(self):
        rng=np.random.default_rng(7);core=[rng.normal(size=(4,14)) for _ in range(5)];mask=np.ones((4,14),bool)
        adapters={('a',5):[p+100 for p in core]}
        a=candidate_score(None,core,adapters,mask)
        b=candidate_score(dict(cell='a',epoch=5,shrink=0),core,adapters,mask)
        np.testing.assert_array_equal(a,b)


class DataTests(unittest.TestCase):
    def test_actual_splits_and_train_only_normalization(self):
        data=prepared();base=data['base']
        self.assertEqual(data['raw']['intraday_window'].shape[2],128)
        self.assertEqual(data['raw']['rich_features'].shape[2],65)
        for year in [2022,2023,2024,2025]:
            tr,va,te,h1,h2=partitions(base,year)
            self.assertGreaterEqual(len(h2),20)
            self.assertLess(max(base['exit'][h1]),min(base['origin'][h2]))
            self.assertFalse(np.intersect1d(h1,h2).size)
            first=normalized(data,tr)
            changed=dict(data,raw={k:v.copy() for k,v in data['raw'].items()})
            for v in changed['raw'].values():v[max(tr)+1:]=1e20
            second=normalized(changed,tr)
            for view in first:
                np.testing.assert_array_equal(first[view][1],second[view][1]);np.testing.assert_array_equal(first[view][2],second[view][2])


if __name__=='__main__':unittest.main()
