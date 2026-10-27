import unittest
import torch
from rishi_synthetic_robustness import make_case, masked_score, forecast
from rishi_synthetic_gate import make_model


class SyntheticRobustnessTests(unittest.TestCase):
    def test_missingness_imputation_and_observable_signal(self):
        d=make_case('missing_own')
        for split in ['train','test']:
            p=d[split];missing=p['missing']
            self.assertTrue(torch.all(p['x'][missing][:,:,5:24]==0))
            self.assertTrue(torch.all(p['x'][missing][:,:,24]==0))
            self.assertTrue(torch.all(p['x'][~missing][:,:,24]==.2))
            torch.testing.assert_close(p['x'][:,:,0,0]/.2,p['target'])
            self.assertTrue(p['eligible'].all())
        self.assertGreater(float(d['test']['missing'].float().mean()),float(d['train']['missing'].float().mean()))

    def test_coverage_null_removes_the_signal_but_keeps_latent_target(self):
        d=make_case('coverage_null')
        self.assertTrue(torch.all(d['train']['x'][:,:,:,0]==0))
        self.assertGreater(float(d['train']['target'].std()),.5)
        self.assertFalse(torch.equal(d['train']['x'][:64],d['test']['x']))

    def test_membership_shift_inactive_nan_and_known_peer_supervision(self):
        d=make_case('membership_peer')
        for split,lo,hi in [('train',6,8),('test',4,6)]:
            p=d[split];e=p['eligible'];counts=e.sum(1)
            self.assertTrue(((counts>=lo)&(counts<=hi)).all())
            self.assertTrue(e[:,:2].all())
            self.assertTrue(torch.isnan(p['x'][~e]).all())
            self.assertTrue(torch.isfinite(p['x'][e]).all())
            torch.testing.assert_close(p['supervised'],e&e[:,torch.arange(8)^1])
            torch.testing.assert_close(p['target'],p['latent'][:,torch.arange(8)^1])
            self.assertTrue((p['supervised'].sum(1)>=2).all())
            self.assertTrue((e&~p['supervised']).any())

    def test_own_supervision_uses_all_current_eligible_ids(self):
        d=make_case('membership_own')
        torch.testing.assert_close(d['test']['supervised'],d['test']['eligible'])
        torch.testing.assert_close(d['test']['target'],d['test']['latent'])
        self.assertTrue(torch.equal(d['train']['x'].nan_to_num(),make_case('membership_own')['train']['x'].nan_to_num()))

    def test_masked_metric_never_reads_unsupervised_nan(self):
        p=torch.tensor([[float('nan'),2.]])
        y=torch.tensor([[float('nan'),2.]])
        m=torch.tensor([[False,True]])
        r=masked_score(p,y,m)
        self.assertEqual(r['r2_zero'],1.)
        self.assertEqual(r['observations'],1)

    def test_actual_paths_mask_inactive_gradients_under_shifted_membership(self):
        torch.set_num_threads(2)
        p=make_case('membership_peer')['test']
        for arm in ['full','temporal_only']:
            with self.subTest(arm=arm):
                model=make_model(7).eval()
                x=p['x'][0].clone().requires_grad_();e=p['eligible'][0];s=p['supervised'][0]
                y=forecast(model,x,e,arm)
                (y[s]-p['target'][0,s]).square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(float(x.grad[~e].abs().max()),0.)
                self.assertGreater(float(x.grad[e].abs().sum()),0.)


if __name__=='__main__':unittest.main()
