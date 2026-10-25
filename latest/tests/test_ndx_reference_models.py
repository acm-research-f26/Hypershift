import unittest
import numpy as np
import torch
from ndx_reference_models import make_reference,wavelet_snapshot
from ndx_reference_training import prepare,factor_history


class ReferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.x=torch.randn(12,16,25);cls.market=torch.randn(16,10);cls.prior=torch.randn(1,13)
        cls.snapshot=wavelet_snapshot([{i,(i+1)%12} for i in range(12)],12)

    def test_master_matches_author_on_all_eligible_and_ignores_inactive_nan(self):
        m=make_reference('master',12).eval();mask=torch.ones(12,dtype=torch.bool)
        expected=m.network(torch.cat([self.x,self.market.expand(12,-1,-1)],-1))
        torch.testing.assert_close(m(self.x,self.market,mask)[0],expected)
        mask[-3:]=False;bad=self.x.clone();bad[-3:]=float('nan')
        torch.testing.assert_close(m(self.x,self.market,mask)[0],m(bad,self.market,mask)[0])

    def test_dynamic_matches_author_on_all_eligible_and_masks_inactive_inputs(self):
        m=make_reference('dynamic_restricted',12,self.snapshot).eval();mask=torch.ones(12,dtype=torch.bool)
        torch.testing.assert_close(m(self.x,None,mask)[0],m.network(self.x))
        mask[-3:]=False;bad=self.x.clone();bad[-3:]=float('nan')
        torch.testing.assert_close(m(self.x,None,mask)[0],m(bad,None,mask)[0])

    def test_prism_frozen_codebook_stays_fixed_and_future_labels_not_needed_at_inference(self):
        m=make_reference('prism',12);m.freeze_representation();m.train()
        self.assertFalse(m.vq.training);self.assertFalse(m.vq.quantizer.training)
        before={k:v.clone() for k,v in m.vq.state_dict().items()}
        mask=torch.ones(12,dtype=torch.bool);score,aux=m(self.x,self.prior,mask)
        loss=score.square().mean()+.001*aux;loss.backward()
        for key,value in m.vq.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        m.eval();mask[-3:]=False;bad=self.x.clone();bad[-3:]=float('nan')
        torch.testing.assert_close(m(self.x,self.prior,mask)[0],m(bad,self.prior,mask)[0])

    def test_causal_input_alignment_and_extra_label_purge(self):
        p=prepare(2025);d=p['data'];dates=d['panel'].dates
        self.assertEqual(p['features'].shape[2:],(16,25))
        self.assertTrue(np.isfinite(p['prior'][np.r_[p['train'],p['validation'],p['test']]]).all())
        self.assertLess(dates[d['entry'][p['pretrain'][-1]]+10],dates[d['origin'][p['validation'][0]]])
        self.assertLess(dates[d['entry'][p['prevalidation'][-1]]+10],dates[d['origin'][p['test'][0]]])


if __name__=='__main__':unittest.main()
