"""Structural invariance tests, not financial learner-performance tests."""
import copy
import importlib.util
from pathlib import Path
import unittest

import torch

from ndx_extended_hours import ExtendedModel
import ndx_reference_models as references


class ComponentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.n = 12
        cls.groups = [{i,(i+1)%cls.n} for i in range(cls.n)]
        cls.permutation = torch.tensor([3,0,9,1,7,4,11,2,8,5,10,6])
        cls.inverse = torch.argsort(cls.permutation)
        cls.source = Path.cwd()/'docs/rishi_mechanism_20261027/source_snapshot'

    def test_parent_consistent_id_transport_preserves_predictions(self):
        torch.manual_seed(43)
        model = ExtendedModel(self.n,self.groups,'after_hours_premarket').eval()
        x = .2*torch.randn(2,16,self.n,5)
        extra = torch.randn(2,self.n,320)
        mask = torch.ones(2,self.n)
        mask[:,-2:] = 0
        p = self.permutation
        transported = copy.deepcopy(model)
        with torch.no_grad():
            spatial = transported.core.base.spatial
            spatial.node_ids.copy_(self.inverse[spatial.node_ids])
            transported.core.market[0].weight.copy_(model.core.market[0].weight[:,p])
            transported.core.market[2].weight.copy_(model.core.market[2].weight[p])
            transported.core.market[2].bias.copy_(model.core.market[2].bias[p])
            expected = model(x,extra,mask)[0][:,p]
            actual = transported(x[:,:,p],extra[:,p],mask[:,p])[0]
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            feature_only = model(x[:,:,p],extra[:,p],mask[:,p])[0]
            self.assertGreater(float((feature_only-expected).abs().max()),.01)

    def dynamic(self):
        spec = importlib.util.spec_from_file_location('component_audit_layers',self.source/'layer.py')
        layers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layers)
        cls = references.definitions(self.source/'model.py',['Model'],vars(layers))['Model']
        original = references.dynamic_class
        try:
            references.dynamic_class = lambda:cls
            return references.make_reference('dynamic_restricted',self.n,references.wavelet_snapshot(self.groups,self.n))
        finally:
            references.dynamic_class = original

    def test_dynamic_consistent_id_transport_includes_graph_and_filters(self):
        torch.manual_seed(44)
        model = self.dynamic().eval()
        transported = copy.deepcopy(model)
        p = self.permutation
        x = .2*torch.randn(self.n,16,25)
        mask = torch.ones(self.n,dtype=torch.bool)
        mask[-2:] = False
        with torch.no_grad():
            transported.wavelets.copy_(model.wavelets[p][:,p])
            transported.wavelets_inv.copy_(model.wavelets_inv[p][:,p])
            for name in ['whconv1','whconv2']:
                getattr(transported.network,name).diagonal_weight_filter.copy_(getattr(model.network,name).diagonal_weight_filter[p])
            expected = model(x,None,mask)[0][p]
            actual = transported(x[p],None,mask[p])[0]
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_dynamic_inactive_features_have_zero_gradient_to_active_outputs(self):
        torch.manual_seed(45)
        model = self.dynamic().eval()
        x = (.2*torch.randn(self.n,16,25)).requires_grad_()
        mask = torch.ones(self.n,dtype=torch.bool)
        mask[-3:] = False
        score = model(x,None,mask)[0]
        score[mask].square().sum().backward()
        self.assertEqual(float(x.grad[-3:].abs().max()),0.)
        self.assertGreater(float(x.grad[:-3].abs().sum()),0.)
        torch.testing.assert_close(score[-3:],torch.zeros(3),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
