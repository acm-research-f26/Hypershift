import unittest
import torch
from robust_models import make_model
from think_model import HypergraphAttention, exp0


class MembershipModelTests(unittest.TestCase):
    def test_inactive_inputs_and_gradients_cannot_affect_active_scores(self):
        torch.set_num_threads(2)
        for kind in ['think','think_mix','think_huber','think_mix_huber','mlp','stockmixer']:
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                model = make_model(kind,4,[{0,1,2},{1,2,3}],causal_membership=True)
                # Exercise learned biases, rather than just their zero initialization.
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if name.endswith('r') or 'bias' in name:p.fill_(.1)
                x = (torch.randn(2,16,4,5)*.1).requires_grad_()
                mask = torch.tensor([[1.,1.,0.,0.],[0.,1.,1.,0.]])
                altered = torch.where(mask[:,None,:,None].bool(),x.detach(),torch.full_like(x,float('nan')))
                a,_ = model(x,mask)
                b,_ = model(altered,mask)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue((a[~mask.bool()]==0).all())
                a.sum().backward()
                hidden_gradient = x.grad.masked_select(~mask[:,None,:,None].bool().expand_as(x))
                self.assertTrue((hidden_gradient==0).all())

    def test_inactive_group_padding_does_not_change_edge_center(self):
        for hyperbolic in [True,False]:
            with self.subTest(hyperbolic=hyperbolic):
                torch.manual_seed(11)
                a = HypergraphAttention(3,3,[{0,1}],hyperbolic=hyperbolic)
                b = HypergraphAttention(3,3,[{0,1,2}],hyperbolic=hyperbolic)
                b.fc.load_state_dict(a.fc.state_dict())
                with torch.no_grad():b.a.copy_(a.a)
                x = torch.randn(2,4,3,3)*.1
                if hyperbolic:x=exp0(x)
                mask = torch.tensor([[1.,1.,0.],[1.,1.,0.]])
                # Different pair-array sizes can change float32 vectorization.
                torch.testing.assert_close(a(x,mask),b(x,mask),rtol=1e-6,atol=1e-9)


if __name__ == '__main__':
    unittest.main()
