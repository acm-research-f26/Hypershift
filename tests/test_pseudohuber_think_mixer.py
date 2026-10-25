import unittest
import numpy as np
import torch
from pseudohuber_think_mixer import pseudo_huber_loss
from robust_models import make_model
from audit_historical_ndx_ranking import diagnostics


class PseudoHuberTests(unittest.TestCase):
    def test_gradient_and_curvature_match_smooth_formula(self):
        x = torch.tensor([[-100., -1., 0., 1., 100.]], dtype=torch.float64, requires_grad=True)
        loss = pseudo_huber_loss(x, torch.zeros_like(x), torch.ones_like(x))
        gradient, = torch.autograd.grad(loss, x, create_graph=True)
        torch.testing.assert_close(gradient, 2*x/torch.sqrt(1+x.square())/x.numel())
        curvature, = torch.autograd.grad(gradient.sum(), x)
        torch.testing.assert_close(curvature, 2/(1+x.square()).pow(1.5)/x.numel())
        self.assertTrue((gradient.abs() <= 2/x.numel()).all())

    def test_small_errors_match_mse_and_outliers_are_downweighted(self):
        x = torch.tensor([[1e-5]], dtype=torch.float64)
        torch.testing.assert_close(pseudo_huber_loss(x, torch.zeros_like(x), torch.ones_like(x)), x.square().mean())
        x = torch.tensor([[1000.]])
        self.assertLess(pseudo_huber_loss(x, torch.zeros_like(x), torch.ones_like(x)), x.square().mean()/100)

    def test_masked_nan_and_empty_targets_have_finite_zero_gradients(self):
        for mask in [torch.tensor([[1., 0.]]), torch.zeros(1, 2)]:
            x = torch.tensor([[1., float('nan')]], requires_grad=True)
            loss = pseudo_huber_loss(x, torch.tensor([[0., float('nan')]]), mask)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(x.grad[0, 1], 0.)

    def test_masked_future_nodes_do_not_enter_mixer_scores(self):
        torch.manual_seed(7)
        model = make_model('think_mix', 4, [{0,1,2}, {1,2,3}], causal_membership=True)
        x = torch.randn(2,16,4,5)*.1
        mask = torch.tensor([[1.,1.,0.,0.], [0.,1.,1.,0.]])
        changed = torch.where(mask[:,None,:,None].bool(), x, torch.full_like(x,float('nan')))
        torch.testing.assert_close(model(x,mask)[0], model(changed,mask)[0], rtol=0, atol=0)

    def test_ndcg_fifteen_ideal_and_tied_random_baseline(self):
        y = np.arange(25.)
        ideal = diagnostics(y, y, (5,10,15,20))
        self.assertAlmostEqual(ideal['NDCG@15'], 1.)
        tied = diagnostics(np.zeros_like(y), y, (5,10,15,20))
        self.assertAlmostEqual(tied['NDCG@15'], tied['Random NDCG@15'])


if __name__ == '__main__':
    unittest.main()
