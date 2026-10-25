import unittest
import torch
from robust_models import pairwise_logistic_loss, robust_objective, make_model


class RankingLossTests(unittest.TestCase):
    def test_correct_order_has_lower_loss_and_wrong_order_gradient_repairs_it(self):
        truth = torch.tensor([[2., 0., -2.]])
        mask = torch.ones_like(truth)
        self.assertLess(pairwise_logistic_loss(truth, truth, mask),
                        pairwise_logistic_loss(-truth, truth, mask))
        score = (-truth).clone().requires_grad_()
        pairwise_logistic_loss(score, truth, mask).backward()
        self.assertLess(score.grad[0, 0], 0.)
        self.assertGreater(score.grad[0, 2], 0.)

    def test_masked_nan_and_ties_do_not_contribute(self):
        score = torch.tensor([[1., 0., float('nan')]], requires_grad=True)
        truth = torch.tensor([[1., 0., float('nan')]])
        mask = torch.tensor([[1., 1., 0.]])
        actual = pairwise_logistic_loss(score, truth, mask)
        expected = pairwise_logistic_loss(score[:, :2], truth[:, :2], mask[:, :2])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        x = torch.zeros(2, 3, requires_grad=True)
        empty = pairwise_logistic_loss(x, torch.zeros_like(x), torch.ones_like(x))
        empty.backward()
        self.assertEqual(empty.item(), 0.)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_return_outlier_magnitude_cannot_amplify_ranking_gradient(self):
        gradients = []
        for magnitude in [1., 1000.]:
            score = torch.zeros(1, 2, requires_grad=True)
            pairwise_logistic_loss(score, torch.tensor([[magnitude, 0.]]), torch.ones_like(score)).backward()
            gradients.append(score.grad)
        torch.testing.assert_close(*gradients)

    def test_same_architecture_and_exact_additive_loss(self):
        groups = [{i} for i in range(12)]
        torch.manual_seed(7)
        a = make_model('think_mix_huber', 12, groups)
        torch.manual_seed(7)
        b = make_model('think_mix_huber_rank_03', 12, groups)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key])
        score, y, mask = torch.randn(2, 12), torch.randn(2, 12), torch.ones(2, 12)
        expected = robust_objective(score, y, mask, 'think_mix_huber') + .3*pairwise_logistic_loss(score, y, mask)
        torch.testing.assert_close(expected, robust_objective(score, y, mask, 'think_mix_huber_rank_03'))

    def test_no_across_date_pairs(self):
        score, y = torch.randn(2, 3), torch.tensor([[2., 0., -2.], [-2., 0., 2.]])
        mask = torch.ones_like(y)
        expected = sum(pairwise_logistic_loss(score[i:i+1], y[i:i+1], mask[i:i+1]) for i in range(2))/2
        torch.testing.assert_close(pairwise_logistic_loss(score, y, mask), expected)


if __name__ == '__main__':
    unittest.main()
