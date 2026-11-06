import copy
import unittest
import torch
from ndx_consistency_training import symmetric_rank_kl, ParameterEMA


class ConsistencyTrainingTests(unittest.TestCase):
    def test_kl_is_symmetric_zero_at_agreement_and_masks_unavailable_stocks(self):
        a = torch.tensor([[1., -1., 100.]], requires_grad=True)
        b = torch.tensor([[.5, -.5, -100.]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = symmetric_rank_kl(a, b, mask)
        self.assertGreater(loss.item(), 0)
        torch.testing.assert_close(loss, symmetric_rank_kl(b, a, mask))
        torch.testing.assert_close(symmetric_rank_kl(a, a, mask), torch.tensor(0.))
        loss.backward()
        self.assertEqual(a.grad[0, 2].item(), 0.)
        self.assertEqual(b.grad[0, 2].item(), 0.)
        self.assertTrue(torch.isfinite(a.grad).all())
        c = a.detach()+37
        torch.testing.assert_close(symmetric_rank_kl(c, b.detach(), mask), loss.detach(), atol=1e-5, rtol=1e-5)

    def test_ema_arithmetic_copies_buffers_and_resumes_exactly(self):
        model = torch.nn.BatchNorm1d(2)
        ema = ParameterEMA(model, .5)
        with torch.no_grad():
            model.weight.fill_(3.); model.running_mean.fill_(7.)
        ema.update(model)
        state = ema.averaged_state(model)
        torch.testing.assert_close(state['weight'], torch.full((2,), 2.))
        torch.testing.assert_close(state['running_mean'], torch.full((2,), 7.))
        restored = ParameterEMA(model, .5); restored.load_state_dict(ema.state_dict(), 'cpu')
        with torch.no_grad(): model.weight.fill_(5.)
        ema.update(model); restored.update(model)
        for key, value in ema.averaged_state(model).items():
            torch.testing.assert_close(value, restored.averaged_state(model)[key], atol=0, rtol=0)

    def test_optimizer_rng_and_ema_resume_preserve_actual_computation(self):
        torch.manual_seed(37)
        model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.2), torch.nn.Linear(6, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        ema = ParameterEMA(model)
        x = torch.arange(20).reshape(5, 4).float()/20
        mask = torch.ones((5, 3), dtype=torch.bool)
        def step(m, o, e):
            o.zero_grad(); a, b = m(x), m(x)
            loss = a.square().mean()+b.square().mean()+symmetric_rank_kl(a, b, mask)
            loss.backward(); o.step(); e.update(m)
        step(model, optimizer, ema)
        saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=ema.state_dict(), rng=torch.get_rng_state()))
        step(model, optimizer, ema)
        expected = copy.deepcopy(model.state_dict()); expected_ema = ema.averaged_state(model)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema'], 'cpu'); torch.set_rng_state(saved['rng'])
        step(model, optimizer, ema)
        for key in expected:
            torch.testing.assert_close(model.state_dict()[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(ema.averaged_state(model)[key], expected_ema[key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
