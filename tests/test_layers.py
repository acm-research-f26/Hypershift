import torch
from hypershift.geometry.poincare import expmap0
from hypershift.models.layers import EucTemporalConv, HypTemporalConv, PoincareLinear, beta_concat, beta_fn

torch.manual_seed(0)


def test_beta_fn_known_values():
    assert abs(beta_fn(0.5, 0.5) - 3.141592653589793) < 1e-9
    assert abs(beta_fn(1.0, 1.0) - 1.0) < 1e-12


def test_poincare_linear_inside_ball_and_grads():
    fc = PoincareLinear(6, 4)
    x = expmap0(torch.randn(10, 6) * 3).requires_grad_(True)
    y = fc(x)
    assert y.shape == (10, 4) and y.norm(dim=-1).max() < 1
    y.sum().backward()
    assert torch.isfinite(fc.z.grad).all() and torch.isfinite(fc.r.grad).all()


def test_beta_concat_single_input_identity():
    x = expmap0(torch.randn(3, 4))
    torch.testing.assert_close(beta_concat([x]), x, atol=1e-6, rtol=1e-5)


def test_hyp_temporal_conv_matches_manual_window():
    conv = HypTemporalConv(in_dim=3, out_dim=5, kernel=4)
    x = expmap0(torch.randn(2, 8, 6, 3))                  # B=2, T=8, N=6, C=3
    y = conv(x)
    assert y.shape == (2, 2, 6, 5)
    manual = conv.fc(beta_concat([x[:, 4 + s] for s in range(4)]))   # second window, [B,N,5]
    torch.testing.assert_close(y[:, 1], manual, atol=1e-5, rtol=1e-4)


def test_euc_temporal_conv_shapes():
    conv = EucTemporalConv(3, 5, 4, activation=False)
    assert conv(torch.randn(2, 16, 6, 3)).shape == (2, 4, 6, 5)
