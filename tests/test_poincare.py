import math
import torch
import pytest
from hypershift.geometry.poincare import (
    MAX_NORM, expmap0, gyromidpoint, lambda_x, logmap0, mobius_add, mobius_scalar,
    poincare_dist, project,
)

torch.manual_seed(0)


def rand_ball(*shape, scale=0.5):
    return expmap0(torch.randn(*shape, dtype=torch.float64) * scale)


def test_exp_log_inverse():
    v = torch.randn(10, 4, dtype=torch.float64)
    torch.testing.assert_close(logmap0(expmap0(v)), v, atol=1e-6, rtol=1e-6)


def test_project_bounds():
    x = torch.tensor([[2.0, 0.0], [0.1, 0.1]])
    assert project(x).norm(dim=-1).max() <= MAX_NORM + 1e-7
    torch.testing.assert_close(project(x)[1], x[1])


def test_mobius_identity_and_left_cancellation():
    x, y = rand_ball(5, 3), rand_ball(5, 3)
    torch.testing.assert_close(mobius_add(x, torch.zeros_like(x)), x)
    torch.testing.assert_close(mobius_add(-x, mobius_add(x, y)), y, atol=1e-9, rtol=1e-7)


def test_distance_properties():
    x, y = rand_ball(6, 3), rand_ball(6, 3)
    torch.testing.assert_close(poincare_dist(x, y), poincare_dist(y, x))
    assert torch.all(poincare_dist(x, x) < 1e-6)
    zero = torch.zeros_like(x)
    torch.testing.assert_close(poincare_dist(zero, x), 2 * torch.atanh(x.norm(dim=-1)))


def test_mobius_scalar():
    x = rand_ball(4, 3)
    torch.testing.assert_close(mobius_scalar(1.0, x), x)
    two = mobius_scalar(2.0, x)
    torch.testing.assert_close(two, mobius_add(x, x), atol=1e-9, rtol=1e-7)


def test_lambda():
    x = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(lambda_x(x).squeeze(-1), torch.tensor([2.0, 2 / (1 - 0.36)], dtype=torch.float64))


def test_gyromidpoint_single_point_is_identity_and_symmetric_pair_is_origin():
    x = rand_ball(3, 4)                          # nodes 0,1,2
    node = torch.tensor([0, 1, 2])
    edge = torch.tensor([0, 1, 2])               # each node its own hyperedge
    torch.testing.assert_close(gyromidpoint(x, node, edge, 3), x, atol=1e-9, rtol=1e-7)
    p = rand_ball(1, 4)
    pair = torch.cat([p, -p])
    m = gyromidpoint(pair, torch.tensor([0, 1]), torch.tensor([0, 0]), 1)
    assert m.norm() < 1e-9


def test_gyromidpoint_batched_leading_dims():
    x = rand_ball(2, 7, 5, 3)                    # [B,T,N,D]
    node = torch.tensor([0, 1, 2, 2, 3, 4])
    edge = torch.tensor([0, 0, 0, 1, 1, 1])
    m = gyromidpoint(x, node, edge, 2)
    assert m.shape == (2, 7, 2, 3)
    assert m.norm(dim=-1).max() < 1


def test_near_boundary_finite_grads():
    v = (torch.randn(8, 4) * 20).requires_grad_(True)   # exp map lands at the boundary
    x = expmap0(v)
    y = expmap0(torch.randn(8, 4))
    loss = poincare_dist(x, y).sum() + logmap0(x).pow(2).sum()
    loss.backward()
    assert torch.isfinite(v.grad).all()
