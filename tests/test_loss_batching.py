"""Phase 1.5 fidelity audit: ranking pairs are formed only within a day, and the loss is a mean (not a sum)."""
import numpy as np
import torch

from hypershift.train.loss import rank_mse_loss


def _data(B=3, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, N, generator=g) * 0.02
    gt = torch.randn(B, N, generator=g) * 0.02
    mask = (torch.rand(B, N, generator=g) > 0.2).float()
    return pred, gt, mask


def test_batch_loss_is_mean_of_per_day_losses():
    pred, gt, mask = _data()
    loss, reg, rank = rank_mse_loss(pred, gt, mask, 0.7)
    per_day = [rank_mse_loss(pred[b:b + 1], gt[b:b + 1], mask[b:b + 1], 0.7) for b in range(pred.shape[0])]
    assert abs(loss.item() - np.mean([p[0].item() for p in per_day])) < 1e-9
    assert abs(rank.item() - np.mean([p[2].item() for p in per_day])) < 1e-9
    assert abs(reg.item() - np.mean([p[1].item() for p in per_day])) < 1e-9


def test_no_cross_day_pairs():
    """Gradient of day 0's predictions is independent of what the other days contain (no pairs across days)."""
    pred, gt, mask = _data()
    p1 = pred.clone().requires_grad_()
    rank_mse_loss(p1, gt, mask, 1.0)[0].backward()
    pred2, gt2 = pred.clone(), gt.clone()
    pred2[1:] = torch.randn_like(pred2[1:]) * 0.05          # scramble the other days entirely
    gt2[1:] = torch.randn_like(gt2[1:]) * 0.05
    p2 = pred2.clone().requires_grad_()
    rank_mse_loss(p2, gt2, mask, 1.0)[0].backward()
    assert torch.allclose(p1.grad[0], p2.grad[0], atol=1e-12)


def test_rank_matches_explicit_pair_loop_and_is_a_mean_over_B_N_N():
    pred, gt, mask = _data()
    B, N = pred.shape
    tot = 0.0
    for b in range(B):
        for i in range(N):
            for j in range(N):                               # ordered pairs incl. i == j, same day b only
                tot += max(0.0, -float((pred[b, i] - pred[b, j]) * (gt[b, i] - gt[b, j])) * float(mask[b, i] * mask[b, j]))
    _, _, rank = rank_mse_loss(pred, gt, mask, 1.0)
    assert abs(rank.item() - tot / (B * N * N)) < 1e-9       # divides by B*N*N (mean), same as RSR/STHAN-SR reduce_mean
    _, reg, _ = rank_mse_loss(pred, gt, mask, 1.0)
    assert abs(reg.item() - float((mask * (pred - gt) ** 2).sum()) / (B * N)) < 1e-9   # divides by B*N, not by mask count


def test_masked_stock_excluded_from_pairs():
    pred, gt, mask = _data()
    mask[:, 0] = 0.0
    p = pred.clone().requires_grad_()
    rank_mse_loss(p, gt, mask, 1.0)[0].backward()
    assert torch.all(p.grad[:, 0] == 0)
