import torch
import torch.nn.functional as F


def rank_mse_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, alpha: float):
    """STHAN-SR/RSR objective: masked MSE on returns + alpha * pairwise hinge on mis-ordered pairs. Inputs [B,N]."""
    reg = (mask * (pred - gt) ** 2).mean()
    dp = pred[:, :, None] - pred[:, None, :]
    dg = gt[:, :, None] - gt[:, None, :]
    mm = mask[:, :, None] * mask[:, None, :]
    rank = F.relu(-dp * dg * mm).mean()
    return reg + alpha * rank, reg, rank
