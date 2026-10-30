"""Rank-aligned training, neutral stock-ID initialization and split validation."""
import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F


def neutralize_stock_initialization(model, family):
    """Remove random stock-ID preferences without changing parameter inventory."""
    with torch.no_grad():
        if family == 'think':
            model.core.market[2].weight.zero_(); model.core.market[2].bias.zero_()
        elif family == 'dynamic':
            model.network.whconv1.diagonal_weight_filter.fill_(1.)
            model.network.whconv2.diagonal_weight_filter.fill_(1.)
        else: raise ValueError('Unknown family')


def rank_objective(score, actual, observed):
    """Date-balanced centered Huber plus pairwise logistic ranking loss.

    Labels are normalized within each observed training/validation date. These
    are ranking scores; they are not calibrated percentage-return forecasts.
    """
    if score.ndim == 1: score, actual, observed = score[None], actual[None], observed[None]
    mask = observed.bool(); count = mask.sum(1, keepdim=True).clamp_min(1)
    y = torch.where(mask, actual, torch.zeros_like(actual))
    ymean = y.sum(1, keepdim=True)/count
    centered = torch.where(mask, y-ymean, torch.zeros_like(y))
    sd = (centered.square().sum(1, keepdim=True)/count).sqrt().clamp_min(.1)
    target = (centered/sd).clamp(-3., 3.)
    clean = torch.where(mask, score, torch.zeros_like(score))
    prediction = torch.where(mask, clean-clean.sum(1, keepdim=True)/count, torch.zeros_like(clean))
    regression = (F.smooth_l1_loss(prediction, target, reduction='none')*mask).sum(1)/count[:, 0]
    difference = prediction[:, :, None]-prediction[:, None, :]
    label_gap = target[:, :, None]-target[:, None, :]
    pairs = mask[:, :, None] & mask[:, None, :] & (label_gap.abs() >= .1)
    pairs &= torch.ones_like(pairs).triu(diagonal=1)
    pair_count = pairs.sum((1, 2)).clamp_min(1)
    pairwise = (F.softplus(-label_gap.sign()*difference)*pairs).sum((1, 2))/pair_count
    valid = observed.bool().sum(-1) >= 2
    return (regression+.2*pairwise)[valid].mean()


def split_validation(data, validation):
    """Select epochs on the first half; purge labels crossing the audit half."""
    validation = np.asarray(validation)
    middle = len(validation)//2
    audit = validation[middle:]
    boundary = data['origin'][audit[0]]
    selection = validation[:middle]
    selection = selection[data['exit'][selection] < boundary]
    if len(selection) < 16 or len(audit) < 16: raise ValueError('Insufficient split validation')
    assert data['exit'][selection].max() < data['origin'][audit].min()
    return selection, audit


def rank_statistics(prediction, actual, observed):
    """Fast tie-aware RankIC and NDCG@10 with ordinal return relevance."""
    values = []; ndcg = []
    for p, y, mask in zip(prediction, actual, observed):
        p, y = p[mask], y[mask]
        if len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Invalid labeled ranking sample')
        rp, ry = rankdata(p), rankdata(y)
        a, b = rp-rp.mean(), ry-ry.mean(); denominator = np.sqrt((a@a)*(b@b))
        values.append(float(a@b/denominator) if denominator > 1e-15 else 0.)
        gain = (ry-1)/max(len(ry)-1, 1); order = np.argsort(-p, kind='stable')
        discounts = 1/np.log2(np.arange(min(10, len(p)))+2)
        ranked = gain[order].copy(); sorted_p = p[order]
        unique, counts = np.unique(sorted_p, return_counts=True)
        for value in unique[counts > 1]:
            tie = sorted_p == value
            if tie.sum() > 1: ranked[tie] = ranked[tie].mean()
        ideal = np.sort(gain)[::-1][:len(discounts)]@discounts
        ndcg.append(float(ranked[:len(discounts)]@discounts/ideal) if ideal else 0.)
    quarters = np.array([np.mean(part) for part in np.array_split(values, 4)])
    mean, dispersion = float(np.mean(values)), float(quarters.std(ddof=1))
    return dict(rankic=mean, quarter_sd=dispersion, ndcg10=float(np.mean(ndcg)),
                criterion=mean-.5*dispersion)


def percentile_predictions(prediction, eligible):
    result = np.zeros_like(prediction, dtype=float)
    for i, mask in enumerate(eligible): result[i, mask] = rankdata(prediction[i, mask])/mask.sum()
    return result


def epoch_choices(history):
    if not history: raise ValueError('No epochs')
    loss = min(history, key=lambda row:(row['selection_loss'], row['epoch']))['epoch']
    ordered = sorted(history, key=lambda row:(-row['selection_criterion'], row['epoch']))
    return dict(loss=[loss], rank=[ordered[0]['epoch']], snapshots=[row['epoch'] for row in ordered[:3]], initial=[0])
