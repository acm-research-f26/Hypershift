"""Causal training-return rank priors with a fixed neutral shrinkage prior."""
import numpy as np
from ndx_rank_retraining import percentile_predictions


def training_rank_prior(data,train,half_life_weeks=0):
    train=np.asarray(train);observed=data['labelmask'][train]
    ranks=percentile_predictions(data['y'][train],observed)
    if half_life_weeks<0 or not len(train):raise ValueError('Invalid prior history')
    age=(data['origin'][train].max()-data['origin'][train])/5.
    weights=np.ones(len(train)) if half_life_weeks==0 else np.exp2(-age/half_life_weeks)
    count=(observed*weights[:,None]).sum(0)
    score=((ranks*weights[:,None]).sum(0)+10.)/(count+20.)
    assert np.isfinite(score).all()
    return score,count
