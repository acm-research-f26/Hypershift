"""Fixed trailing factor controls; all features end at the decision origin."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def eligible_percentiles(values, eligible):
    result = np.zeros_like(values, dtype=float)
    for row, mask in enumerate(eligible):
        known = mask & np.isfinite(values[row])
        result[row, mask] = .5  # Missing trailing history is neutral, not excluded.
        if known.any(): result[row, known] = rankdata(values[row, known])/known.sum()
    return result


def factor_scores(data, ids):
    price = pd.DataFrame(data['marks']); returns = pd.DataFrame(data['returns'])
    vol20 = returns.rolling(20, min_periods=20).std()*np.sqrt(252)
    vol60 = returns.rolling(60, min_periods=40).std()*np.sqrt(252)
    middle = ((price/price.shift(63)-1)/vol20.clip(lower=.05)).to_numpy()
    longer = ((price.shift(21)/price.shift(252)-1)/vol60.clip(lower=.05)).to_numpy()
    origins = data['origin'][ids]; mask = data['mask'][ids]
    a = eligible_percentiles(middle[origins], mask)
    b = eligible_percentiles(longer[origins], mask)
    liquid = eligible_percentiles(np.log1p(data['adv'][origins]), mask)
    return dict(momentum63_risk=a, momentum252_21_risk=b,
                liquid_consensus=(a+b+liquid)/3)


def factor_blend(factor, neural, neural_weight):
    if (factor.shape != neural.shape or not 0 <= neural_weight <= 1
            or not np.isfinite(factor).all() or not np.isfinite(neural).all()):
        raise ValueError('Invalid factor blend')
    return (1-neural_weight)*factor+neural_weight*neural
