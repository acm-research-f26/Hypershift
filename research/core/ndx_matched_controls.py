"""Causal controls for the frozen NDX drawdown study; no fitted parameters."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ndx_drawdown_engine import issuer_weights


def momentum_scores(data, ids, lookback=252, skip=21):
    """12-minus-1 momentum: close[t-21]/close[t-252]-1, at prior close t.

    Require both endpoint observations to be tradable. Missing histories stay
    unavailable for this control and never change the neural eligibility mask.
    """
    if not 0 < skip < lookback:
        raise ValueError('Momentum requires 0 < skip < lookback')
    result = np.full((len(ids), len(data['panel'].tickers)), np.nan)
    for row, i in enumerate(ids):
        day = int(data['origin'][i])
        if day < lookback:
            continue
        end, start = day-skip, day-lookback
        valid = data['tradable'][start] & data['tradable'][end]
        valid &= np.isfinite(data['marks'][start]) & np.isfinite(data['marks'][end])
        valid &= data['marks'][start] > 0
        result[row, valid] = data['marks'][end, valid]/data['marks'][start, valid]-1
    return result


def momentum_weights(data, ids):
    score = momentum_scores(data, ids)
    weights = np.zeros_like(score)
    support = []
    for row, i in enumerate(ids):
        eligible = data['mask'][i] & np.isfinite(score[row])
        if eligible.sum() < 5:
            raise ValueError('Fewer than five supported momentum securities')
        weights[row] = issuer_weights(score[row], eligible, data['panel'].tickers, 5)
        support.append(dict(origin=str(data['panel'].dates[data['origin'][i]].date()),
                            neural_eligible=int(data['mask'][i].sum()),
                            momentum_eligible=int(eligible.sum()),
                            missing_history=int((data['mask'][i] & ~eligible).sum())))
    return weights, pd.DataFrame(support)


def etf_data(data, ticker):
    """One-asset adapter retaining the parent calendar, caps and close fills."""
    root = data['manifest']['root']
    price = pd.read_csv(root+'/etf_adj_close.csv', index_col=0, parse_dates=True)[ticker]
    volume = pd.read_csv(root+'/etf_dollar_volume.csv', index_col=0, parse_dates=True)[ticker]
    if not price.index.equals(data['panel'].dates) or not volume.index.equals(price.index):
        raise ValueError('ETF and neural calendars differ')
    if price.isna().any() or (price <= 0).any():
        raise ValueError('Invalid ETF marks')
    returns = price.pct_change(fill_method=None).to_numpy()[:, None]
    return dict(panel=SimpleNamespace(dates=price.index, tickers=[ticker]),
                marks=price.to_numpy()[:, None], returns=returns,
                tradable=np.ones((len(price), 1), dtype=bool),
                adv=volume.rolling(20, min_periods=20).mean().to_numpy()[:, None],
                origin=data['origin'].copy(), entry=data['entry'].copy(),
                events=[], manifest=data['manifest'])


def paired_inference(matrix, names, contrasts, block, replicates=10000, seed=20261027):
    """Joint circular-block draws; eight-contrast simultaneous active-mean CIs.

    SR/CAGR differences remain pointwise. Neither interval corrects the older
    model/risk-rule search or retraining/data-vintage uncertainty.
    """
    index = {name: j for j, name in enumerate(names)}
    left = np.array([index[a] for a, _ in contrasts])
    right = np.array([index[b] for _, b in contrasts])
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    active = matrix[:, left]-matrix[:, right]
    observed = active.mean(0)*252
    rng = np.random.default_rng(seed+block)
    means, sr_diffs, cagr_diffs = [], [], []
    for start in range(0, replicates, 100):
        count = min(100, replicates-start)
        origins = rng.integers(0, n, (count, int(np.ceil(n/block))))
        rows = ((origins[:, :, None]+np.arange(block)) % n).reshape(count, -1)[:, :n]
        sample = matrix[rows]
        sr = np.sqrt(252)*(sample.mean(1)-(1.03**(1/252)-1))/sample.std(1, ddof=1)
        cagr = np.expm1(np.log1p(sample).sum(1)*252/n)
        means.append((sample[:, :, left]-sample[:, :, right]).mean(1)*252)
        sr_diffs.append(sr[:, left]-sr[:, right])
        cagr_diffs.append(cagr[:, left]-cagr[:, right])
    means, sr_diffs, cagr_diffs = map(np.concatenate, [means, sr_diffs, cagr_diffs])
    se = means.std(0, ddof=1)
    if (se <= 1e-14).any():
        raise ValueError('Degenerate active-return contrast')
    critical = np.quantile(np.max(np.abs((means-observed)/se), axis=1), .95)
    result = []
    for j, (a, b) in enumerate(contrasts):
        result.append(dict(strategy=a, comparator=b, block_sessions=block,
                           replicates=replicates, annual_active_mean=float(observed[j]),
                           simultaneous_active_low=float(observed[j]-critical*se[j]),
                           simultaneous_active_high=float(observed[j]+critical*se[j]),
                           paired_sr_low=float(np.quantile(sr_diffs[:, j], .025)),
                           paired_sr_high=float(np.quantile(sr_diffs[:, j], .975)),
                           paired_cagr_low=float(np.quantile(cagr_diffs[:, j], .025)),
                           paired_cagr_high=float(np.quantile(cagr_diffs[:, j], .975))))
    return pd.DataFrame(result)
