import numpy as np

from hypershift.data.rsr import MarketData
from hypershift.eval.metrics import cumulative_return, evaluate_all, irr, max_drawdown, sharpe
from hypershift.train.loop import gather_batch, window_offsets


def baseline_scores(data: MarketData, seq: int, split: str, kind: str, seed: int = 0, lookback: int = 5):
    offs = window_offsets(data, seq, split)
    _, m, _, g = gather_batch(data, offs, seq)
    gt, mask = g.T, m.T
    ends = offs + seq - 1
    if kind == "oracle":
        pred = gt.copy()
    elif kind == "random":
        pred = np.random.default_rng(seed).standard_normal(gt.shape)
    elif kind in ("momentum", "reversal"):
        past = data.base_price[:, np.maximum(ends - lookback, 0)]
        mom = data.base_price[:, ends] / np.maximum(past, 1e-8) - 1
        pred = mom if kind == "momentum" else -mom
    else:
        raise ValueError(kind)
    return pred, gt, mask


def market_daily_returns(gt, mask) -> np.ndarray:
    return (gt * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)


def evaluate_baselines(data, seq=16, split="test", k=5, periods_per_year=252, random_seeds=25) -> dict:
    res = {}
    for kind in ("oracle", "momentum", "reversal"):
        res[kind] = evaluate_all(*baseline_scores(data, seq, split, kind), k=k, periods_per_year=periods_per_year)
    runs = [evaluate_all(*baseline_scores(data, seq, split, "random", seed=s), k=k,
                         periods_per_year=periods_per_year) for s in range(random_seeds)]
    res["random"] = {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
    res["random"]["sr_std"] = float(np.std([r["sr"] for r in runs]))
    _, gt, mask = baseline_scores(data, seq, split, "oracle")
    mr = market_daily_returns(gt, mask)
    res["market"] = {"sr": sharpe(mr, periods_per_year), "irr": irr(mr), "cumret": cumulative_return(mr),
                     "mdd": max_drawdown(mr)}
    return res
