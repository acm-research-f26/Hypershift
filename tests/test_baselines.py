import numpy as np
from hypershift.data.universe import select_universe
from hypershift.eval.baselines import evaluate_baselines, market_daily_returns


def test_baselines_ordering(synthetic_market):
    res = evaluate_baselines(synthetic_market, seq=8, random_seeds=5)
    assert set(res) == {"random", "momentum", "reversal", "market", "oracle"}
    assert res["oracle"]["sr"] > res["random"]["sr"]
    assert res["oracle"]["irr"] >= res["momentum"]["irr"]


def test_baselines_same_universe(synthetic_market, synthetic_hypergraph):
    sub, _ = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    full = evaluate_baselines(synthetic_market, seq=8, random_seeds=5)
    part = evaluate_baselines(sub, seq=8, random_seeds=5)
    assert full["oracle"]["irr"] >= part["oracle"]["irr"]          # bigger universe -> better oracle top-5


def test_market_returns_equal_weight():
    gt = np.array([[0.02, 0.0], [0.04, 0.1]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(market_daily_returns(gt, mask), [0.03, 0.1])
