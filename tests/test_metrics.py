import math
import numpy as np
import pytest
from hypershift.eval.metrics import (
    cumulative_return, evaluate_all, irr, masked_mse, max_drawdown, ndcg_at_k, sharpe,
    topk_daily_returns, topk_daily_returns_net,
)


def test_topk_picks_highest_predictions():
    pred = np.array([[0.9, 0.1], [0.5, 0.8], [0.1, 0.9]])
    gt = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    r = topk_daily_returns(pred, gt, np.ones_like(gt), k=2)
    np.testing.assert_allclose(r, [(0.01 + 0.03) / 2, (0.04 + 0.06) / 2])


def test_topk_skips_masked():
    pred = np.array([[9.0], [1.0], [0.5]])
    gt = np.array([[0.5], [0.01], [0.02]])
    mask = np.array([[0.0], [1.0], [1.0]])
    np.testing.assert_allclose(topk_daily_returns(pred, gt, mask, k=1), [0.01])


def test_sharpe_matches_sthan_constant():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * math.sqrt(252))
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * 15.8745, rel=1e-4)
    assert sharpe(np.zeros(5)) == 0.0


def test_irr_cumret_mdd():
    r = np.array([0.1, -0.5, 0.2])
    assert irr(r) == pytest.approx(-0.2)
    assert cumulative_return(r) == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_ndcg_perfect_and_reversed():
    gt = np.array([[0.05], [0.03], [0.01], [-0.02], [0.0], [0.04]])
    m = np.ones_like(gt)
    assert ndcg_at_k(gt, gt, m, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(-gt, gt, m, k=5) < 0.6


def test_mse_masked():
    assert masked_mse(np.array([[1.0, 5.0]]), np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])) == pytest.approx(1.0)


def test_costs_reduce_returns_by_turnover():
    pred = np.array([[1.0, 1.0], [0.0, 0.0]])
    gt = np.array([[0.01, 0.01], [0.0, 0.0]])
    net = topk_daily_returns_net(pred, gt, np.ones_like(gt), k=1, cost_bps=10)
    np.testing.assert_allclose(net, [0.01 - 2 * 10e-4, 0.01])   # day 1 full turnover, day 2 none


def test_evaluate_all_keys():
    rng = np.random.default_rng(0)
    p, g = rng.normal(size=(20, 30)), rng.normal(0, 0.01, size=(20, 30))
    out = evaluate_all(p, g, np.ones_like(g))
    assert set(out) == {"sr", "irr", "cumret", "mdd", "ann_vol", "ndcg5", "ndcg_sthan", "mse", "n_days"}
    assert out["n_days"] == 30


def test_random_tiebreak_matches_default_without_ties_and_differs_with_ties():
    from hypershift.eval.metrics import evaluate_all, evaluate_random_ties
    rng = np.random.default_rng(0)
    p, g = rng.normal(size=(30, 40)), rng.normal(0, 0.01, size=(30, 40))
    m = np.ones_like(g)
    base, rt = evaluate_all(p, g, m), evaluate_random_ties(p, g, m, draws=3)
    assert rt["sr_rt"] == pytest.approx(base["sr"]) and rt["sr_rt_sd"] == pytest.approx(0, abs=1e-12)
    assert rt["ndcg5_rt"] == pytest.approx(base["ndcg5"])          # sklearn NDCG equals manual NDCG when there are no ties
    # constant prediction: default picks the lowest indices; random tie-break averages over all stocks
    c = np.zeros_like(g)
    d = evaluate_all(c, g, m)["sr"]
    r = evaluate_random_ties(c, g, m, draws=50)
    assert r["sr_rt"] != pytest.approx(d) and r["sr_rt_sd"] > 0
    # ndcg: random tie-break expectation ~ sklearn's tie-averaged value
    assert r["ndcg5_rt"] == pytest.approx(evaluate_all(c, g, m)["ndcg5"], abs=0.03)
    # deterministic given the seed; default untouched
    assert evaluate_random_ties(c, g, m, draws=5, seed=1) == evaluate_random_ties(c, g, m, draws=5, seed=1)
    assert evaluate_all(c, g, m)["sr"] == d
