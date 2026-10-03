import numpy as np
import pytest
from hypershift.eval import forensics as F
from hypershift.eval.metrics import topk_daily_returns, topk_daily_returns_net, sharpe


def panel():
    # 7 stocks x 3 days; stock 6 masked on day 1; day 2 has an exact tie across the k=2 boundary
    pred = np.array([[.9, .1, .5], [.8, .2, .5], [.1, .9, .5], [.2, .8, .1], [.3, .3, .0], [.0, .0, .0], [.5, .95, .2]])
    gt = np.arange(21, dtype=float).reshape(7, 3) / 100
    mask = np.ones((7, 3), bool); mask[6, 1] = False
    return pred, gt, mask


def test_portfolio_matches_existing_evaluator_and_respects_mask():
    pred, gt, mask = panel()
    r, baskets = F.portfolio(pred, gt, mask, k=2)
    np.testing.assert_allclose(r, topk_daily_returns(pred, gt, mask.astype(float), 2))
    assert 6 not in baskets[1] and set(baskets[1]) == {2, 3}


def test_stable_reverse_random_tie_orders():
    pred, gt, mask = panel()
    idx = np.arange(7)
    assert list(F.select(pred[:, 2], idx, 2, "stable")) == [0, 1]       # tie 0,1,2 at .5 -> lowest index
    assert list(F.select(pred[:, 2], idx, 2, "reverse")) == [2, 1]      # tie -> highest index first
    rng = np.random.default_rng(0)
    seen = {tuple(sorted(F.select(pred[:, 2], idx, 2, "random", rng))) for _ in range(200)}
    assert seen == {(0, 1), (0, 2), (1, 2)}


def test_boundary_stats_margin_and_exact_tie():
    pred, _, mask = panel()
    b = F.boundary_stats(pred, mask, k=2)
    assert b["s_k"][0] == .8 and b["s_k1"][0] == .5 and b["margin"][0] == pytest.approx(.3)
    assert b["exact_tie"].tolist() == [False, False, True]
    assert b["n_valid"].tolist() == [7, 6, 7]


def test_turnover_and_net_match_existing_cost_convention():
    pred, gt, mask = panel()
    r, baskets = F.portfolio(pred, gt, mask, k=2)
    to = F.turnover(baskets)
    assert to[0] == 1.0
    np.testing.assert_allclose(F.net(r, to, 10), topk_daily_returns_net(pred, gt, mask.astype(float), 2, 10))


def test_perf_uses_population_sd_and_sqrt252():
    r = np.array([.01, -.005, .02, 0.0])
    p = F.perf(r)
    assert p["sr"] == pytest.approx(r.mean() / r.std(ddof=0) * np.sqrt(252))
    assert p["sr"] == pytest.approx(sharpe(r))
    assert p["n"] == 4


def test_fewer_than_k_valid_stocks():
    pred = np.array([[.3], [.2], [.1]]); gt = np.array([[.01], [.02], [.03]])
    mask = np.array([[True], [False], [False]])
    r, b = F.portfolio(pred, gt, mask, k=5)
    assert list(b[0]) == [0] and r[0] == pytest.approx(.01)
