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


def test_tie_groups_gap_chaining_is_transitive():
    s = np.array([1.0, 0.95, 0.90, 0.5, 0.49, 0.0])
    assert F.tie_groups(s, 0.06).tolist() == [0, 0, 0, 1, 1, 2]     # 1.0~0.95~0.90 chain into one group
    assert F.tie_groups(s, 0.0).tolist() == [0, 1, 2, 3, 4, 5]


def test_select_eps_keeps_clear_winners_and_randomises_boundary_group():
    scores = np.array([.9, .5, .5005, .4995, .1]); idx = np.arange(5)
    rng = np.random.default_rng(0)
    picks = {tuple(sorted(F.select_eps(scores, idx, 2, 0.001, rng))) for _ in range(300)}
    assert all(0 in p for p in picks) and {p[1] for p in picks} == {1, 2, 3}


def test_zero_spread_day_jitter_is_zero_and_eps_frac_randomises_all():
    pred = np.full((6, 1), .3); mask = np.ones((6, 1), bool)
    np.testing.assert_array_equal(F.jitter(pred, mask, 0.5, np.random.default_rng(0)), pred)
    rng = np.random.default_rng(1)
    seen = {tuple(sorted(F.portfolio_eps(pred, np.zeros((6, 1)), mask, 2, eps_frac=0.1, rng=rng)[1][0])) for _ in range(300)}
    assert len(seen) == 15                                              # C(6,2): all pairs reachable


def test_eps_draws_matches_reference_and_random_ties():
    pred, gt, mask = panel()
    out = F.eps_draws(pred, gt, mask, 2, 400, np.random.default_rng(0), eps_abs=0.0)
    assert out["returns"].shape == (400, 3)
    np.testing.assert_allclose(out["returns"][:, :2], np.broadcast_to(F.portfolio(pred, gt, mask, 2)[0][:2], (400, 2)))
    day2 = {round(x, 10) for x in out["returns"][:, 2]}                 # 3-way tie at .5 -> 3 possible pairs
    assert day2 == {round(gt[list(p), 2].mean(), 10) for p in ((0, 1), (0, 2), (1, 2))}
    assert out["group_size"].tolist() == [1, 1, 3] and out["amb_frac"] == pytest.approx(1 / 3)


def test_jitter_is_seed_deterministic():
    pred = np.random.default_rng(3).normal(size=(50, 4)); mask = np.ones_like(pred, bool)
    a = F.jitter(pred, mask, 0.25, np.random.default_rng(7)); b = F.jitter(pred, mask, 0.25, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_affine_rescaling_leaves_selection_unchanged():
    pred, gt, mask = panel()
    _, b0 = F.portfolio(pred, gt, mask, k=2)
    _, b1 = F.portfolio(1000.0 * pred + 7.0, gt, mask, k=2)
    assert all((x == y).all() for x, y in zip(b0, b1))


def test_evaluator_permutation_preserves_identity_without_ties_and_can_change_with_ties():
    pred, gt, mask = panel()
    perm = np.array([6, 5, 4, 3, 2, 1, 0])
    p2, g2, m2 = F.evaluator_permutation(pred, gt, mask, perm)
    r0, b0 = F.portfolio(pred, gt, mask, k=2); r1, b1 = F.portfolio(p2, g2, m2, k=2)
    assert set(perm[b1[0]]) == set(b0[0]) and r1[0] == r0[0]           # day 0: no tie at boundary
    assert set(perm[b1[2]]) != set(b0[2])                              # day 2: 3-way tie -> lowest *new* index wins


def test_momentum_and_vol_are_causal():
    close = np.cumprod(np.full((2, 40), 1.01), axis=1); close[1] = 1.0
    m = F.momentum_scores(close, np.array([30]), lb=20)
    assert m[0, 0] == pytest.approx(close[0, 29] / close[0, 9] - 1) and m[1, 0] == 0.0
    close2 = close.copy(); close2[:, 30:] *= 5                          # future change must not move the features
    np.testing.assert_array_equal(F.momentum_scores(close2, np.array([30]), lb=20), m)
    np.testing.assert_array_equal(F.rolling_vol(close2, np.array([30])), F.rolling_vol(close, np.array([30])))
    assert F.rolling_vol(close, np.array([30]))[1, 0] == 0.0


def test_daily_spearman_and_projection():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(50, 4)); mask = np.ones_like(f, bool)
    pred = 3 * f + 1
    np.testing.assert_allclose(F.daily_spearman(pred, f, mask), 1.0)
    fitted, resid = F.project_scores(pred, {"f": f}, mask)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)
    assert np.isnan(F.daily_spearman(np.zeros_like(f), f, mask)).all()


def test_graph_degree():
    assert F.graph_degree(((0, 1, 2), (2, 3)), 5).tolist() == [1, 1, 2, 1, 0]


def test_empirical_p_formula():
    null = np.arange(99, dtype=float)
    assert F.empirical_p(null, 98.0) == pytest.approx(2 / 100)
    assert F.empirical_p(null, 1000.0) == pytest.approx(1 / 100)
    assert F.empirical_p(null, -1.0, tail="lower") == pytest.approx(1 / 100)


def test_nulls_preserve_mask_and_cross_section_and_are_reproducible():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(30, 12)); mask = np.ones((30, 12), bool); mask[:5, 3] = False; mask[29, :] = False
    a = F.null_random_topk(gt, mask, 3, 400, np.random.default_rng(1))
    b = F.null_random_topk(gt, mask, 3, 400, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    valid_mean = np.array([gt[mask[:, d], d].mean() for d in range(12)])
    np.testing.assert_allclose(a.mean(0), valid_mean, atol=0.25)            # unbiased for the valid cross-section mean
    f = F.null_fixed(gt, mask, 3, 200, np.random.default_rng(2))
    assert f.shape == (200, 12)                                               # stock 29 never valid -> never drawn (no NaN)
    assert np.isfinite(f).all()
    pred = rng.normal(size=(30, 12))
    lp = F.null_label_perm(pred, gt, mask, 3, 50, np.random.default_rng(3))
    assert lp.shape == (50, 12) and np.isfinite(lp).all()


def test_matched_null_keeps_strata_counts():
    gt = np.random.default_rng(0).normal(size=(10, 3)); mask = np.ones((10, 3), bool)
    strata = np.array([0] * 5 + [1] * 5)
    baskets = [np.array([0, 1, 5])] * 3
    R = F.null_matched(baskets, gt, mask, strata, 100, np.random.default_rng(4))
    assert R.shape == (100, 3)


def test_stale_runs_flags_unchanged_close():
    c = np.array([[1, 1, 1, 1, 2], [1, 2, 3, 4, 5]], float)
    s = F.stale_runs(c, min_len=3)
    assert s[0].tolist() == [False, False, True, True, False] and not s[1].any()


def test_duplicate_rows_detects_identical_series():
    x = np.random.default_rng(0).normal(size=(4, 10, 5)); x[3] = x[1]
    assert F.duplicate_rows(x) == [(1, 3)]


def test_persistence_helpers():
    b = [np.array([0, 1]), np.array([0, 1]), np.array([0, 2]), np.array([3, 4])]
    np.testing.assert_allclose(F.jaccard_series(b), [1.0, 1 / 3, 0.0])
    assert sorted(F.durations(b).tolist()) == [1, 1, 1, 2, 3]
    f = F.selection_freq(b, 5)
    assert f.tolist() == [3, 2, 1, 1, 1] and F.top_share(f, 1) == pytest.approx(3 / 8)


def test_topk_diag_on_hand_panel():
    gt = np.array([[.05], [.04], [.03], [.02], [.01], [0.], [-.01], [-.02], [-.03], [-.04]])
    mask = np.ones_like(gt, bool)
    perfect = F.topk_diag(gt.copy(), gt, mask, 2)
    assert perfect["prec_at_k"] == 1.0 and perfect["hit_top10"] == 0.5 and perfect["hit_top20"] == 1.0
    assert perfect["ndcg_k"] == pytest.approx(1.0)
    worst = F.topk_diag(-gt, gt, mask, 2)
    assert worst["prec_at_k"] == 0.0 and worst["hit_top20"] == 0.0


def test_local_ic_and_calibration():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(200, 5)); mask = np.ones_like(gt, bool)
    assert F.local_ic(gt.copy(), gt, mask, 0.2) == pytest.approx(1.0)
    cal = F.calibration(gt.copy(), gt, mask, 10)
    assert np.all(np.diff(cal) > 0)
    const = np.zeros_like(gt)
    assert F.local_ic(const, gt, mask, 0.2) == 0.0


def test_margin_buckets_exact_tie_bucket_separate():
    margin = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]); r = np.arange(12) / 100
    rows = F.margin_buckets(margin, r, 5)
    assert rows[0]["bucket"] == "exact_tie" and rows[0]["n"] == 2 and sum(x["n"] for x in rows) == 12
