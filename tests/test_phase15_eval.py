"""Phase 1.5 fidelity audit, workstream A: evaluator, target alignment, normalization leakage.

Spec and findings: docs/phase1_5/A_evaluator.md. Real-data tests are marked `data` and skip when
data/raw/rsr/data is absent.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hypershift.data.rsr import MISSING, load_rsr, parse_eod
from hypershift.eval.metrics import ndcg_at_k, sharpe, topk_daily_returns
from hypershift.train.loop import apply_input_mode, gather_batch, window_offsets

RSR_ROOT = Path("data/raw/rsr/data")
needs_data = pytest.mark.skipif(not (RSR_ROOT / "2013-01-01").exists(), reason="RSR data not downloaded")


# ---------------------------------------------------------------- synthetic RSR-format panel
def _write_panel(root: Path, raw_close: dict[str, np.ndarray], full_max_norm: bool) -> None:
    """Write RSR-format files: columns [date_idx, ma5, ma10, ma20, ma30, close]. MAs are trailing (past only).
    full_max_norm=True mimics the shipped files (every column divided by the stock's FULL-series max close)."""
    (root / "2013-01-01").mkdir(parents=True, exist_ok=True)
    (root / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv").write_text("\n".join(raw_close) + "\n")
    for tk, c in raw_close.items():
        cols = []
        for w in (5, 10, 20, 30):
            ma = np.array([c[max(0, t - w + 1): t + 1].mean() for t in range(len(c))])
            cols.append(ma)
        arr = np.stack(cols + [c], axis=1)
        if full_max_norm:
            arr = arr / c.max()
        out = np.concatenate([np.arange(len(c))[:, None].astype(float), arr], axis=1)
        np.savetxt(root / "2013-01-01" / f"NYSE_{tk}_1.csv", out, delimiter=",", fmt="%.10f")


def _prices(T=1100, N=4, seed=0):
    rng = np.random.default_rng(seed)
    return {f"S{i}": 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, T)) for i in range(N)}


def _perturb(prices, start, factor=3.0):
    out = {k: v.copy() for k, v in prices.items()}
    out["S0"][start:] *= factor          # future RAW price level changes (a later rally)
    out["S1"][start] *= 1.5              # and a single future spike
    return out


def _feat_changed(a, b, upto):
    return not np.allclose(a.features[:, :upto], b.features[:, :upto], rtol=1e-6, atol=0)


@pytest.mark.parametrize("start", [800, 1008])   # after valid_index (756) / at test_index
def test_norm_leak_paper_exists_train_none(tmp_path, start):
    base = _prices()
    pert = _perturb(base, start)
    for tag, p in (("a", base), ("b", pert)):
        _write_panel(tmp_path / tag, p, full_max_norm=True)     # shipped format
    pa, pb = (load_rsr(tmp_path / t, "NYSE", "paper") for t in "ab")
    ta, tb = (load_rsr(tmp_path / t, "NYSE", "train") for t in "ab")
    # paper norm: features of days BEFORE the perturbation depend on future raw prices (the leak)
    assert _feat_changed(pa, pb, start), "paper norm should leak future raw prices into earlier inputs"
    # the leak is a per-stock rescale: stock S2 (unperturbed) is untouched, S0 changes by a constant factor
    np.testing.assert_allclose(pa.features[2], pb.features[2])
    ratio = pb.features[0, :start, -1] / pa.features[0, :start, -1]
    assert np.allclose(ratio, ratio[0]) and abs(ratio[0] - 1) > 0.05
    # train norm: everything before the perturbation is unchanged, for every stock and feature
    np.testing.assert_allclose(ta.features[:, :start], tb.features[:, :start], rtol=1e-6)
    np.testing.assert_allclose(ta.base_price[:, :start], tb.base_price[:, :start], rtol=1e-6)
    np.testing.assert_allclose(ta.gt[:, :start], tb.gt[:, :start], atol=1e-6)
    # and through the window layer: every val/test window whose inputs and target lie before `start`
    for split in ("val", "test"):
        offs = window_offsets(ta, 16, split)
        offs = offs[offs + 16 < start]
        if len(offs) == 0:
            continue
        xa, ma, _, ga = gather_batch(ta, offs, 16)
        xb, mb, _, gb = gather_batch(tb, offs, 16)
        for mode in ("level", "relative"):
            np.testing.assert_allclose(apply_input_mode(xa, mode), apply_input_mode(xb, mode), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga, gb, atol=1e-6)


def test_norm_train_scale_uses_only_train_period(tmp_path):
    """Perturbing a price AFTER valid_index must not change the train-norm scale; perturbing one inside the
    training period does (documented caveat: within-train look-ahead of the scale, not val/test leakage)."""
    base = _prices()
    _write_panel(tmp_path / "a", base, True)
    _write_panel(tmp_path / "b", _perturb(base, 757), True)
    _write_panel(tmp_path / "c", _perturb(base, 400), True)
    ta, tb, tc = (load_rsr(tmp_path / t, "NYSE", "train") for t in "abc")
    np.testing.assert_allclose(ta.features[:, :757], tb.features[:, :757], rtol=1e-6)
    # inside-train perturbation: days before 400 are rescaled by the (train-period) max, a within-train effect only
    assert _feat_changed(ta, tc, 400)


# ---------------------------------------------------------------- parse_eod: missing prices
def test_missing_prices_never_create_returns_or_eligibility():
    close = np.array([1.0, 1.1, MISSING, 1.3, 1.4, 1.5], dtype=np.float64)
    raw = np.stack([np.arange(6.0), close, close, close, close, close], axis=1)
    feats, mask, gt, base = parse_eod(raw, drop_last=False)
    assert mask.tolist() == [1, 1, 0, 1, 1, 1]
    # return into and out of the missing day is 0 (not computed through the gap) ...
    assert gt[2] == 0 and gt[3] == 0
    assert gt[1] == pytest.approx(0.1) and gt[4] == pytest.approx(1.4 / 1.3 - 1)
    assert (feats[2] == 1.1).all()           # FILL, same as the authors' code
    # ... and any window that touches the gap is ineligible, so those zero returns are never scored
    from hypershift.data.rsr import MarketData
    md = MarketData(["X"], feats[None], mask[None], gt[None], base[None], 4, 5)
    _, m, _, g = gather_batch(md, np.array([0, 1]), seq=2)   # windows [0,1]->t2, [1,2]->t3 touch the gap
    assert m.tolist() == [[0.0], [0.0]]
    _, m2, _, g2 = gather_batch(md, np.array([2]), seq=2)    # inputs 2,3 -> 2 is missing
    assert m2.tolist() == [[0.0]]
    _, m3, _, g3 = gather_batch(md, np.array([3]), seq=2)    # inputs 3,4, target 5: fully observed
    assert m3.tolist() == [[1.0]] and g3[0, 0] == pytest.approx(1.5 / 1.4 - 1)


# ---------------------------------------------------------------- alignment, synthetic
def test_prediction_at_window_end_is_scored_on_next_day_return(synthetic_market):
    m, seq = synthetic_market, 8
    close = m.base_price
    for split in ("train", "val", "test"):
        offs = window_offsets(m, seq, split)
        x, mk, base, gt = gather_batch(m, offs, seq)
        for k, o in enumerate(offs[:5]):
            t = o + seq - 1                                   # last input day
            np.testing.assert_allclose(base[k], close[:, t])  # base price = close at window end
            ok = mk[k] > 0.5
            np.testing.assert_allclose(gt[k][ok], close[ok, t + 1] / close[ok, t] - 1, rtol=1e-4, atol=1e-6)
    tr, va, te = (window_offsets(m, seq, s) for s in ("train", "val", "test"))
    assert tr.max() + seq < m.valid_index and va.min() + seq == m.valid_index and te.min() + seq == m.test_index
    assert te.max() + seq == m.num_steps - 1                   # last scored target = last day
    # val/test windows reach back into the previous period for their inputs (history), targets never do
    assert va.min() < m.valid_index and te.min() < m.test_index


# ---------------------------------------------------------------- evaluator vs authors' loop
def _authors_daily(pred, gt, mask):
    """Verbatim logic of the STHAN-SR evaluator (training/evaluator.py, commit 8d7861c) for top-5."""
    out = []
    for i in range(pred.shape[1]):
        rank_pre = np.argsort(pred[:, i])
        top = set()
        for j in range(1, pred.shape[0] + 1):
            cur = rank_pre[-1 * j]
            if mask[cur][i] < 0.5:
                continue
            if len(top) < 5:
                top.add(cur)
        out.append(sum(gt[p][i] for p in top) / 5)
    return np.array(out)


def test_sharpe_matches_authors_up_to_annualisation_constant():
    rng = np.random.default_rng(0)
    N, D = 40, 60
    pred, gt = rng.standard_normal((N, D)), rng.normal(0, 0.02, (N, D))
    mask = (rng.random((N, D)) > 0.1).astype(float)
    ra, ro = _authors_daily(pred, gt, mask), topk_daily_returns(pred, gt, mask, 5)
    np.testing.assert_allclose(ra, ro, atol=1e-12)                  # tie-free: identical portfolios
    authors_sr = np.mean(ra) / np.std(ra) * 15.87                   # np.std: ddof = 0
    assert sharpe(ro) == pytest.approx(authors_sr * math.sqrt(252) / 15.87)
    assert sharpe(ro) == pytest.approx(authors_sr, rel=3e-4)        # 15.87 vs sqrt(252)=15.8745


def test_tie_break_is_lowest_index_and_matters():
    """Exact ties are resolved by stable sort: lowest eligible index wins. The authors' np.argsort-from-the-end
    picks other stocks, so Sharpe of tied (near-constant) predictors is tie-rule dependent."""
    N, D = 20, 3
    gt = np.tile(np.linspace(-0.01, 0.01, N)[:, None], (1, D))     # return rises with index
    r = topk_daily_returns(np.zeros((N, D)), gt, np.ones((N, D)), 5)
    np.testing.assert_allclose(r, gt[:5, 0].mean())                  # picks indices 0..4
    rev = topk_daily_returns(np.zeros((N, D))[::-1], gt[::-1], np.ones((N, D)), 5)
    assert rev[0] > r[0] + 0.01                                      # reversed order -> a different portfolio


# ---------------------------------------------------------------- NDCG definition
def _manual_ndcg(rel, score, k):
    """Linear gain, log2(rank+1) discount, ties in score averaged (sklearn ignore_ties=False semantics)."""
    disc = 1 / np.log2(np.arange(len(rel)) + 2)
    def dcg(s):
        vals = np.unique(s)[::-1]
        total, pos = 0.0, 0
        for v in vals:
            grp = np.nonzero(s == v)[0]
            lo, hi = pos, pos + len(grp)
            take = max(0, min(hi, k) - lo)
            total += rel[grp].mean() * disc[lo:lo + take].sum()
            pos = hi
        return total
    return dcg(score) / dcg(rel)


def test_ndcg_definition_linear_gain_shifted_relevance_tie_averaged():
    rng = np.random.default_rng(1)
    N, D = 30, 4
    gt = rng.normal(0, 0.02, (N, D))
    pred = np.round(rng.normal(size=(N, D)), 1)                      # deliberate ties
    mask = (rng.random((N, D)) > 0.2).astype(float)
    exp = []
    for d in range(D):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        rel = gt[idx, d] - gt[idx, d].min()                          # negative returns -> shifted by the day's min
        exp.append(_manual_ndcg(rel, pred[idx, d], 5))
    assert ndcg_at_k(pred, gt, mask, 5) == pytest.approx(np.mean(exp))


def test_ndcg_skips_degenerate_days_and_random_baseline_same_function():
    gt = np.zeros((6, 2)); gt[:, 1] = np.arange(6)
    mask = np.ones((6, 2))
    pred = np.arange(6.0)[::-1][:, None].repeat(2, 1)               # day 0: all-equal returns -> skipped
    assert ndcg_at_k(pred, gt, mask, 5) == pytest.approx(ndcg_at_k(pred[:, 1:], gt[:, 1:], mask[:, 1:], 5))
    # a random scorer has the same expected NDCG under any relabelling of stocks (no index preference)
    rng = np.random.default_rng(0)
    g = rng.normal(size=(50, 30)); m = np.ones_like(g)
    a = np.mean([ndcg_at_k(rng.standard_normal(g.shape), g, m) for _ in range(40)])
    b = np.mean([ndcg_at_k(rng.standard_normal(g.shape), g[::-1], m) for _ in range(40)])
    assert abs(a - b) < 0.02


# ---------------------------------------------------------------- alignment, real RSR data
@pytest.fixture(scope="module")
def nyse():
    d = load_rsr(RSR_ROOT, "NYSE", "paper")
    close = np.stack([np.loadtxt(RSR_ROOT / "2013-01-01" / f"NYSE_{t}_1.csv", delimiter=",")[:, 5] for t in d.tickers])
    return d, close


@needs_data
@pytest.mark.data
def test_real_split_and_scored_dates(nyse):
    import pandas as pd
    d, _ = nyse
    dates = pd.to_datetime(pd.read_csv(RSR_ROOT / "NYSE_aver_line_dates.csv", header=None)[0]).iloc[-d.num_steps:].reset_index(drop=True)
    assert d.num_steps == 1245 and len(dates) == 1245                # date file has 29 warm-up rows before day 0
    offs = {s: window_offsets(d, 16, s) for s in ("train", "val", "test")}
    tgt = {s: (dates[o[0] + 16], dates[o[-1] + 16]) for s, o in offs.items()}
    assert len(offs["test"]) == 237 and len(offs["val"]) == 252
    assert str(tgt["test"][0].date()) == "2017-01-03" and str(tgt["test"][1].date()) == "2017-12-08"
    assert str(tgt["val"][0].date()) == "2016-01-04" and str(tgt["val"][1].date()) == "2016-12-30"
    assert str(tgt["train"][1].date()) == "2015-12-31"
    # first test window's last INPUT day is 2016-12-30, i.e. the day before the first scored target
    assert str(dates[offs["test"][0] + 15].date()) == "2016-12-30"
    # splits are target-date splits: test inputs start in the validation year, val inputs in the training years
    assert dates[offs["test"][0]].year == 2016 and dates[offs["val"][0]].year == 2015


@needs_data
@pytest.mark.data
def test_real_gt_mask_trace_to_raw_close(nyse):
    d, close = nyse
    offs = window_offsets(d, 16, "test")
    x, mk, base, gt = gather_batch(d, offs, 16)
    miss = close < -1000
    n_elig = 0
    for k, o in enumerate(offs):
        t = o + 15
        win_missing = miss[:, o:o + 17].any(axis=1)                   # 16 input days + the target day
        np.testing.assert_array_equal(mk[k] > 0.5, ~win_missing)      # eligible iff every price observed
        ok = ~win_missing
        np.testing.assert_allclose(gt[k][ok], close[ok, t + 1] / close[ok, t] - 1, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(base[k][ok], close[ok, t], rtol=1e-6)
        n_elig += int(ok.sum())
    assert n_elig > 400_000
    # no fabricated feature values on observed days
    assert not ((d.mask > 0.5) & (d.features == 1.1).any(axis=2)).any()


@needs_data
@pytest.mark.data
def test_real_daily_return_from_scratch(nyse):
    """Independent recomputation of the daily top-5 return straight from the raw close columns."""
    d, close = nyse
    offs = window_offsets(d, 16, "test")
    _, mk, _, gt = gather_batch(d, offs, 16)
    pred = np.random.default_rng(0).standard_normal((d.num_nodes, len(offs)))
    ours = topk_daily_returns(pred, gt.T, mk.T, 5)
    miss = close < -1000
    ref = []
    for k in range(len(offs)):
        t = 1007 + k                                                  # window end = day before the target
        ok = np.nonzero(~miss[:, t - 15:t + 2].any(axis=1))[0]
        top = ok[np.argsort(-pred[ok, k], kind="stable")[:5]]
        ref.append(np.mean(close[top, t + 1] / close[top, t] - 1))
    np.testing.assert_allclose(ours, ref, atol=1e-6)
    assert len(ours) == 237


# ---------------------------------------------------------------- real-data statements about the leak
@needs_data
@pytest.mark.data
def test_real_paper_norm_uses_full_series_max(nyse):
    d, close = nyse
    c = np.where(close > -1000, close, 0)
    np.testing.assert_allclose(c.max(axis=1), 1.0, rtol=1e-6)         # shipped files: max close == 1 over ALL days
    frac_peak_after_train = (c.argmax(axis=1) >= d.valid_index).mean()
    assert frac_peak_after_train > 0.4                                # for half the stocks the divisor was set after 2015
    t = load_rsr(RSR_ROOT, "NYSE", "train")
    ct = np.where(t.mask > 0, t.features[:, :, -1], 0)
    np.testing.assert_allclose(ct[:, :d.valid_index].max(axis=1), 1.0, rtol=1e-5)  # train norm: max over train == 1
