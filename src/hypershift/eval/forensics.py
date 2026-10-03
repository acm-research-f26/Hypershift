"""Phase 1.5a forensic calculations (CPU, pure numpy). Grids are predeclared; do not edit after inspecting returns."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS_ABS = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
EPS_FRAC = (0.001, 0.01, 0.05, 0.10, 0.25)
JITTER_FRAC = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
K_GRID = (1, 5, 10, 20, 50)
K_PRIMARY = 5
COST_BPS = (0, 5, 10, 25)
MARGIN_BUCKETS = 5
DROP_DAYS = (1, 3, 5, 10, 20)
DROP_STOCKS = (1, 2, 3, 5, 10)
TOP_FREQ = (1, 5, 10, 20)
LOCAL_IC_Q = (0.05, 0.10, 0.20)
CALIB_BINS = 10
B_NULL, B_PERM, R_TIE, R_JIT, N_BOOT, BLOCK, BLOCK_SENS = 10_000, 2_000, 1_000, 200, 5_000, 10, (5, 20)
SEED_RNG = 20261003
RNG_OFFSETS = {"tie": 1, "eps": 2, "jitter": 3, "null_random": 4, "null_fixed": 5, "null_perm": 6,
               "null_matched": 7, "boot": 8, "index_perm": 9}


def rng_for(name: str, extra: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED_RNG + 1000 * RNG_OFFSETS[name] + extra)


@dataclass
class RunArrays:
    pred: np.ndarray
    gt: np.ndarray
    mask: np.ndarray
    daily: np.ndarray
    metrics: dict
    config: dict
    history: list
    path: Path


def load_run(exp: str, label: str, seed: int, root: Path | str = Path("results")) -> RunArrays:
    p = Path(root) / exp / label / f"seed_{seed}"
    pred = np.load(p / "test_pred.npy").astype(np.float64)
    gt = np.load(p / "test_gt.npy").astype(np.float64)
    mask = np.load(p / "test_mask.npy") > 0.5
    if not (np.isfinite(pred[mask]).all() and np.isfinite(gt[mask]).all()):
        raise ValueError(f"non-finite pred/gt in valid cells: {p}")
    hist = [json.loads(line) for line in open(p / "history.jsonl") if line.strip()]
    return RunArrays(pred, gt, mask, np.load(p / "test_daily.npy"), json.loads((p / "metrics.json").read_text()),
                     json.loads((p / "config.json").read_text()), hist, p)


def select(scores_d, idx, k, order="stable", rng=None):
    s = scores_d[idx]
    if order == "stable":
        o = np.argsort(-s, kind="stable")
    elif order == "reverse":
        o = np.lexsort((-np.arange(len(s)), -s))           # last key primary: score desc, then index desc
    elif order == "random":
        o = np.lexsort((rng.random(len(s)), -s))
    else:
        raise ValueError(order)
    return idx[o[:k]]


def portfolio(pred, gt, mask, k=K_PRIMARY, order="stable", rng=None):
    r, baskets = np.zeros(pred.shape[1]), []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        top = select(pred[:, d], idx, k, order, rng) if len(idx) else idx
        baskets.append(top)
        r[d] = gt[top, d].mean() if len(top) else 0.0
    return r, baskets


def hold_all(gt, mask):
    return np.array([gt[mask[:, d], d].mean() if mask[:, d].any() else 0.0 for d in range(gt.shape[1])])


def boundary_stats(pred, mask, k=K_PRIMARY):
    D = pred.shape[1]
    out = {key: np.zeros(D) for key in ("s_k", "s_k1", "margin", "sd", "ptp")}
    out["exact_tie"], out["n_valid"] = np.zeros(D, bool), np.zeros(D, int)
    for d in range(D):
        s = np.sort(pred[mask[:, d], d])[::-1]
        out["n_valid"][d] = len(s)
        if len(s) == 0:
            continue
        out["sd"][d], out["ptp"][d] = s.std(), s[0] - s[-1]
        if len(s) > k:
            out["s_k"][d], out["s_k1"][d] = s[k - 1], s[k]
            out["margin"][d] = s[k - 1] - s[k]
            out["exact_tie"][d] = s[k - 1] == s[k]
    return out


def turnover(baskets):
    to, prev = np.zeros(len(baskets)), set()
    for d, b in enumerate(baskets):
        cur = set(b.tolist())
        to[d] = 1.0 if not prev else len(cur - prev) / max(len(cur), 1)
        prev = cur
    return to


def net(r, to, bps):
    return r - 2 * bps * 1e-4 * to


def perf(r):
    r = np.asarray(r, dtype=np.float64)
    sd = r.std()
    w = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return {"mean": float(r.mean()), "vol_d": float(sd), "sr": 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(252)),
            "cumret": float(w[-1] - 1) if len(w) else 0.0, "mdd": float((w / peak - 1).min()) if len(w) else 0.0,
            "n": int(len(r))}


def tie_groups(sorted_desc, eps):
    gaps = -np.diff(sorted_desc)
    return np.concatenate([[0], np.cumsum(gaps > eps)]).astype(int)


def select_eps(scores_d, idx, k, eps, rng):
    s = scores_d[idx]
    o = np.argsort(-s, kind="stable")
    if len(o) <= k:
        return idx[o]
    g = tie_groups(s[o], eps)
    gb = g[k - 1]
    above = o[g < gb]
    group = o[g == gb]
    fill = rng.choice(group, size=k - len(above), replace=False)
    return idx[np.concatenate([above, fill])]


def portfolio_eps(pred, gt, mask, k=K_PRIMARY, eps_abs=None, eps_frac=None, rng=None):
    r, baskets = np.zeros(pred.shape[1]), []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        eps = eps_abs if eps_abs is not None else eps_frac * (pred[idx, d].std() if len(idx) else 0.0)
        top = select_eps(pred[:, d], idx, k, eps, rng) if len(idx) else idx
        if eps_frac is not None and len(idx) and pred[idx, d].std() == 0:
            top = idx[rng.choice(len(idx), size=min(k, len(idx)), replace=False)]   # zero-spread: whole set tied
        baskets.append(top)
        r[d] = gt[top, d].mean() if len(top) else 0.0
    return r, baskets


def eps_draws(pred, gt, mask, k, R, rng, eps_abs=None, eps_frac=None):
    D = pred.shape[1]
    stable = portfolio(pred, gt, mask, k)[1]
    rets, jac = np.zeros((R, D)), np.zeros((R, D))
    gsize = np.zeros(D, int)
    for d in range(D):
        idx = np.nonzero(mask[:, d])[0]
        if not len(idx):
            continue
        s = pred[idx, d]
        sd = s.std()
        o = np.argsort(-s, kind="stable")
        if eps_frac is not None and sd == 0:                       # zero-spread day: whole valid set is one group
            above, group = o[:0], o
        else:
            eps = eps_abs if eps_abs is not None else eps_frac * sd
            g = tie_groups(s[o], eps)
            gb = g[min(k, len(o)) - 1]
            above, group = o[g < gb], o[g == gb]
        gsize[d] = len(group)
        need = min(k, len(o)) - len(above)
        fill = np.argsort(rng.random((R, len(group))), axis=1)[:, :need]       # R independent refills
        picks = np.concatenate([np.broadcast_to(above, (R, len(above))), group[fill]], axis=1)
        rets[:, d] = gt[idx[picks], d].mean(1)
        st = set(stable[d].tolist())
        jac[:, d] = [len(st & set(idx[p].tolist())) / len(st | set(idx[p].tolist())) for p in picks]
    return {"returns": rets, "jaccard": jac.mean(1), "amb_frac": float((gsize > 1).mean()), "group_size": gsize}


def jitter(pred, mask, frac, rng):
    out = pred.copy()
    for d in range(pred.shape[1]):
        i = mask[:, d]
        sd = pred[i, d].std() if i.any() else 0.0
        if sd > 0:
            out[i, d] = pred[i, d] + rng.normal(0.0, frac * sd, size=int(i.sum()))
    return out


def evaluator_permutation(pred, gt, mask, perm):
    """Row p of the output holds old stock perm[p]; the scores travel with their stock."""
    return pred[perm], gt[perm], mask[perm]


def sr_rows(R):
    sd = R.std(axis=1)
    return np.where(sd > 0, R.mean(1) / np.where(sd > 0, sd, 1) * math.sqrt(252), 0.0)


def graph_degree(edges, n):
    deg = np.zeros(n, int)
    for e in edges:
        deg[list(e)] += 1
    return deg


from scipy.stats import rankdata


def momentum_scores(close, target_days, lb=20):
    t = np.asarray(target_days) - 1
    return close[:, t] / close[:, t - lb] - 1


def rolling_vol(close, target_days, lb=20):
    r = np.zeros_like(close)
    r[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    return np.stack([r[:, t - lb + 1:t + 1].std(axis=1) for t in np.asarray(target_days) - 1], axis=1)


def train_beta(gt_full, mask_full, valid_index):
    g, m = gt_full[:, 1:valid_index], mask_full[:, 1:valid_index] > 0.5
    mkt = np.array([g[m[:, d], d].mean() if m[:, d].any() else 0.0 for d in range(g.shape[1])])
    beta = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        ok = m[i]
        if ok.sum() >= 250:
            x, y = mkt[ok], g[i, ok]
            beta[i] = np.cov(x, y, ddof=0)[0, 1] / x.var()
    return beta


def industry_of(tickers, json_path):
    lab = {t: ind for ind, ts in json.loads(Path(json_path).read_text()).items() for t in ts}
    return [lab.get(t, "n/a") for t in tickers]


def proxy_features(features, gt_full, mask_full, valid_index, target_days, edges):
    t = np.asarray(target_days) - 1
    close = features[:, :, 4]
    n, D = features.shape[0], len(t)
    out = {f"ma{w}_rel": features[:, t, c] / close[:, t] for c, w in ((0, 5), (1, 10), (2, 20), (3, 30))}
    out.update(ret1=momentum_scores(close, target_days, 1), ret5=momentum_scores(close, target_days, 5),
               ret20=momentum_scores(close, target_days, 20), vol20=rolling_vol(close, target_days, 20),
               close_level=close[:, t])
    out["beta"] = np.repeat(train_beta(gt_full, mask_full, valid_index)[:, None], D, axis=1)
    out["degree"] = np.repeat(graph_degree(edges, n)[:, None].astype(float), D, axis=1)
    out["index"] = np.repeat(np.arange(n, dtype=float)[:, None], D, axis=1)
    return out


def daily_spearman(a, b, mask):
    out = np.full(a.shape[1], np.nan)
    for d in range(a.shape[1]):
        i = mask[:, d] & np.isfinite(a[:, d]) & np.isfinite(b[:, d])
        if i.sum() >= 5 and np.ptp(a[i, d]) > 0 and np.ptp(b[i, d]) > 0:
            out[d] = np.corrcoef(rankdata(a[i, d]), rankdata(b[i, d]))[0, 1]
    return out


def project_scores(pred, feats, mask):
    fitted, resid = np.full_like(pred, np.nan), np.full_like(pred, np.nan)
    names = sorted(feats)
    for d in range(pred.shape[1]):
        X = np.stack([feats[k][:, d] for k in names], axis=1)
        i = mask[:, d] & np.isfinite(X).all(1)
        if i.sum() <= X.shape[1] + 1:
            continue
        Z = (X[i] - X[i].mean(0)) / np.where(X[i].std(0) > 0, X[i].std(0), 1)
        A = np.column_stack([np.ones(i.sum()), Z])
        coef, *_ = np.linalg.lstsq(A, pred[i, d], rcond=None)
        fitted[i, d] = A @ coef
        resid[i, d] = pred[i, d] - fitted[i, d]
    return fitted, resid


def null_random_topk(gt, mask, k, B, rng, chunk=500):
    """== within-day score permutation null (a uniform random k-subset of valid stocks); identical for any scores."""
    D = gt.shape[1]
    out = np.zeros((B, D))
    for d in range(D):
        idx = np.nonzero(mask[:, d])[0]
        g = gt[idx, d]
        kk = min(k, len(idx))
        for s in range(0, B, chunk):
            n = min(chunk, B - s)
            pick = np.argpartition(rng.random((n, len(idx))), kk - 1, axis=1)[:, :kk]
            out[s:s + n, d] = g[pick].mean(1)
    return out


def null_fixed(gt, mask, k, B, rng):
    """Random fixed k-baskets of stocks valid on every day, equal weight, daily rebalanced."""
    always = np.nonzero(mask.all(axis=1))[0]
    out = np.zeros((B, gt.shape[1]))
    for b in range(B):
        out[b] = gt[rng.choice(always, size=k, replace=False)].mean(0)
    return out


def null_label_perm(pred, gt, mask, k, B, rng):
    """One stock-label permutation per draw, applied on every day: keeps score persistence, breaks score->stock identity."""
    out = np.zeros((B, gt.shape[1]))
    for b in range(B):
        perm = rng.permutation(pred.shape[0])
        out[b] = portfolio(pred[perm], gt, mask, k)[0]
    return out


def null_matched(baskets, gt, mask, strata, B, rng):
    """Per day, replace each selected stock with a random valid stock of the same stratum (industry or beta bucket)."""
    out = np.zeros((B, gt.shape[1]))
    for d, bk in enumerate(baskets):
        if not len(bk):
            continue
        valid = np.nonzero(mask[:, d])[0]
        pools = {s: valid[strata[valid] == s] for s in np.unique(strata[bk])}
        draws = np.stack([rng.choice(pools[strata[i]], size=B) for i in bk], axis=1)
        out[:, d] = gt[draws, d].mean(1)
    return out


def empirical_p(null, obs, tail="upper"):
    null = np.asarray(null)
    hits = (null >= obs).sum() if tail == "upper" else (null <= obs).sum()
    return float((1 + hits) / (len(null) + 1))


def stale_runs(close, min_len=3):
    """True where close has been unchanged for >= min_len consecutive days, ending at t."""
    same = np.zeros_like(close, dtype=bool)
    same[:, 1:] = close[:, 1:] == close[:, :-1]
    run = np.zeros(close.shape, dtype=int)
    for t in range(1, close.shape[1]):
        run[:, t] = np.where(same[:, t], run[:, t - 1] + 1, 0)
    return run + 1 >= min_len


def duplicate_rows(x):
    flat = x.reshape(x.shape[0], -1)
    seen, out = {}, []
    for i, row in enumerate(flat):
        h = row.tobytes()
        if h in seen:
            out.append((seen[h], i))
        else:
            seen[h] = i
    return out
