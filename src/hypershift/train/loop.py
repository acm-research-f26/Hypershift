from __future__ import annotations

import functools
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hypershift.config import RunConfig, dump_json
from hypershift.data.hypergraph import (
    Hypergraph, build_rsr_hypergraph, canonical, clique_expand, correlation_hyperedges,
    decompose, drop_hub_edges, random_like,
)
from hypershift.data.rsr import MarketData, load_rsr
from hypershift.data.universe import select_universe
from hypershift.eval.metrics import evaluate_all, topk_daily_returns
from hypershift.models.think import THINK
from hypershift.train.loss import rank_mse_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_offsets(data: MarketData, seq: int, split: str) -> np.ndarray:
    lo, hi = {"train": (seq, data.valid_index), "val": (data.valid_index, data.test_index),
              "test": (data.test_index, data.num_steps)}[split]
    offsets = np.arange(max(lo, seq), hi) - seq
    if data.eligible_ends is not None:
        offsets = offsets[np.isin(offsets + seq - 1, data.eligible_ends)]
    return offsets


def gather_batch(data: MarketData, offsets: np.ndarray, seq: int):
    offsets = np.asarray(offsets)
    idx = offsets[:, None] + np.arange(seq)[None, :]
    x = data.features[:, idx].transpose(1, 0, 2, 3)
    midx = offsets[:, None] + np.arange(seq + 1)[None, :]
    mask = data.mask[:, midx].min(axis=2).T
    base = data.base_price[:, offsets + seq - 1].T
    gt = data.gt[:, offsets + seq].T
    return (np.ascontiguousarray(x, dtype=np.float32), mask.astype(np.float32),
            base.astype(np.float32), gt.astype(np.float32))


def apply_input_mode(x: np.ndarray, mode: str) -> np.ndarray:
    """x: [B, N, seq, C] with last feature = close. 'relative' divides every feature of a window by
    that window's last-day close and subtracts 1 (stationary, near 0 -> near the ball origin)."""
    if mode == "level":
        return np.asarray(x, dtype=np.float32)
    if mode == "relative":
        den = x[:, :, -1:, -1:]
        den = np.where(den <= 1e-8, 1.0, den)
        return (x / den - 1.0).astype(np.float32)
    raise ValueError(f"unknown input_mode: {mode!r}")


@functools.lru_cache(maxsize=4)
def _load_market_cached(market: str, data_root: str, norm: str, fresh_name: str) -> MarketData:
    """One parse per process: run_grid trains hundreds of runs on the same market. Never mutate the result."""
    if market in ("NYSE", "NASDAQ"):
        return load_rsr(data_root, market, norm)
    if market == "FRESH":
        from hypershift.data.fresh import load_panel
        return load_panel(Path("data/fresh") / fresh_name)
    raise ValueError(market)


def load_market(cfg: RunConfig) -> MarketData:
    return _load_market_cached(cfg.market, cfg.data_root, cfg.norm, cfg.fresh_name)


def base_hypergraph(cfg: RunConfig, data: MarketData) -> Hypergraph:
    srcs = tuple(cfg.sources)
    edges: list[tuple[int, ...]] = []
    rsr = tuple(s for s in srcs if s in ("industry", "wiki"))
    if rsr:
        edges += build_rsr_hypergraph(cfg.data_root, cfg.market, rsr).edges
    if "corr" in srcs:
        tr = slice(1, data.valid_index)
        edges += correlation_hyperedges(data.gt[:, tr], data.mask[:, tr], cfg.corr_clusters).edges
    for level in ("sector", "subindustry"):
        if level in srcs:
            from hypershift.data.fresh import gics_hypergraph
            edges += gics_hypergraph(Path("data/fresh") / cfg.fresh_name, level).edges
    hg = Hypergraph(data.num_nodes, canonical(edges))
    if "random" in srcs:
        hg = random_like(hg, seed=1000 + cfg.seed)
    return hg


def shuffle_train_labels(data: MarketData, seed: int) -> MarketData:
    """Leakage null (D10.1): permute returns across the observed stocks of each training day.

    Unlike a permutation of days, this also destroys stock-level drift, so no learnable signal remains.
    """
    rng = np.random.default_rng(seed)
    gt = data.gt.copy()
    for t in range(data.valid_index):
        ok = np.nonzero(data.mask[:, t] > 0)[0]
        gt[ok, t] = data.gt[rng.permutation(ok), t]
    return replace(data, gt=gt)


def prepare(cfg: RunConfig, data: MarketData | None = None, hg: Hypergraph | None = None):
    data = data if data is not None else load_market(cfg)
    hg = hg if hg is not None else base_hypergraph(cfg, data)
    data, hg = select_universe(data, hg, cfg.universe_size, cfg.universe_seed)
    hg = decompose(hg, cfg.decompose_mode, cfg.decompose_size)
    if cfg.structure == "clique":
        hg = clique_expand(hg)
    hg = drop_hub_edges(hg, cfg.drop_hub_degree)
    if cfg.shuffle_train_labels:
        data = shuffle_train_labels(data, cfg.seed)
    return data, hg


def build_model(cfg: RunConfig, in_dim: int, data: MarketData | None = None):
    if cfg.model == "rsr_i":
        from hypershift.models.baselines import RSRI, relation_entries
        pi, pj, ep, ec, K = relation_entries(cfg.data_root, cfg.market, data.tickers)
        return RSRI(in_dim, cfg.hidden, pi, pj, ep, ec, K)
    if cfg.model == "sthgcn":
        from hypershift.models.baselines import STHGCN
        return STHGCN(in_dim, cfg.seq, data.num_nodes)
    if cfg.model != "think":
        raise ValueError(f"unknown model {cfg.model!r}")
    return THINK(in_dim=in_dim, hidden=cfg.hidden, seq=cfg.seq, kernel=cfg.kernel, temporal=cfg.temporal,
                 spatial=cfg.spatial, structure=cfg.structure, attn_score=cfg.attn_score, attn_dist=cfg.attn_dist)


def _to_return(out, base, target):
    return out if target == "return" else (out - base) / base


@torch.no_grad()
def predict_split(model, data, thg, cfg, split, device):
    model.eval()
    offs = window_offsets(data, cfg.seq, split)
    step = max(1, cfg.micro_batch_days or cfg.batch_days)
    preds, gts, masks = [], [], []
    for i in range(0, len(offs), step):
        x, m, b, g = gather_batch(data, offs[i:i + step], cfg.seq)
        x = apply_input_mode(x, cfg.input_mode)
        out = model(torch.as_tensor(x, device=device), thg)
        preds.append(_to_return(out, torch.as_tensor(b, device=device), cfg.target).cpu().numpy())
        gts.append(g)
        masks.append(m)
    return np.concatenate(preds).T, np.concatenate(gts).T, np.concatenate(masks).T


def train_one_run(cfg: RunConfig, data: MarketData | None = None, hg: Hypergraph | None = None) -> dict:
    out = cfg.run_dir()
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    dump_json(cfg.to_dict(), out / "config.json")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data, hg = prepare(cfg, data, hg)
    thg = hg.to_torch(device)
    model = build_model(cfg, data.features.shape[2], data).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_offs = window_offsets(data, cfg.seq, "train")
    rng = np.random.default_rng(cfg.seed)
    best, bad, epoch_secs, test_srs = None, 0, [], []
    with open(out / "history.jsonl", "w") as hist:
        for epoch in range(cfg.epochs):
            t0 = time.time()
            model.train()
            rng.shuffle(train_offs)
            losses = []
            micro = cfg.micro_batch_days if cfg.micro_batch_days > 0 else cfg.batch_days
            for i in range(0, len(train_offs), cfg.batch_days):
                batch = train_offs[i:i + cfg.batch_days]
                opt.zero_grad()
                step_loss = 0.0
                for j in range(0, len(batch), micro):   # gradient accumulation == one unsplit step
                    chunk = batch[j:j + micro]
                    x, m, b, g = gather_batch(data, chunk, cfg.seq)
                    x = apply_input_mode(x, cfg.input_mode)
                    x, m, b, g = (torch.as_tensor(a, device=device) for a in (x, m, b, g))
                    pred = _to_return(model(x, thg), b, cfg.target)
                    loss, _, _ = rank_mse_loss(pred, g, m, cfg.alpha)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss: epoch {epoch}, step {i} (see decision node D5)")
                    (loss * (len(chunk) / len(batch))).backward()
                    step_loss += loss.item() * len(chunk) / len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(step_loss)
            vp, vg, vm = predict_split(model, data, thg, cfg, "val", device)
            tp, tg, tm = predict_split(model, data, thg, cfg, "test", device)
            vmet = evaluate_all(vp, vg, vm, cfg.topk, cfg.periods_per_year)
            tmet = evaluate_all(tp, tg, tm, cfg.topk, cfg.periods_per_year)
            sec = time.time() - t0
            epoch_secs.append(sec)
            test_srs.append(tmet["sr"])
            hist.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "sec": sec,
                                   "val": vmet, "test": tmet}) + "\n")
            hist.flush()
            if best is None or vmet["sr"] > best["val"]["sr"]:
                best = {"best_epoch": epoch, "val": vmet, "test": tmet}
                bad = 0
                np.save(out / "test_pred.npy", tp)
                np.save(out / "test_gt.npy", tg)
                np.save(out / "test_mask.npy", tm)
                np.save(out / "test_daily.npy", topk_daily_returns(tp, tg, tm, cfg.topk))
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    metrics = {
        **best,
        "test_oracle_sr": float(max(test_srs)),
        "test_oracle_epoch": int(np.argmax(test_srs)),
        "epochs_run": len(test_srs),
        "sec_per_epoch": float(np.mean(epoch_secs)),
        "num_nodes": data.num_nodes,
        "num_edges": len(hg.edges),
        "covered_frac": float((hg.node_degree() > 0).mean()),     # share of stocks in >= 1 hyperedge
        "config": cfg.to_dict(),
    }
    dump_json(metrics, out / "metrics.json")
    return metrics
