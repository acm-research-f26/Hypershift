"""Optional: 3-class movement classification (paper Table II 'Clf', NASDAQ), macro-F1."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from hypershift.config import RunConfig, dump_json
from hypershift.models.think import THINK
from hypershift.train.loop import apply_input_mode, gather_batch, prepare, set_seed, window_offsets


def tertile_thresholds(data, seq) -> np.ndarray:
    _, m, _, g = gather_batch(data, window_offsets(data, seq, "train"), seq)
    return np.quantile(g[m > 0.5], [1 / 3, 2 / 3])


def _f1(model, data, thg, cfg, split, th, device) -> tuple[float, float]:
    model.eval()
    ys, ps = [], []
    offs = window_offsets(data, cfg.seq, split)
    with torch.no_grad():
        for i in range(0, len(offs), max(1, cfg.batch_days)):
            x, m, _, g = gather_batch(data, offs[i:i + max(1, cfg.batch_days)], cfg.seq)
            x = apply_input_mode(x, cfg.input_mode)
            pred = model(torch.as_tensor(x, device=device), thg).argmax(-1).cpu().numpy()
            keep = m > 0.5
            ys.append(np.digitize(g, th)[keep])
            ps.append(pred[keep])
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(f1_score(y, p, average="macro")), float(f1_score(y, p, average="micro"))


def train_clf_run(cfg: RunConfig, data=None, hg=None) -> dict:
    out = cfg.run_dir()
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data, hg = prepare(cfg, data, hg)
    thg = hg.to_torch(device)
    model = THINK(in_dim=data.features.shape[2], hidden=cfg.hidden, seq=cfg.seq, kernel=cfg.kernel,
                  temporal=cfg.temporal, spatial=cfg.spatial, structure=cfg.structure,
                  attn_score=cfg.attn_score, attn_dist=cfg.attn_dist, out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    th = tertile_thresholds(data, cfg.seq)
    offs = window_offsets(data, cfg.seq, "train")
    rng = np.random.default_rng(cfg.seed)
    best, bad = None, 0
    for epoch in range(cfg.epochs):
        model.train()
        rng.shuffle(offs)
        for i in range(0, len(offs), cfg.batch_days):
            x, m, _, g = gather_batch(data, offs[i:i + cfg.batch_days], cfg.seq)
            x = apply_input_mode(x, cfg.input_mode)
            logits = model(torch.as_tensor(x, device=device), thg)
            y = torch.as_tensor(np.digitize(g, th), device=device)
            mt = torch.as_tensor(m, device=device).reshape(-1)
            loss = (F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none") * mt).sum() / mt.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        (v, vmi), (t, tmi) = _f1(model, data, thg, cfg, "val", th, device), _f1(model, data, thg, cfg, "test", th, device)
        if best is None or v > best["val_f1"]:   # selection on validation macro-F1
            best, bad = {"best_epoch": epoch, "val_f1": v, "test_f1": t, "val_micro_f1": vmi, "test_micro_f1": tmi}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    dump_json({**best, "config": cfg.to_dict()}, out / "metrics.json")
    return best
