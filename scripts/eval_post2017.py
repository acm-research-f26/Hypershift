"""Phase 1.5b Task 4: locked 2018-2023 inference of the frozen R5_f3_alpha0_train best_state.pt weights (CPU only).

  eval_post2017.py sanity    # 2017 pipeline check (seed 0 HH, Alpaca panel vs stored RSR-panel test_pred); NOT 2018+
  eval_post2017.py run       # ONE locked pass 2018-01-02..2023-12-29 -> results/post2017_frozen/ (refuses to run twice)
  eval_post2017.py handcheck # stock-day and portfolio-day recomputed from raw bars with pandas (independent path)

Panel: Alpaca adjustment=split only, 1,647 identity-pass nodes (docs/phase1_5b/alpaca_audit_tickers.csv), frozen RSR node order,
masked after the last bar, warm-up from Alpaca only (history from 2016-01-04). Graph v2 as in training.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hypershift.config import RunConfig
from hypershift.data.alpaca_panel import build_eval_panel, calendar_from_bars, daily_tie_flags, target_dates
from hypershift.data.rsr import read_ticker_file
from hypershift.eval.metrics import evaluate_all, topk_daily_returns
from hypershift.train.loop import base_hypergraph, build_model, predict_split, prepare

RSR = Path("data/raw/rsr/data")
BARS = Path("data/raw/alpaca_post2017/bars.pkl.gz")
IDENT = Path("docs/phase1_5b/alpaca_audit_tickers.csv")
SRC = Path("results/R5_f3_alpha0_train")
OUT = Path("results/post2017_frozen")
START, END = "2016-01-04", "2023-12-29"
TEST0 = "2018-01-02"
SCALE_END = "2016-12-30"          # A1: fixed per-ticker scale from the pre-2017 part of the new source only (irrelevant under input_mode=relative, R4)
ARMS, SEEDS = ("HH", "EH"), range(5)


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_panel():
    order = read_ticker_file(RSR / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    bars = pd.read_pickle(BARS, compression="gzip")["split"]       # our own local file
    tab = pd.read_csv(IDENT)
    assert tab.ticker.tolist() == order, "identity table is not in the frozen node order"
    ok = tab[tab.identity_ok]
    assert len(ok) == 1647
    series = {r.ticker: bars[r.symbol]["c"] for r in ok.itertuples()}
    cal = calendar_from_bars(bars, START, END)                     # same rule as the audit (A2 item 7)
    data = build_eval_panel(series, order, order, cal, TEST0, SCALE_END)
    return data, order, series, bars, tab


def load_model(arm, seed, data):
    cfg = RunConfig(**{k: v for k, v in json.loads((SRC / arm / f"seed_{seed}" / "config.json").read_text()).items()})
    cfg.device = "cpu"
    return cfg, build_model(cfg, data.features.shape[2], data)


def infer(cfg, model, data, hg_t, last=None):
    sd = torch.load(SRC / cfg.label / f"seed_{cfg.seed}" / "best_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return predict_split(model, data, hg_t, cfg, "test", torch.device("cpu"))


def graph(cfg, data):
    _, hg = prepare(cfg, data)
    assert hg.num_nodes == 1737 and len(hg.edges) == 4350
    return hg.to_torch(torch.device("cpu"))


def sanity():
    import dataclasses
    data, order, *_ = load_panel()
    cal = pd.DatetimeIndex(data.timestamps)
    i0, i1 = int(cal.searchsorted(pd.Timestamp("2017-01-03"))), int(cal.searchsorted(pd.Timestamp("2017-12-08"), side="right"))
    d17 = dataclasses.replace(data, features=data.features[:, :i1], mask=data.mask[:, :i1], gt=data.gt[:, :i1],
                              base_price=data.base_price[:, :i1], valid_index=i0, test_index=i0, timestamps=data.timestamps[:i1])
    cfg, model = load_model("HH", 0, d17)
    p, g, m = infer(cfg, model, d17, graph(cfg, d17))
    ref = np.load(SRC / "HH/seed_0/test_pred.npy")
    rm = np.load(SRC / "HH/seed_0/test_mask.npy")
    print("2017 days", p.shape, ref.shape)
    both = (m > 0.5) & (rm > 0.5)
    rho = np.nanmean([np.corrcoef(p[both[:, d], d], ref[both[:, d], d])[0, 1] for d in range(p.shape[1]) if both[:, d].sum() > 10])
    print("alpaca-panel 2017 metrics", {k: round(v, 4) for k, v in evaluate_all(p, g, m).items() if k in ("sr", "ndcg5", "n_days")})
    print("stored RSR-panel 2017 SR", json.loads((SRC / "HH/seed_0/metrics.json").read_text())["test"]["sr"])
    print("mean daily cross-sectional Pearson(pred_alpaca, pred_rsr) on common names", round(float(rho), 4),
          "| eligible/day alpaca", round(float(m.sum(0).mean())), "rsr", round(float(rm.sum(0).mean())))


def run():
    if (OUT / "LOCKED.json").exists():
        raise SystemExit("locked pass already done (results/post2017_frozen/LOCKED.json); refusing to run a second time")
    man = Path("docs/phase1_5b/FREEZE_MANIFEST.md")
    assert man.exists(), "freeze manifest must exist before any 2018+ inference"
    data, order, *_ = load_panel()
    dates = target_dates(data)
    assert str(dates[0])[:10] == TEST0 and str(dates[-1])[:10] == END
    OUT.mkdir(parents=True, exist_ok=True)
    cfg0, _ = load_model("HH", 0, data)
    thg = graph(cfg0, data)
    hashes = {}
    np.save(OUT / "dates.npy", dates.astype("datetime64[D]"))
    hashes["dates.npy"] = sha(OUT / "dates.npy")
    summary = {}
    for arm in ARMS:
        for seed in SEEDS:
            cfg, model = load_model(arm, seed, data)
            p, g, m = infer(cfg, model, data, thg)
            d = OUT / arm / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            daily = topk_daily_returns(p, g, m, 5)
            np.save(d / "test_pred.npy", p.astype(np.float32))
            np.save(d / "test_gt.npy", g.astype(np.float32))
            np.save(d / "test_mask.npy", m.astype(np.float32))
            np.save(d / "test_daily.npy", daily)
            np.save(d / "eligible_count.npy", m.sum(0).astype(np.int32))
            np.save(d / "tie_flag.npy", daily_tie_flags(p, m, 5))
            met = evaluate_all(p, g, m, 5, 252)
            met["tie_days"] = int(daily_tie_flags(p, m, 5).sum())
            (d / "metrics.json").write_text(json.dumps(met, indent=1))
            for f in sorted(d.iterdir()):
                hashes[f"{arm}/seed_{seed}/{f.name}"] = sha(f)
            summary[f"{arm}/{seed}"] = round(met["sr"], 4)
            print(arm, seed, round(met["sr"], 4), "ties", met["tie_days"], flush=True)
    (OUT / "OUTPUT_HASHES.json").write_text(json.dumps(hashes, indent=1))
    (OUT / "LOCKED.json").write_text(json.dumps({"done": pd.Timestamp.now("UTC").isoformat(), "sr_gross": summary}, indent=1))


def handcheck():
    data, order, series, bars, tab = load_panel()
    dates = np.load(OUT / "dates.npy")
    d = OUT / "HH/seed_0"
    p, g, m, daily = (np.load(d / f"test_{n}.npy") for n in ("pred", "gt", "mask", "daily"))
    day = 700
    names = np.nonzero(m[:, day] > 0.5)[0]
    top = names[np.argsort(-p[names, day], kind="stable")[:5]]
    ts = pd.Timestamp(dates[day])
    print("portfolio day", ts.date(), "eligible", len(names))
    rets = []
    for i in top:
        sym = tab.symbol[i]
        c = bars[sym]["c"]
        pos = c.index.get_loc(ts)
        r = c.iloc[pos] / c.iloc[pos - 1] - 1
        rets.append(r)
        print(f"  node {i} {order[i]} ({sym}) close {c.index[pos - 1].date()}={c.iloc[pos - 1]:.4f} -> {ts.date()}={c.iloc[pos]:.4f} ret {r:.6f} | stored gt {g[i, day]:.6f} | score {p[i, day]:.6f}")
    print("  recomputed equal-weight top-5 return", round(float(np.mean(rets)), 8), "| stored test_daily", round(float(daily[day]), 8))
    assert abs(float(np.mean(rets)) - float(daily[day])) < 1e-6
    # stock-day: features of one stock on ts against raw closes
    i = int(top[0])
    cs = bars[tab.symbol[i]]["c"]
    cal = pd.DatetimeIndex(data.timestamps)
    t = int(cal.searchsorted(ts))
    seq = cs.reindex(cal).to_numpy()
    ma = {w: float(np.mean(seq[t - 1 - w + 1:t])) for w in (5, 10, 20, 30)}       # last day of the input window = t-1
    f = data.features[i, t - 1]
    scale = np.nanmax(cs.reindex(cal).to_numpy()[: int(cal.searchsorted(pd.Timestamp(SCALE_END))) + 1])
    print("stock-day", order[i], "input day", cal[t - 1].date(), "feature [MA5,MA10,MA20,MA30,close] stored/scale:",
          [round(float(x) * scale, 4) for x in f], "| recomputed:", [round(ma[w], 4) for w in (5, 10, 20, 30)] + [round(float(seq[t - 1]), 4)])
    assert np.allclose([float(x) * scale for x in f], [ma[5], ma[10], ma[20], ma[30], seq[t - 1]], rtol=1e-4)
    print("handcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sanity", "run", "handcheck"])
    {"sanity": sanity, "run": run, "handcheck": handcheck}[ap.parse_args().cmd]()
