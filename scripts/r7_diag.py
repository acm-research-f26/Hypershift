"""R7 diagnosis: re-run clf training (same loop as train_clf_run) on CPU, logging per epoch the predicted class
distribution, per-day prediction agreement and F1 on val/test. Writes results/E11_clf_g2_diag/<arm>_s<k>.json (new folder)."""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
from hypershift.config import RunConfig
from hypershift.experiments.grid import _geo
from hypershift.models.think import THINK
from hypershift.train.loop import apply_input_mode, gather_batch, prepare, set_seed, window_offsets
from hypershift.train.clf import tertile_thresholds

def evalsplit(model, data, thg, cfg, split, th):
    model.eval(); ys, ps, agree = [], [], []
    offs = window_offsets(data, cfg.seq, split)
    with torch.no_grad():
        for i in range(0, len(offs), cfg.batch_days):
            x, m, _, g = gather_batch(data, offs[i:i + cfg.batch_days], cfg.seq)
            pred = model(torch.as_tensor(apply_input_mode(x, cfg.input_mode)), thg).argmax(-1).numpy()
            y = np.digitize(g, th); keep = m > 0.5
            for d in range(pred.shape[0]):
                k = keep[d]
                agree.append(np.bincount(pred[d][k], minlength=3).max() / k.sum())
            ys.append(y[keep]); ps.append(pred[keep])
    y, p = np.concatenate(ys), np.concatenate(ps)
    return dict(macro=f1_score(y, p, average="macro"), micro=f1_score(y, p, average="micro"),
                pred_share=(np.bincount(p, minlength=3) / len(p)).round(4).tolist(),
                true_share=(np.bincount(y, minlength=3) / len(y)).round(4).tolist(),
                mean_day_agree=float(np.mean(agree)), cm=confusion_matrix(y, p, labels=[0, 1, 2]).tolist())

arm, seed = sys.argv[1], int(sys.argv[2]); nep = int(sys.argv[3]) if len(sys.argv) > 3 else 60
cfg = RunConfig(exp="E11_clf_g2_diag", label=arm, seed=seed, **_geo(arm, market="NASDAQ", alpha=0.1)); cfg.device = "cpu"; cfg.epochs = nep
set_seed(seed); data, hg = prepare(cfg, None, None); thg = hg.to_torch(torch.device("cpu"))
model = THINK(in_dim=data.features.shape[2], hidden=cfg.hidden, seq=cfg.seq, kernel=cfg.kernel, temporal=cfg.temporal, spatial=cfg.spatial,
              structure=cfg.structure, attn_score=cfg.attn_score, attn_dist=cfg.attn_dist, out_dim=3)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
th = tertile_thresholds(data, cfg.seq); offs = window_offsets(data, cfg.seq, "train"); rng = np.random.default_rng(seed)
hist, best, bad = [], None, 0; t0 = time.time()
for ep in range(cfg.epochs):
    model.train(); rng.shuffle(offs); tl = []
    for i in range(0, len(offs), cfg.batch_days):
        x, m, _, g = gather_batch(data, offs[i:i + cfg.batch_days], cfg.seq)
        logits = model(torch.as_tensor(apply_input_mode(x, cfg.input_mode)), thg)
        y = torch.as_tensor(np.digitize(g, th)); mt = torch.as_tensor(m).reshape(-1)
        loss = (F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none") * mt).sum() / mt.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); tl.append(float(loss))
    v, t = evalsplit(model, data, thg, cfg, "val", th), evalsplit(model, data, thg, cfg, "test", th)
    hist.append(dict(epoch=ep, loss=float(np.mean(tl)), val=v, test=t))
    print(f"{arm} s{seed} ep{ep} loss {np.mean(tl):.4f} val macro {v['macro']:.3f} pred {v['pred_share']} | test macro {t['macro']:.3f} micro {t['micro']:.3f} pred {t['pred_share']} agree {t['mean_day_agree']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    if best is None or v["macro"] > hist[best]["val"]["macro"]: best, bad = ep, 0
    else:
        bad += 1
        if bad >= cfg.patience: break
out = Path("results/E11_clf_g2_diag"); out.mkdir(parents=True, exist_ok=True)
json.dump(dict(arm=arm, seed=seed, best_epoch=best, history=hist), open(out / f"{arm}_s{seed}.json", "w"))
print("BEST", best, hist[best]["test"])
