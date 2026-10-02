"""Phase 1.5 F diagnostics (CPU): activation norms per layer, loss-gradient vs weight-decay-gradient at init/after k steps."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np, torch
sys.path.insert(0, "scripts")
import known_signal as ks
from hypershift.train.loop import build_model, gather_batch, apply_input_mode, window_offsets, prepare
from hypershift.train.loss import rank_mse_loss

def main(level="high", mode="level", arm="HH_hyper", steps=0, wd=5e-4, lr=1e-3, every=20):
    torch.set_num_threads(2)
    real, hg = ks._universe()
    phi, gamma, sigma = ks.LEVELS[level]
    syn, A = ks.plant_signal(real, hg, phi, gamma, sigma, 0)
    cfg = ks.make_cfg("f_diag", arm, 0, arm, mode, 1, "cpu", "results", weight_decay=wd, lr=lr)
    data, g = prepare(cfg, syn, hg)
    thg = g.to_torch(torch.device("cpu"))
    torch.manual_seed(0)
    model = build_model(cfg, 5, data)
    acts = {}
    for n, m in model.named_modules():
        if n in ("tconv1", "tconv2", "spatial"):
            m.register_forward_hook(lambda mod, i, o, n=n: acts.__setitem__(n, (i[0].norm(dim=-1).mean().item(), o.norm(dim=-1).mean().item(), o.norm(dim=-1).max().item())))
    offs = window_offsets(data, cfg.seq, "train")
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    def batch(k=8):
        o = rng.choice(offs, k, replace=False)
        x, m, b, gg = gather_batch(data, o, cfg.seq)
        return [torch.as_tensor(a) for a in (apply_input_mode(x, mode), m, b, gg)]
    def ic(p, gg, m):
        p, gg = p.detach().numpy(), gg.numpy()
        return float(np.mean([np.corrcoef(p[i], gg[i])[0, 1] if p[i].std() > 0 else 0 for i in range(len(p))]))
    for s in range(steps):
        x, m, b, gg = batch(); opt.zero_grad()
        p = model(x, thg); l, _, _ = rank_mse_loss(p, gg, m, 1.0); l.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        opt.step()
        if s % every == 0 or s == steps - 1:
            zn = {n.replace('.fc.z', ''): round(q.norm().item(), 3) for n, q in model.named_parameters() if n.endswith('.z')}
            print(f"  step {s:4d} loss {l.item():.3e} pred_sd {p.std().item():.2e} IC(batch) {ic(p, gg, m):+.3f} preclip_gradnorm {gnorm:.2e} |z| {zn}")
    x, m, b, gg = batch(); model.zero_grad()
    p = model(x, thg); l, reg, rk = rank_mse_loss(p, gg, m, 1.0); l.backward()
    print(f"[{arm} {mode} {level} steps={steps}] pred sd {p.std().item():.2e} loss {l.item():.3e}")
    print("  acts (in_norm, out_mean, out_max):", {k: tuple(round(v, 4) for v in t) for k, t in acts.items()})
    for n, prm in model.named_parameters():
        gn = prm.grad.norm().item() if prm.grad is not None else 0
        print(f"  {n:22s} |w|={prm.norm().item():.3e} |grad loss|={gn:.2e}  |wd*w|(5e-4)={5e-4*prm.norm().item():.2e}  ratio={gn/(5e-4*prm.norm().item()+1e-30):.2e}")

if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:3]), steps=int(a[3]) if len(a) > 3 else 0, wd=float(a[4]) if len(a) > 4 else 5e-4, lr=float(a[5]) if len(a) > 5 else 1e-3)
