"""A10 / G5 on full NYSE (R5_g2: HH_none, EE_none, THINK_nodist vs THINK_paperProtocol / EE). CPU only:
    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/a10_g5_analysis.py
Reuses scripts/r5_g2_analysis.py helpers (diag, vec, series)."""
import math, sys
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import r5_g2_analysis as r
from hypershift.eval.metrics import sharpe
from hypershift.eval.stats import holm, sharpe_diff_ci, verdict, wilcoxon_paired

NEW = ["HH_none", "EE_none", "THINK_nodist"]
for a in NEW:
    r.R[a] = r.runs(a)
ARMS = ["THINK_paperProtocol", "EE"] + NEW
D = {a: r.diag(a) for a in ARMS}
market = sharpe((r.G * r.K).sum(0) / np.maximum(r.K.sum(0), 1.0))
print(f"hold-all {market:.3f}; const {r.CONST:.3f}; random NDCG@5 {r.random_ndcg():.4f}")
print("\n| arm | n | leak-free | best-test | val Sharpe at sel | NDCG@5 | NDCG_sthan | IC | hit pp | spread | spread<1e-5 | median sel ep | median best ep | best-test==const |\n|" + "---|" * 14)
for a in ARMS:
    d = D[a]
    print(f"| {a} | {len(r.R[a])} | {r.ms(d['sr_v'])} | {r.ms(d['sr_o'])} | {r.ms(d['val_sr_at_sel'])} | {r.ms(d['nd_v'],4)} | {r.ms(d['ndsthan_v'],3)} | {r.ms(d['ic'],4)} | {r.ms(d['hit'],2)} | {r.ms(d['spread'],3)} | {int(d['spread_const'].sum())}/{len(d['spread_const'])} | {int(np.median(d['best_ep']))} | {int(np.median(d['orc_ep']))} | {int(d['orc_const'].sum())}/{len(d['orc_const'])} |")
s10 = list(range(10))
for a in ["THINK_paperProtocol"]:
    print(f"THINK seeds 0-9 only: leak-free {r.vec(a,s10,'lf').mean():.3f} best-test {r.vec(a,s10,'bt').mean():.3f}")
pairs = [("THINK_paperProtocol", "HH_none", "G5: THINK(HH) vs HH_none"), ("EE", "EE_none", "G5: EE vs EE_none"), ("THINK_paperProtocol", "THINK_nodist", "A10: THINK vs THINK_nodist")]
rows, ps = [], {}
for a, b, q in pairs:
    c = sorted(set(r.R[a]) & set(r.R[b]))
    sa, sb = r.vec(a, c, "lf"), r.vec(b, c, "lf"); ba, bb = r.vec(a, c, "bt"), r.vec(b, c, "bt")
    ci = sharpe_diff_ci(r.series(a, c), r.series(b, c)); p = wilcoxon_paired(sa, sb); ps[q] = p
    allA = r.vec(a, sorted(r.R[a]), "lf"); allAb = r.vec(a, sorted(r.R[a]), "bt")
    rows.append((q, len(c), sa.mean(), sb.mean(), sa.mean() - sb.mean(), p, ci, int((sa > sb).sum()), mannwhitneyu(sa, sb).pvalue,
                 ba.mean(), bb.mean(), ba.mean() - bb.mean(), wilcoxon_paired(ba, bb), int((ba > bb).sum()), mannwhitneyu(ba, bb).pvalue,
                 mannwhitneyu(allA, sb).pvalue, allA.mean() - sb.mean(), mannwhitneyu(allAb, bb).pvalue, allAb.mean() - bb.mean()))
adj = holm(ps)
print("\nLeak-free (paired on seeds 0-9; Holm over 3)\n| comparison | n | with | without | diff | Wilcoxon p | Holm p | CI | with wins | MW p | verdict | MW all-25 vs 10: diff, p |\n|" + "---|" * 12)
for x in rows:
    q = x[0]; ci = x[6]
    print(f"| {q} | {x[1]} | {x[2]:.3f} | {x[3]:.3f} | {x[4]:+.3f} | {x[5]:.4f} | {adj[q]:.4f} | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {x[7]}/{x[1]} | {x[8]:.3f} | {verdict(adj[q], ci['lo'], ci['hi'])} | {x[16]:+.3f}, {x[15]:.3f} |")
print("\nBest-test (raw p, diagnostic)\n| comparison | n | with | without | diff | Wilcoxon p | with wins | MW p | MW all-25 vs 10: diff, p |\n|" + "---|" * 9)
for x in rows:
    print(f"| {x[0]} | {x[1]} | {x[9]:.3f} | {x[10]:.3f} | {x[11]:+.3f} | {x[12]:.4f} | {x[13]}/{x[1]} | {x[14]:.3f} | {x[18]:+.3f}, {x[17]:.3f} |")
for a in ARMS:
    lf = D[a]["sr_v"]
    print(f"{a}: leak-free > hold-all {int((lf>market).sum())}/{len(lf)}; best-test > hold-all {int((D[a]['sr_o']>market).sum())}/{len(lf)}; ensemble {sharpe(r.series(a, sorted(r.R[a]))):.3f}")
