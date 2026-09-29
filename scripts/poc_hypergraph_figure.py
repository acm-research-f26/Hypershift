"""Industry-level view of the 309-stock PoC hypergraph -> docs/figures/fig4_hypergraph.png. Run from repo root."""
import itertools, json, sys
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "scripts")
from poc_sectors import INDUSTRIES, ROOT
from hypershift.data.hypergraph import build_rsr_hypergraph
from hypershift.data.rsr import read_ticker_file

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63"
tick = read_ticker_file(ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"); pos = {t: i for i, t in enumerate(tick)}
g = json.loads((ROOT / "relation/sector_industry/NYSE_industry_ticker.json").read_text())
ind_of, sector = {}, {}
for sec, names in INDUSTRIES.items():
    for n in names:
        sector[n] = sec
        for t in g[n]:
            if t in pos: ind_of[pos[t]] = n
names = [n for sec in INDUSTRIES for n in INDUSTRIES[sec]]
size = Counter(ind_of.values())
hg = build_rsr_hypergraph(ROOT, "NYSE")
edges = {tuple(sorted(v for v in e if v in ind_of)) for e in hg.edges}
wiki = [e for e in edges if len(e) >= 2 and not (len({ind_of[v] for v in e}) == 1 and len(e) == size[ind_of[e[0]]])]
pair, within = Counter(), Counter()
for e in wiki:
    inds = sorted({ind_of[v] for v in e})
    if len(inds) == 1: within[inds[0]] += 1
    for a, b in itertools.combinations(inds, 2): pair[(a, b)] += 1

ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(names), endpoint=False)
xy = {n: (np.cos(a), np.sin(a)) for n, a in zip(names, ang)}
fig, ax = plt.subplots(figsize=(11, 9.5))
mx = max(pair.values())
for (a, b), c in sorted(pair.items(), key=lambda kv: kv[1]):
    (x1, y1), (x2, y2) = xy[a], xy[b]
    ax.plot([x1, x2], [y1, y2], color=MUTED, lw=0.6 + 5 * c / mx, alpha=0.35 + 0.5 * c / mx, zorder=1)
for n in names:
    x, y = xy[n]; col = BLUE if sector[n].startswith("Energy") else ORANGE
    ax.scatter([x], [y], s=60 + 45 * size[n], color=col, edgecolor="white", lw=2, zorder=3)
    ax.text(x, y, str(size[n]), ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    lx, ly = x + 0.22 * np.sign(x) * (abs(x) > 0.3), 1.24 * y if abs(x) <= 0.3 else y; ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
    extra = f"\n{within[n]} Wikidata edge(s) inside" if within[n] else ""
    ax.text(lx, ly, n + extra, ha=ha, va="center", fontsize=9, color=INK)
ax.scatter([], [], s=120, color=BLUE, label="Energy & Utilities industry (one hyperedge)")
ax.scatter([], [], s=120, color=ORANGE, label="Finance industry (one hyperedge)")
ax.plot([], [], color=MUTED, lw=3, label="Wikidata hyperedges linking stocks across the two industries (thicker = more)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9)
ax.set_title(f"Fig 4. The 309-stock hypergraph: {len(names)} industry hyperedges + {len(wiki)} Wikidata hyperedges\n"
             "(circle = one industry hyperedge, number = its stocks)", loc="left", fontsize=12, color=INK)
ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.5, 1.45); ax.axis("off"); fig.tight_layout()
fig.savefig("docs/figures/fig4_hypergraph.png", dpi=150)
print("top cross-industry links:", pair.most_common(6)); print("within:", dict(within)); print("wiki total", len(wiki))
