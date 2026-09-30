import json, glob, numpy as np
for f in sorted(glob.glob("results/E11_clf_g2_diag/*.json")):
    d = json.load(open(f)); h = d["history"]; b = d["best_epoch"]
    print(f, "epochs run", len(h), "best(val macro)", b)
    print("  loss first/last", round(h[0]["loss"], 4), round(h[-1]["loss"], 4), "ln3=1.0986")
    for e in sorted({0, b, len(h) - 1}):
        v, t = h[e]["val"], h[e]["test"]
        print(f"  ep{e}: val macro {v['macro']:.3f} micro {v['micro']:.3f} predshare {v['pred_share']} | test macro {t['macro']:.3f} micro {t['micro']:.3f} predshare {t['pred_share']} day-agree {t['mean_day_agree']:.2f}")
    vm = np.array([x["val"]["macro"] for x in h]); tm = np.array([x["test"]["macro"] for x in h])
    print(f"  val macro range {vm.min():.3f}-{vm.max():.3f}; test macro range {tm.min():.3f}-{tm.max():.3f}; epochs with test macro>=0.333: {(tm>=1/3).sum()}; val>=0.333: {(vm>=1/3).sum()}")
    print("  test pred share at best:", h[b]["test"]["pred_share"], "cm", h[b]["test"]["cm"])
    print("  val pred share at best:", h[b]["val"]["pred_share"], "true val share", h[b]["val"]["true_share"])
