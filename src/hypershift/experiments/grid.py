"""Every experiment in the study, as lists of RunConfig. Labels are referenced as "<exp>/<label>"."""
from __future__ import annotations

import json
from pathlib import Path

from hypershift.config import RunConfig, load_yaml

SEEDS_FINAL = tuple(range(25))
SEEDS_SWEEP = tuple(range(15))   # verdict-capable: min p = 6e-5, Holm floor for m = 8 is 5e-4
SEEDS_SCREEN = tuple(range(3))   # D7: t_run > 15 min
MEMORY_LIGHT = {"micro_batch_days": 1}   # for clique/decomposition arms; identical math, less GPU memory
FRESH_DAILY_NAME = "sp500_daily"
FRESH_HOURLY_NAME = "sp500_1h"
GEOMS = {
    "HH": {"temporal": "hyp", "spatial": "hyp"},
    "HE": {"temporal": "hyp", "spatial": "euc"},
    "EH": {"temporal": "euc", "spatial": "hyp"},   # = paper's "TCONV + DHHAN"
    "EE": {"temporal": "euc", "spatial": "euc"},   # ~ STHGCN-style Euclidean hypergraph model
}
EXPERIMENTS = ["E0_budget", "E_tune", "E_attn", "E1_main", "E2_geometry", "E3_hhn", "E4_grouping",
               "E5_structure", "E6_decompose", "E7_hubs", "E8_universe", "E9_fresh_daily", "E10_hourly"]


def _read(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text()) if p.suffix == ".json" else load_yaml(p)


def _geo(g: str, **kw) -> dict:
    """Precedence (later wins): geometry < tuned.json[g] < global.yaml < chosen.yaml < experiment kwargs."""
    tuned = _read("results/tuned.json").get(g, {})
    return {**GEOMS[g], **tuned, **_read("configs/global.yaml"), **_read("configs/chosen.yaml"), **kw}


def _mk(exp: str, label: str, seeds, **kw) -> list[RunConfig]:
    return [RunConfig(exp=exp, label=label, seed=s, **kw) for s in seeds]


def _hub_thresholds() -> list[int]:
    from hypershift.data.hypergraph import build_rsr_hypergraph, hub_schedule
    return hub_schedule(build_rsr_hypergraph(RunConfig().data_root, "NYSE"))


def experiment(name: str) -> list[RunConfig]:
    E: list[RunConfig] = []
    if name == "E0_budget":
        E += _mk(name, "HH", (0,), **_geo("HH", epochs=3, patience=100))
        E += _mk(name, "HH_clique", (0,), **_geo("HH", epochs=1, patience=100, structure="clique", **MEMORY_LIGHT))
    elif name == "E_tune":
        for g in GEOMS:                               # every geometry gets the same budget (fairness rule)
            for lr in (5e-4, 1e-3):
                for alpha in (0.1, 1.0, 10.0):
                    kw = {**GEOMS[g], **_read("configs/global.yaml"), "lr": lr, "alpha": alpha}
                    E += _mk(name, f"{g}_lr{lr}_a{alpha}", (0, 1, 2), **kw)
    elif name == "E_attn":
        for sc in ("mobius", "concat"):
            for di in ("mult", "neg", "off"):
                E += _mk(name, f"{sc}_{di}", SEEDS_SCREEN, **_geo("HH", attn_score=sc, attn_dist=di))
    elif name == "E1_main":
        E += _mk(name, "THINK", SEEDS_FINAL, **_geo("HH"))
        E += _mk(name, "THINK_paperNorm", SEEDS_FINAL, **_geo("HH", norm="paper"))
        # Paper/STHAN-SR protocol: full-series-max norm, all 100 epochs, test read every epoch -> test_oracle_sr.
        E += _mk(name, "THINK_paperProtocol", SEEDS_FINAL, **_geo("HH", norm="paper", epochs=100, patience=1000))
        E += _mk(name, "THINK_NASDAQ", SEEDS_FINAL, **_geo("HH", market="NASDAQ", alpha=0.1))
        E += _mk(name, "THINK_NASDAQ_paperNorm", SEEDS_FINAL, **_geo("HH", market="NASDAQ", alpha=0.1, norm="paper"))
        E += _mk(name, "THINK_shuffled", (0, 1, 2), **_geo("HH", shuffle_train_labels=True))
    elif name == "E2_geometry":
        for g in ("HE", "EH", "EE"):
            E += _mk(name, g, SEEDS_FINAL, **_geo(g))
    elif name == "E3_hhn":
        E += _mk(name, "HHN", SEEDS_FINAL, **_geo("HH", attn_dist="off"))
    elif name == "E4_grouping":
        # "both" (industry+wiki) = E1_main/THINK and E2_geometry/EE; "none" = E5_structure/*_none. Not re-run here.
        groups = {"industry": {"sources": ("industry",)}, "wiki": {"sources": ("wiki",)},
                  "random": {"sources": ("industry", "wiki", "random")}, "corr": {"sources": ("corr",)}}
        for g in ("HH", "EE"):
            for gname, kw in groups.items():
                E += _mk(name, f"{g}_{gname}", SEEDS_SWEEP, **_geo(g, **kw))
    elif name == "E5_structure":
        for g in ("HH", "EE"):
            E += _mk(name, f"{g}_clique", SEEDS_FINAL, **_geo(g, structure="clique", **MEMORY_LIGHT))
            E += _mk(name, f"{g}_none", SEEDS_FINAL, **_geo(g, structure="none"))
    elif name == "E6_decompose":
        for g in ("HH", "EE"):
            for s in (15, 9, 5, 3):
                E += _mk(name, f"{g}_large_{s}", SEEDS_SCREEN,
                         **_geo(g, decompose_mode="large_first", decompose_size=s, **MEMORY_LIGHT))
                E += _mk(name, f"{g}_small_{s}", SEEDS_SCREEN,
                         **_geo(g, decompose_mode="small_first", decompose_size=s, **MEMORY_LIGHT))
    elif name == "E7_hubs":
        for g in ("HH", "EE"):
            for th in _hub_thresholds():
                E += _mk(name, f"{g}_hub{th}", SEEDS_SCREEN, **_geo(g, drop_hub_degree=th))
    elif name == "E8_universe":
        for g in ("HH", "EE"):
            for n in (50, 100, 250, 500, 1000):
                for u in (0, 1, 2):
                    E += _mk(name, f"{g}_N{n}_u{u}", SEEDS_SCREEN, **_geo(g, universe_size=n, universe_seed=u))
    elif name == "E9_fresh_daily":
        for g in ("HH", "EE"):
            for st in ("hyper", "clique", "none"):
                light = MEMORY_LIGHT if st == "clique" else {}
                E += _mk(name, f"{g}_{st}", SEEDS_SWEEP, **_geo(g, market="FRESH", fresh_name=FRESH_DAILY_NAME,
                                                               sources=("subindustry",), structure=st, alpha=1.0,
                                                               **light))
    elif name == "E10_hourly":
        # hday: 112 hourly bars = 16 trading days, first kernel 28 bars = 4 days -> 4 steps, exactly like the
        # daily arm (16 days, kernel 4). Same look-back, target, trades and architecture; only granularity differs.
        arms = {"daily": (f"{FRESH_HOURLY_NAME}_daily", 252, {}),
                "hday": (f"{FRESH_HOURLY_NAME}_hday", 252, {"seq": 112, "kernel": 28}),
                "hourly": (f"{FRESH_HOURLY_NAME}_hourly", 1764, {})}
        for arm, (fresh, ppy, extra) in arms.items():
            for g in ("HH", "EE"):
                E += _mk(name, f"{arm}_{g}", SEEDS_SWEEP, **_geo(g, market="FRESH", fresh_name=fresh,
                                                                sources=("subindustry",), periods_per_year=ppy,
                                                                alpha=1.0, **extra))
    else:
        raise KeyError(name)
    return E


FAMILIES = {
    "Q0_normalization_leak": [("E1_main/THINK_paperNorm", "E1_main/THINK")],
    "Q2_hyperbolic": [("E1_main/THINK", "E2_geometry/EE"), ("E1_main/THINK", "E2_geometry/EH"),
                      ("E1_main/THINK", "E2_geometry/HE"), ("E2_geometry/EH", "E2_geometry/EE"),
                      ("E2_geometry/HE", "E2_geometry/EE")],
    "Q_attention": [("E1_main/THINK", "E3_hhn/HHN")],
    "Q3_hyperedges": [("E1_main/THINK", "E5_structure/HH_clique"), ("E1_main/THINK", "E5_structure/HH_none"),
                      ("E2_geometry/EE", "E5_structure/EE_clique"), ("E2_geometry/EE", "E5_structure/EE_none")],
    "Q1_grouping": [(both, f"E4_grouping/{g}_{x}")
                    for g, both in (("HH", "E1_main/THINK"), ("EE", "E2_geometry/EE"))
                    for x in ("industry", "wiki", "random", "corr")],
    "Q6_daily_fresh": [("E9_fresh_daily/HH_hyper", "E9_fresh_daily/EE_hyper"),
                       ("E9_fresh_daily/HH_hyper", "E9_fresh_daily/HH_clique"),
                       ("E9_fresh_daily/HH_hyper", "E9_fresh_daily/HH_none")],
    "Q7_hourly": [("E10_hourly/hday_HH", "E10_hourly/daily_HH"), ("E10_hourly/hday_EE", "E10_hourly/daily_EE"),
                  ("E10_hourly/hday_HH", "E10_hourly/hday_EE"), ("E10_hourly/hourly_HH", "E10_hourly/hourly_EE")],
}
INTERACTIONS = {
    "Q4_interaction": [
        ("E1_main/THINK", "E5_structure/HH_clique", "E2_geometry/EE", "E5_structure/EE_clique"),
        ("E1_main/THINK", "E5_structure/HH_none", "E2_geometry/EE", "E5_structure/EE_none"),
    ],
}
