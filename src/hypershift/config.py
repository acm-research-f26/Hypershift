from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    exp: str = "debug"
    label: str = "THINK"
    market: str = "NYSE"                    # NYSE | NASDAQ | FRESH
    data_root: str = "data/raw/rsr/data"
    fresh_name: str = ""                    # folder under data/fresh when market == FRESH
    wf_test_year: int = 0                   # 1.5c walk-forward: >0 swaps the NYSE price panel for the Alpaca panel (data_root/alpaca_panel_2016_2023.npz) with splits for this test year; graph stays the frozen RSR v2 (market stays NYSE)
    norm: str = "train"                     # train | paper
    sources: tuple = ("industry", "wiki")   # industry, wiki, corr, sector, subindustry, random
    corr_clusters: int = 100
    model: str = "think"                    # think | rsr_i | sthgcn (baselines ignore temporal/spatial/attn_*)
    structure: str = "hyper"                # hyper | clique | none
    decompose_mode: str = "none"            # none | large_first | small_first
    decompose_size: int = 0
    drop_hub_degree: int = 0
    universe_size: int = 0
    universe_seed: int = 0
    temporal: str = "hyp"                   # hyp | euc
    spatial: str = "hyp"                    # hyp | euc
    attn_score: str = "eq14"                # eq14 (paper) | mobius | concat
    attn_dist: str = "mult"                 # mult | neg | off
    attn_odot: str = "product"              # product (default) | mobius: eq. 7 read as tanh/artanh scalar (see attention.odot_mobius)
    attn_norm: str = "softmax"              # softmax (default) | none (raw eq. 14 value) | sum (s / sum|s|)
    input_mode: str = "level"               # level | relative
    target: str = "return"                # return | price
    shuffle_train_labels: bool = False
    input_std: bool = False                 # F: per-channel (x - mu)/sd with train-window stats after input_mode (DEPARTURE; off = paper-agnostic default)
    input_scale: float = 1.0                # F: multiply inputs after input_mode / input_std (keeps inputs inside the ball)
    init_gain: float = 1.0                  # F: multiplier on the PoincareLinear z init (hyperbolic layers; 1.0 = HNN++ init)
    head_scale: float = 0.0                 # F: >0 adds a learnable output scale exp(t), t0 = log(head_scale), excluded from weight decay (DEPARTURE)
    spatial_residual: bool = False          # F: DHHAN self path, h' = h + relu(agg) in tangent space (DEPARTURE from eq. 15)
    decoupled_wd: bool = False              # F: AdamW (decoupled weight decay) instead of Adam with coupled L2
    save_weights: bool = False              # R2: write best_state.pt + epoch_preds/{val,test}_eNNN.npy (opt-in; default run folders unchanged)
    log_ic: bool = False                    # F: add val/test IC and test pred sd to history.jsonl each epoch (diagnostic only)
    seq: int = 16
    kernel: int = 4
    hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    alpha: float = 1.0
    epochs: int = 100
    patience: int = 20
    batch_days: int = 1                     # days per optimizer step (must match across compared arms)
    micro_batch_days: int = 0               # >0: split each step into gradient-accumulated chunks (memory only)
    topk: int = 5
    periods_per_year: int = 252
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    out_root: str = "results"

    def run_dir(self) -> Path:
        return Path(self.out_root) / self.exp / self.label / f"seed_{self.seed}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = list(self.sources)
        return d


def config_from_dict(d: dict) -> RunConfig:
    names = {f.name for f in fields(RunConfig)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    d = dict(d)
    if "sources" in d:
        d["sources"] = tuple(d["sources"])
    return RunConfig(**d)


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def apply_overrides(base: dict, pairs: list[str]) -> dict:
    out = dict(base)
    defaults = RunConfig()
    for p in pairs:
        k, v = p.split("=", 1)
        cur = getattr(defaults, k)
        if isinstance(cur, bool):
            out[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            out[k] = int(v)
        elif isinstance(cur, float):
            out[k] = float(v)
        elif isinstance(cur, tuple):
            out[k] = tuple(s for s in v.split(",") if s)
        else:
            out[k] = v
    return out


def dump_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
