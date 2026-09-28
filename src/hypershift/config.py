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
    norm: str = "train"                     # train | paper
    sources: tuple = ("industry", "wiki")   # industry, wiki, corr, sector, subindustry, random
    corr_clusters: int = 100
    structure: str = "hyper"                # hyper | clique | none
    decompose_mode: str = "none"            # none | large_first | small_first
    decompose_size: int = 0
    drop_hub_degree: int = 0
    universe_size: int = 0
    universe_seed: int = 0
    temporal: str = "hyp"                   # hyp | euc
    spatial: str = "hyp"                    # hyp | euc
    attn_score: str = "mobius"              # mobius | concat
    attn_dist: str = "mult"                 # mult | neg | off
    target: str = "return"                  # return | price
    shuffle_train_labels: bool = False
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
