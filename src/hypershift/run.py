"""python -m hypershift.run --config configs/think_nyse.yaml --seeds 0-4 --set epochs=50 exp=manual"""
from __future__ import annotations

import argparse
import json

from hypershift.config import apply_overrides, config_from_dict, load_yaml
from hypershift.train.loop import train_one_run


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    base = apply_overrides(load_yaml(args.config), args.set)
    for s in parse_seeds(args.seeds):
        m = train_one_run(config_from_dict({**base, "seed": s}))
        print(json.dumps({"seed": s, "best_epoch": m["best_epoch"], "val_sr": m["val"]["sr"],
                          "test_sr": m["test"]["sr"], "test_ndcg5": m["test"]["ndcg5"]}))


if __name__ == "__main__":
    main()
