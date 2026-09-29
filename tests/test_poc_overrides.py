import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import poc_sectors  # noqa: E402


def test_set_overrides_reach_runconfig():
    c = poc_sectors.cfg("HH_hyper", "HH", "hyper", 3, 30, "X", overrides=[
        "shuffle_train_labels=true", "attn_dist=off", "decompose_size=10", "decompose_mode=large_first",
        "drop_hub_degree=8", "micro_batch_days=1"])
    assert c.shuffle_train_labels is True and c.attn_dist == "off"
    assert c.decompose_size == 10 and c.decompose_mode == "large_first"
    assert c.drop_hub_degree == 8 and c.micro_batch_days == 1
    assert c.temporal == "hyp" and c.batch_days == 8 and c.seed == 3       # rest of the arm untouched
    assert json.loads(json.dumps(c.to_dict()))["attn_dist"] == "off"        # what config.json will record
    d = poc_sectors.cfg("HH_hyper", "HH", "hyper", 3, 30, "X")
    assert d.attn_dist == "mult" and d.shuffle_train_labels is False


def test_widened_grid_selects_on_validation_only(tmp_path, monkeypatch):
    """--grid-alpha widens the tuning grid; selection uses mean VALIDATION Sharpe over the full-seed combos only."""
    import numpy as np
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(poc_sectors, "GRID_LR", poc_sectors.GRID_LR)
    monkeypatch.setattr(poc_sectors, "GRID_ALPHA", poc_sectors.GRID_ALPHA)
    poc_sectors.set_grid(lr=[5e-4, 1e-3, 3e-3], alpha=[1, 10, 30, 100])
    assert poc_sectors.GRID_ALPHA == (1.0, 10.0, 30.0, 100.0)
    for lr in poc_sectors.GRID_LR:
        for alpha in poc_sectors.GRID_ALPHA:
            for s in range(3):
                d = tmp_path / "results" / "T" / poc_sectors.tune_label("HH", lr, alpha) / f"seed_{s}"
                d.mkdir(parents=True)
                val = 1.0 if (lr, alpha) == (1e-3, 30.0) else 0.0
                test = 9.0 if (lr, alpha) == (3e-3, 100.0) else 0.0      # a test-only winner must NOT be picked
                (d / "metrics.json").write_text(json.dumps({"val": {"sr": val}, "test": {"sr": test}}))
                np.save(d / "test_daily.npy", np.zeros(3))
    poc_sectors.tune_select("T", geoms=["HH"])
    assert json.loads((tmp_path / "results" / "T" / "tuned.json").read_text()) == {"HH": {"lr": 1e-3, "alpha": 30.0}}
