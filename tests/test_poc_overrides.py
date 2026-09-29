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
