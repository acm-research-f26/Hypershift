from collections import Counter
import pytest
from hypershift.experiments.grid import EXPERIMENTS, FAMILIES, INTERACTIONS, experiment

NO_DATA_NEEDED = [e for e in EXPERIMENTS if e != "E7_hubs"]


@pytest.mark.parametrize("name", NO_DATA_NEEDED)
def test_experiment_unique_runs(name):
    cfgs = experiment(name)
    assert cfgs
    keys = Counter((c.exp, c.label, c.seed) for c in cfgs)
    assert max(keys.values()) == 1


def test_sizes():
    assert len(experiment("E2_geometry")) == 3 * 25
    assert len({c.label for c in experiment("E_attn")}) == 6
    assert len(experiment("E_tune")) == 4 * 2 * 3 * 3            # 4 geometries x lr x alpha x 3 seeds
    assert {c.label.split("_")[0] for c in experiment("E_tune")} == {"HH", "HE", "EH", "EE"}
    hday = [c for c in experiment("E10_hourly") if c.label.startswith("hday_")]
    assert all((c.seq, c.kernel) == (112, 28) for c in hday)      # 16 days x 7 bars; 4-day first kernel
    assert all(c.micro_batch_days == 1 for c in experiment("E5_structure") if c.structure == "clique")


def _seeds(key):
    exp, label = key.split("/")
    return {c.seed for c in experiment(exp) if c.label == label}


def test_holm_floor_allows_verdicts():
    """Smallest attainable Holm-adjusted Wilcoxon p = m * 2^(1-n); it must leave room below 0.01."""
    for fam, items in {**FAMILIES, **INTERACTIONS}.items():
        m = len(items)
        for keys in items:
            n = len(set.intersection(*(_seeds(k) for k in keys)))
            assert m * 2.0 ** (1 - n) <= 1e-3, (fam, keys, n)


def test_comparisons_reference_existing_labels():
    labels = {f"{c.exp}/{c.label}" for e in NO_DATA_NEEDED for c in experiment(e)}
    refs = [x for pairs in FAMILIES.values() for p in pairs for x in p]
    refs += [x for quads in INTERACTIONS.values() for q in quads for x in q]
    missing = [r for r in refs if r not in labels]
    assert not missing, missing
