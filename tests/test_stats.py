import numpy as np
import pytest
from hypershift.eval.stats import (
    holm, sharpe_contrast_ci, sharpe_diff_ci, stationary_bootstrap_indices, verdict,
    wilcoxon_one_sample, wilcoxon_paired,
)


def test_holm_known_example():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_wilcoxon():
    rng = np.random.default_rng(0)
    b = rng.normal(size=25)
    assert wilcoxon_paired(b, b) == 1.0
    assert wilcoxon_paired(b + 1 + 0.01 * rng.normal(size=25), b) < 0.01
    assert wilcoxon_one_sample(np.full(25, 0.5) + 0.01 * rng.normal(size=25)) < 0.01


def test_bootstrap_indices_valid():
    idx = stationary_bootstrap_indices(100, 10, np.random.default_rng(0))
    assert idx.shape == (100,) and idx.min() >= 0 and idx.max() < 100


def test_sharpe_diff_ci():
    rng = np.random.default_rng(0)
    rb = rng.normal(0, 0.01, 250)
    same = sharpe_diff_ci(rb, rb, n_boot=300)
    assert same["est"] == 0 and same["lo"] == 0 and same["hi"] == 0
    better = sharpe_diff_ci(rb + 0.005, rb, n_boot=300)
    assert better["est"] > 0 and better["lo"] > 0


def test_contrast_matches_diff():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.001, 0.01, 200), rng.normal(0, 0.01, 200)
    d = sharpe_diff_ci(a, b, n_boot=200, seed=3)
    c = sharpe_contrast_ci([a, b], [1, -1], n_boot=200, seed=3)
    assert d == c


def test_verdict():
    assert verdict(0.001, 0.1, 0.5) == "STRONG"
    assert verdict(0.001, -0.1, 0.5) == "SEED-ROBUST ONLY"
    assert verdict(0.2, 0.1, 0.5) == "NO EVIDENCE"
