import numpy as np
import pytest
import torch

from hypershift.data.pygt import (DEFAULT_PATH, load_chickenpox, make_windows, neighbourhood_hypergraph,
                                  pairwise_hypergraph)

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="chickenpox.json not downloaded")


def test_loader_shape_and_standardization():
    d = load_chickenpox()
    assert d.x.shape == (20, 521)
    assert d.edges.shape == (102, 2)
    assert abs(d.x.mean()) < 0.05 and abs(d.x.std() - 1) < 0.05   # stored already global z-scored


def test_windows_no_leak():
    d = load_chickenpox()
    inp, tgt = make_windows(d.x, 4)
    assert inp.shape == (517, 20, 4, 1) and tgt.shape == (517, 20)
    np.testing.assert_array_equal(inp[10, :, :, 0], d.x[:, 10:14])
    np.testing.assert_array_equal(tgt[10], d.x[:, 14])


def test_hypergraphs():
    d = load_chickenpox()
    nb = neighbourhood_hypergraph(d)
    assert 0 < len(nb.edges) <= 20 and all(len(e) >= 2 for e in nb.edges)
    pw = pairwise_hypergraph(d)
    assert all(len(e) == 2 for e in pw.edges)


def test_train_smoke_one_epoch():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_pygt", Path(__file__).parents[1] / "scripts" / "run_pygt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = load_chickenpox()
    for protocol in ("pygt", "leakfree"):
        sp = m.split_data(d.x, protocol, 4)
        assert len(sp["train"][0]) > 0 and len(sp["val"][0]) > 0 and len(sp["test"][0]) > 0
    r, hist = m.train_one("THINK", sp, neighbourhood_hypergraph(d), 0, 8, 4, 2, 5e-3, 1, 5, 64, 0.0)
    assert len(hist) == 1 and np.isfinite(r["test_mse"])
    # leak-free: train-period standardization
    tr_end = int(0.7 * 521)
    xs = (d.x - d.x[:, :tr_end].mean()) / d.x[:, :tr_end].std()
    assert abs(xs[:, :tr_end].mean()) < 1e-5 and abs(xs[:, :tr_end].std() - 1) < 1e-5
    b = m.baselines(sp)
    assert b["ar_pooled"] < b["mean"] < b["persistence"]


# ------------------------------------------------------------------ R1 tennis / R3 windmill
from hypershift.data.pygt import (DEFAULT_TENNIS, DEFAULT_WINDMILL, load_tennis, load_windmill,  # noqa: E402
                                  make_windows_tennis, quantile_hypergraph, tennis_neighbourhood,
                                  tennis_union_edges, topk_hypergraph, topk_pairs)


def _runner():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_pygt", Path(__file__).parents[1] / "scripts" / "run_pygt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not DEFAULT_TENNIS.exists(), reason="tennis json not downloaded")
def test_tennis_loader_and_windows():
    d = load_tennis()
    assert d.x.shape == (1000, 120, 16) and d.y.shape == (1000, 120) and len(d.edges) == 120
    assert set(np.unique(d.x)) == {0.0, 1.0} and (d.x.sum(-1) == 2).all()       # 1 degree bin + 1 transitivity bin
    assert d.y.min() >= 0 and d.y.max() < np.log1p(20000)                       # log1p of counts
    inp, tgt, hist = make_windows_tennis(d, 4)
    assert inp.shape == (117, 1000, 4, 16) and tgt.shape == (117, 1000) and hist.shape == (117, 1000, 4)
    np.testing.assert_array_equal(inp[10, :, 2], d.x[:, 12])
    np.testing.assert_array_equal(tgt[10], d.y[:, 13])            # target of the window's last snapshot
    np.testing.assert_array_equal(hist[10, :, -1], d.y[:, 12])    # previous snapshot's target (label lag 1)
    assert np.isnan(hist[0, :, 0]).all() and not np.isnan(hist[1]).any()


@pytest.mark.skipif(not DEFAULT_TENNIS.exists(), reason="tennis json not downloaded")
def test_tennis_hypergraph_uses_train_edges_only():
    d = load_tennis()
    m = _runner()
    for protocol in ("pygt", "leakfree"):
        sp = m.split_tennis(d, protocol, 4)
        end = sp["train_end_time"]
        assert end <= 96 and len(sp["test"][0]) > 0 and len(sp["val"][0]) > 0
        # no edge that only appears in snapshots >= end may be in the union
        u = {tuple(e) for e in tennis_union_edges(d, end)}
        future = {tuple(e) for s in d.edges[end:] for e in s} - {tuple(e) for s in d.edges[:end] for e in s}
        assert future and not (u & future)
        assert 0 < len(tennis_neighbourhood(d, end).edges) <= 1000
    # leakfree train windows end strictly before the val period (target snapshot < 84)
    sp = m.split_tennis(d, "leakfree", 4)
    assert sp["train_end_time"] == int(0.7 * 120)
    b = m.baselines(sp)
    assert b["ar_pooled"] < b["mean"]


@pytest.mark.skipif(not DEFAULT_TENNIS.exists(), reason="tennis json not downloaded")
def test_tennis_smoke_train():
    d = load_tennis()
    m = _runner()
    sp = m.split_tennis(d, "leakfree", 4)
    hg = tennis_neighbourhood(d, sp["train_end_time"])
    r, hist = m.train_one("THINK", sp, hg, 0, 8, 4, 2, 5e-3, 1, 5, 32, 0.0, in_dim=16)
    assert len(hist) == 1 and np.isfinite(r["test_mse"])


@pytest.mark.skipif(not DEFAULT_WINDMILL.exists(), reason="windmill json not downloaded")
def test_windmill_loader_hypergraphs_and_split():
    d = load_windmill()
    assert d.x.shape == (319, 17472) and d.edges.shape == (101761, 2) and d.weights.shape == (101761,)
    tk = topk_hypergraph(d, 5)
    assert 0 < len(tk.edges) <= 319 and all(2 <= len(e) <= 6 for e in tk.edges)
    q = quantile_hypergraph(d, 0.10)
    assert len(q.edges) > 0 and max(len(e) for e in q.edges) > 6
    pw = topk_pairs(d, 5)
    assert all(len(e) == 2 for e in pw.edges)
    m = _runner()
    a = m.split_data(d.x, "pygt", 8, per_node=True)
    b = m.split_data(d.x, "leakfree", 8, per_node=True)
    assert len(a["train"][0]) + len(a["val"][0]) + len(a["test"][0]) == 17472 - 8
    assert len(b["train"][0]) + len(b["val"][0]) + len(b["test"][0]) == 17472 - 8
    # pygt z-score is full-series per node; leakfree is train-period per node (train part ~ zero mean, unit std)
    tr_end = int(0.7 * 17472)
    train_series = b["train"][1]
    assert abs(train_series.mean()) < 1e-2 and abs(train_series.std() - 1) < 0.05
    # first test target of leakfree is at target time va_end and depends on train-period stats only: recompute
    raw = d.x
    mu, sd = raw[:, :tr_end].mean(1), raw[:, :tr_end].std(1) + 1e-10
    va_end = int(0.8 * 17472)
    np.testing.assert_allclose(b["test"][1][0], (raw[:, va_end] - mu) / sd, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not DEFAULT_WINDMILL.exists(), reason="windmill json not downloaded")
def test_windmill_smoke_train():
    d = load_windmill()
    m = _runner()
    sp = m.split_data(d.x, "leakfree", 8, per_node=True)
    small = {k: (sp[k][0][:64], sp[k][1][:64]) for k in ("train", "val", "test")}
    r, hist = m.train_one("THINK", small, topk_hypergraph(d, 5), 0, 8, 8, 2, 5e-3, 1, 5, 32, 0.0, max_train=32,
                          oracle=False)
    assert len(hist) == 1 and np.isfinite(r["test_mse"]) and "test_oracle_mse" not in r
