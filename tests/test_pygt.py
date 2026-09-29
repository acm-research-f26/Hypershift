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
