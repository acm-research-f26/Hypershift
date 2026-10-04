import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypershift.data import alpaca_wf as W
from hypershift.data.rsr import FILL, MarketData
from hypershift.train.loop import window_offsets


def _synth(tmp_path, n=6):
    cal = pd.bdate_range("2016-01-04", "2023-12-29")
    T = len(cal)
    rng = np.random.default_rng(0)
    feats = rng.uniform(0.5, 1, (n, T, 5)).astype(np.float32)
    mask = np.ones((n, T), np.float32)
    ident = np.array([1, 1, 0, 1, 1, 1], np.uint8)[:n]
    mask[ident == 0] = 0
    feats[ident == 0] = FILL
    mask[3, 1500:] = 0                                   # delisted
    md = MarketData([f"T{i}" for i in range(n)], feats, mask, rng.normal(0, .01, (n, T)).astype(np.float32),
                    feats[:, :, 4].copy(), 0, 0, np.asarray(cal.values))
    p = tmp_path / W.PANEL_NAME
    W.save_panel_npz(md, ident, p)
    return p, md, cal


@pytest.mark.parametrize("year", W.WF_YEARS)
def test_split_boundaries_exact(tmp_path, year):
    p, md, cal = _synth(tmp_path)
    d = W.load_wf_panel(p, year)
    dates = pd.DatetimeIndex(d.timestamps)
    assert dates[-1] <= pd.Timestamp(f"{year}-12-31") and dates[-1] >= pd.Timestamp(f"{year}-12-25")
    assert dates[d.test_index] >= pd.Timestamp(f"{year}-01-01") > dates[d.test_index - 1]
    assert dates[d.valid_index] >= pd.Timestamp(f"{year-1}-01-01") > dates[d.valid_index - 1]
    tr = dates[window_offsets(d, 16, "train") + 16]
    va = dates[window_offsets(d, 16, "val") + 16]
    te = dates[window_offsets(d, 16, "test") + 16]
    assert tr.max() <= pd.Timestamp(f"{year-2}-12-31")
    assert va.min() >= pd.Timestamp(f"{year-1}-01-01") and va.max() <= pd.Timestamp(f"{year-1}-12-31")
    assert te.min() >= pd.Timestamp(f"{year}-01-01") and te.max() <= pd.Timestamp(f"{year}-12-31")
    assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0 and len(set(tr) & set(va)) == 0


def test_2019_first_train_target_in_2016(tmp_path):
    p, *_ = _synth(tmp_path)
    d = W.load_wf_panel(p, 2019)
    first = pd.DatetimeIndex(d.timestamps)[window_offsets(d, 16, "train")[0] + 16]
    assert first.year == 2016


def test_masking_and_identity(tmp_path):
    p, md, _ = _synth(tmp_path)
    d = W.load_wf_panel(p, 2021)
    assert d.mask[2].sum() == 0 and d.num_nodes == 6
    assert d.mask[3, 1500:].sum() == 0
    assert d.features.shape[1] == d.mask.shape[1] == d.gt.shape[1] == len(d.timestamps)
    np.testing.assert_array_equal(d.features, md.features[:, :d.num_steps])   # truncation does not alter earlier values


def test_uncovered_year_raises(tmp_path):
    p, *_ = _synth(tmp_path)
    with pytest.raises(ValueError):
        W.load_wf_panel(p, 2024)


@pytest.mark.skipif(not Path("data/raw/alpaca_post2017/bars.pkl.gz").exists(), reason="no Alpaca bars")
def test_real_panel_matches_15b_and_rsr_order(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    import eval_post2017 as E
    from hypershift.data.rsr import read_ticker_file
    data, order, *_ = E.load_panel()
    ident = pd.read_csv(E.IDENT).identity_ok.to_numpy()
    p = tmp_path / W.PANEL_NAME
    W.save_panel_npz(data, ident, p)
    d = W.load_wf_panel(p, 2023)
    assert d.tickers == read_ticker_file(E.RSR / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv") == list(order)
    assert d.num_nodes == 1737 and int(ident.sum()) == 1647
    n = d.num_steps
    np.testing.assert_array_equal(d.features, data.features[:, :n])
    np.testing.assert_array_equal(d.mask, data.mask[:, :n])
    np.testing.assert_array_equal(d.gt, data.gt[:, :n])
    assert d.mask[~ident.astype(bool)].sum() == 0


def test_load_market_dispatch_and_graph_alignment(tmp_path):
    from hypershift.config import RunConfig
    from hypershift.train.loop import load_market
    p, md, _ = _synth(tmp_path)
    cfg = RunConfig(data_root=str(tmp_path), wf_test_year=2020)
    d = load_market(cfg)
    assert d.num_nodes == 6 and d.test_index > d.valid_index
    assert pd.DatetimeIndex(d.timestamps)[-1].year == 2020
    assert RunConfig().wf_test_year == 0
