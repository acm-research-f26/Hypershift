from pathlib import Path
import numpy as np
import pytest
from hypershift.data.rsr import FILL, load_rsr, parse_eod, renormalize_train

REAL = Path("data/raw/rsr/data")


def test_parse_eod_missing_and_gt():
    raw = np.array([
        [0, .5, .5, .5, .5, .50],
        [1, .5, .5, .5, .5, .55],
        [2, -1234, -1234, -1234, -1234, -1234],
        [3, .5, .5, .5, .5, .60],
    ])
    f, m, g, b = parse_eod(raw, drop_last=False)
    assert m.tolist() == [1, 1, 0, 1]
    assert g[1] == pytest.approx(0.1)
    assert g[2] == 0 and g[3] == 0          # current missing / previous missing -> 0
    assert f[2].tolist() == pytest.approx([FILL] * 5)
    assert b[2] == pytest.approx(FILL)
    assert f.shape == (4, 5)


def test_parse_eod_drop_last():
    raw = np.tile(np.array([[0, .5, .5, .5, .5, .5]]), (3, 1))
    f, m, g, b = parse_eod(raw, drop_last=True)
    assert f.shape == (2, 5)


def _write_market(root: Path, market: str, tickers, T):
    (root / "2013-01-01").mkdir(parents=True)
    (root / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv").write_text("\n".join(tickers) + "\n")
    rng = np.random.default_rng(0)
    for t in tickers:
        close = np.linspace(0.2, 0.9, T) + rng.normal(0, 0.01, T)
        rows = np.column_stack([np.arange(T), close, close, close, close, close])
        np.savetxt(root / "2013-01-01" / f"{market}_{t}_1.csv", rows, delimiter=",", fmt="%.6f")


def test_load_rsr_synthetic_and_train_norm(tmp_path):
    _write_market(tmp_path, "NYSE", ["AAA", "BBB"], T=1100)
    d = load_rsr(tmp_path, "NYSE", norm="paper")
    assert d.features.shape == (2, 1100, 5)
    assert (d.valid_index, d.test_index) == (756, 1008)
    dt = renormalize_train(d)
    train_close_max = dt.features[:, :756, -1].max(axis=1)
    np.testing.assert_allclose(train_close_max, 1.0, rtol=1e-5)
    np.testing.assert_allclose(dt.gt, d.gt)  # returns are scale-invariant


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_shapes():
    ny = load_rsr(REAL, "NYSE", norm="paper")
    assert ny.features.shape == (1737, 1245, 5)
    na = load_rsr(REAL, "NASDAQ", norm="paper")
    assert na.features.shape == (1026, 1245, 5)
