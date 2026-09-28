import numpy as np
import pandas as pd
import pytest
from hypershift.data.fresh import (
    NY, build_panel, daily_horizon, gics_hypergraph, load_panel, save_panel, to_daily, to_rth_hourly,
)


def make_30min_bars(days, tickers=("AAA", "BBB"), drop_ticker=None):
    rows = []
    for di, d in enumerate(days):
        for k, t in enumerate(pd.date_range(f"{d} 09:00", f"{d} 16:30", freq="30min", tz=NY, inclusive="left")):
            for j, tk in enumerate(tickers):
                if tk == drop_ticker and di % 2 == 0:
                    continue
                px = 100 + 10 * j + di + 0.1 * k
                rows.append({"timestamp": t, "ticker": tk, "open": px, "high": px + .05, "low": px - .05,
                             "close": px, "volume": 100})
    return pd.DataFrame(rows)


def test_to_rth_hourly_anchor_and_premarket():
    h = to_rth_hourly(make_30min_bars(["2024-03-04", "2024-03-05"]))
    a = h[h.ticker == "AAA"].sort_values("timestamp")
    assert len(a) == 14
    hhmm = a.timestamp.dt.tz_convert(NY).dt.strftime("%H:%M").unique().tolist()
    assert hhmm == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert a.close.iloc[0] == pytest.approx(100.2)      # 09:30 bin closes with the 10:00 half-hour bar
    assert a.close.iloc[6] == pytest.approx(101.3)      # 15:30 bin = last RTH bar; 16:00 bar excluded


def test_to_daily():
    d = to_daily(to_rth_hourly(make_30min_bars(["2024-03-04", "2024-03-05"])))
    a = d[d.ticker == "AAA"].sort_values("timestamp")
    assert len(a) == 2 and a.close.iloc[0] == pytest.approx(101.3)


def _hourly_panel():
    days = [x.strftime("%Y-%m-%d") for x in pd.bdate_range("2024-01-02", periods=30)]
    bars = to_rth_hourly(make_30min_bars(days, tickers=("AAA", "BBB", "CCC"), drop_ticker="CCC"))
    return build_panel(bars)


def test_build_panel_filters_and_normalizes():
    p = _hourly_panel()
    assert p.tickers == ["AAA", "BBB"]                   # CCC missing ~50% -> dropped (D13)
    assert p.features.shape == (2, 210, 5)
    assert (p.valid_index, p.test_index) == (126, 168)
    np.testing.assert_allclose(p.features[:, :126, -1].max(axis=1), 1.0, rtol=1e-6)
    c = p.base_price
    np.testing.assert_allclose(p.gt[:, 5], c[:, 5] / c[:, 4] - 1, atol=1e-6)   # float32 prices -> absolute tol


def test_daily_horizon_targets():
    p = _hourly_panel()
    dh = daily_horizon(p)
    assert dh.eligible_ends.tolist() == [7 * i + 6 for i in range(29)]
    e, e2 = 6, 13
    np.testing.assert_allclose(dh.gt[:, e + 1], p.base_price[:, e2] / p.base_price[:, e] - 1, atol=1e-6)
    assert pd.to_datetime(p.timestamps[0], utc=True).year == 2024     # timestamps stored as ns


def test_save_load_and_gics(tmp_path):
    p = _hourly_panel()
    uni = pd.DataFrame({"ticker": ["AAA", "BBB", "ZZZ"], "sector": ["Tech", "Tech", "Energy"],
                        "subindustry": ["Semis", "Software", "Oil"]})
    save_panel(p, tmp_path / "x", uni)
    q = load_panel(tmp_path / "x")
    np.testing.assert_allclose(q.features, p.features)
    assert q.tickers == p.tickers and q.valid_index == p.valid_index and q.eligible_ends is None
    assert gics_hypergraph(tmp_path / "x", "sector").edges == ((0, 1),)
    assert gics_hypergraph(tmp_path / "x", "subindustry").edges == ()
    save_panel(daily_horizon(p), tmp_path / "y", uni)
    assert load_panel(tmp_path / "y").eligible_ends is not None
