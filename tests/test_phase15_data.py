"""Phase 1.5 fidelity audit (docs/phase1_5/C_data_graph.md): programmatic checks of the RSR data and graph
against the raw files, using an independent minimal rebuild that does not import hypergraph.py."""
import collections
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from hypershift.data.rsr import load_rsr
from hypershift.train.loop import gather_batch

R = Path("data/raw/rsr/data")
pytestmark = [pytest.mark.data, pytest.mark.skipif(not R.exists(), reason="RSR data not downloaded")]
MISS = -1234.0


def _tickers(m):
    return [ln.strip().split("\t")[0] for ln in open(R / f"{m}_tickers_qualify_dr-0.98_min-5_smooth.csv") if ln.strip()]


def _raw(m):
    return np.stack([np.loadtxt(R / "2013-01-01" / f"{m}_{t}_1.csv", delimiter=",") for t in _tickers(m)])


@pytest.fixture(scope="module")
def nyse_raw():
    return _raw("NYSE")


@pytest.fixture(scope="module")
def nyse():
    return load_rsr(R, "NYSE", norm="paper")


def _qids(m):
    return [r[1] for r in csv.reader(open(R / f"{m}_wiki.csv"))]


def _selected():
    return {r[0] for r in csv.reader(open(R / "relation/wikidata/selected_wiki_connections.csv"), delimiter=" ")}


def _wiki_raw(m):
    """-> (star dict {(src, prop): set(targets)}, ordered second-order pairs {type: set((i,j))})."""
    by = collections.defaultdict(list)
    for i, q in enumerate(_qids(m)):
        by[q].append(i)
    sel = _selected()
    con = json.loads((R / f"relation/wikidata/{m}_connections.json").read_text())
    star, ordered = collections.defaultdict(set), collections.defaultdict(set)
    for qa, d in con.items():
        for qb, paths in d.items():
            for p in paths:
                typ = "_".join(p)
                if typ not in sel:
                    continue
                for i in by[qa]:
                    for j in by[qb]:
                        if i == j:
                            continue
                        if len(p) == 1:
                            star[(i, p[0])].add(j)
                        else:
                            ordered[typ].add((i, j))
    return star, ordered


def _rebuild(m):
    ix = {t: i for i, t in enumerate(_tickers(m))}
    raw = []
    for mem in json.loads((R / f"relation/sector_industry/{m}_industry_ticker.json").read_text()).values():
        raw.append(tuple(ix[t] for t in mem))
    star, ordered = _wiki_raw(m)
    raw += [(i, *ts) for (i, _), ts in star.items()]
    raw += [p for s in ordered.values() for p in s]
    seen, out = set(), []
    for e in raw:
        t = tuple(sorted(set(e)))
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------- data
def test_ticker_order_and_shapes(nyse_raw):
    t = _tickers("NYSE")
    assert len(t) == len(set(t)) == 1737
    assert [r[0] for r in csv.reader(open(R / "NYSE_wiki.csv"))] == t
    ind = json.loads((R / "relation/sector_industry/NYSE_industry_ticker.json").read_text())
    assert sorted(x for v in ind.values() for x in v) == sorted(t)
    assert nyse_raw.shape == (1737, 1245, 6)
    assert all((a[:, 0] == np.arange(1245)).all() for a in nyse_raw)


def test_dates_offset_29():
    for m, rows in (("NYSE", 1245), ("NASDAQ", 1246)):
        d = [dt.datetime.strptime(x.strip(), "%Y-%m-%d %H:%M:%S") for x in open(R / f"{m}_aver_line_dates.csv") if x.strip()]
        assert len(d) - rows == 29  # MA30 warm-up days precede row 0
        assert all(b > a for a, b in zip(d, d[1:])) and all(x.weekday() < 5 for x in d)
    d = [x.strip() for x in open(R / "NYSE_aver_line_dates.csv") if x.strip()]
    assert d[29 + 756].startswith("2016-01-04") and d[29 + 1008].startswith("2017-01-03")


def test_aapl_known_drop_aligns_with_date_vector():
    # External-knowledge anchor (not in the repo): AAPL fell about 12.35% on 2013-01-24.
    a = np.loadtxt(R / "2013-01-01" / "NASDAQ_AAPL_1.csv", delimiter=",")
    c = a[:, -1]
    r = c[1:300] / c[:299] - 1
    k = int(np.argmin(r)) + 1
    dates = [x.strip() for x in open(R / "NASDAQ_aver_line_dates.csv") if x.strip()]
    assert dates[k + 29].startswith("2013-01-24") and r.min() == pytest.approx(-0.1235, abs=1e-4)


def test_columns_are_ma5_ma10_ma20_ma30_close_and_max_close_is_one(nyse_raw):
    x = nyse_raw
    ok = np.abs(x[:, :, 1:] - MISS) > 1e-8
    close = x[:, :, 5]
    for col, w in ((1, 5), (2, 10), (3, 20), (4, 30)):
        errs = []
        for i in range(0, 1737, 11):
            for t in range(w, 1245):
                win = close[i, t - w + 1: t + 1]
                if (np.abs(win - MISS) < 1e-8).any() or not ok[i, t, col - 1]:
                    continue
                errs.append(abs(win.mean() - x[i, t, col]))
        assert max(errs) < 1e-5, (col, max(errs))
    cl = np.where(np.abs(close - MISS) < 1e-8, -np.inf, close)
    np.testing.assert_allclose(cl.max(axis=1), 1.0, atol=1e-6)  # "paper" norm = full-series max close


def test_missing_mask_counts_and_close_only_definition(nyse_raw):
    miss = np.abs(nyse_raw[:, :, 1:] - MISS) < 1e-8
    assert int(miss[:, :, 4].sum()) == 1389
    assert int((~miss[:, :, 4] & miss[:, :, :4].any(2)).sum()) == 0
    # 331 missing-close days still carry real MAs; the loader keys the mask on the close only
    assert int((miss[:, :, 4] & ~miss[:, :, :4].all(2)).sum()) == 331


def test_gt_formula_and_no_unmasked_fabricated_returns(nyse_raw, nyse):
    close = nyse_raw[:, :, 5]
    mk = np.abs(close - MISS) < 1e-8
    assert (nyse.mask == (~mk)).all()
    c = np.where(mk, np.nan, close)
    ref = np.zeros_like(c)
    ref[:, 1:] = (c[:, 1:] - c[:, :-1]) / c[:, :-1]
    np.testing.assert_allclose(nyse.gt, np.nan_to_num(ref), atol=1e-6)
    orphan = np.zeros_like(mk)
    orphan[:, 1:] = (~mk[:, 1:]) & mk[:, :-1]  # observed today, missing yesterday -> gt stored as 0
    assert int(orphan.sum()) == 349
    # the per-day mask alone does NOT protect these ...
    assert (nyse.gt[orphan] == 0).all() and (nyse.mask[orphan] == 1).all()
    # ... but the window mask in gather_batch (min over input days and the target day) does
    seq = 16
    stocks = np.nonzero(orphan.any(1))[0][:40]
    sub = nyse.subset(stocks)
    so = orphan[stocks]
    ts = np.nonzero(so.any(0))[0]
    ts = ts[ts >= seq]
    _, m, _, _ = gather_batch(sub, ts - seq, seq)
    assert (m.T[so[:, ts]] == 0).all()


def test_nasdaq_last_row_is_partially_missing_and_dropped():
    a = _raw("NASDAQ")
    assert a.shape == (1026, 1246, 6)
    assert int((np.abs(a[:, -1, -1] - MISS) < 1e-8).sum()) == 474  # not all missing, contrary to CLAUDE.md
    d = load_rsr(R, "NASDAQ", norm="paper")
    assert d.features.shape == (1026, 1245, 5)
    kept = np.where(np.abs(a[:, :-1, -1] - MISS) < 1e-8, -np.inf, a[:, :-1, -1]).max(1)
    assert int((kept < 1 - 1e-5).sum()) == 24  # paper scale includes the dropped row for 24 stocks


# ---------------------------------------------------------------- graph
def test_independent_rebuild_matches_cached_v2():
    for m, n_edges in (("NYSE", 4350), ("NASDAQ", 1066)):
        c = json.loads((R / "hypergraph_cache" / f"{m}_industry-wiki.json").read_text())
        assert c["version"] == 2
        cache = {tuple(e) for e in c["edges"]}
        assert len(cache) == len(c["edges"]) == n_edges
        assert cache == set(_rebuild(m))


def test_first_order_stars_grouped_per_source_and_relation():
    star, _ = _wiki_raw("NYSE")
    assert len(star) == 23 and len({i for i, _ in star}) == 23
    star2, _ = _wiki_raw("NASDAQ")
    assert len(star2) == 9 and len({i for i, _ in star2}) == 7  # two sources hold two relation types
    ix = _tickers("NASDAQ").index("GOOGL")
    assert {p for (i, p) in star2 if i == ix} == {"P355", "P155"}


def test_second_order_pairs_have_exact_transposes():
    for m in ("NYSE", "NASDAQ"):
        _, ordered = _wiki_raw(m)
        for k, s in ordered.items():
            a, b = k.split("_")
            if a == b:
                assert {(j, i) for i, j in s} == s
            else:
                assert {(j, i) for i, j in s} == ordered[b + "_" + a]
        assert not {"P31_P31", "P414_P414", "P17_P17"} & set(ordered)


def test_graph_statistics_nyse():
    e = _rebuild("NYSE")
    sz = np.array([len(x) for x in e])
    assert sz.max() == 500 and int((sz >= 3).sum()) == 100 and int((sz == 2).sum()) == 4250
    deg = np.zeros(1737, int)
    for x in e:
        for v in x:
            deg[v] += 1
    assert deg.max() == 114 and int((deg == 0).sum()) == 17 and int((deg >= 31).sum()) == 86
    assert sum(q == "unknown" for q in _qids("NYSE")) == 1140
