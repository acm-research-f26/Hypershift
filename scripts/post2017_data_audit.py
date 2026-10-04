"""Phase 1.5b data gate audit (CPU only). Contract: docs/phase1_5b/DATA_COMPATIBILITY.md (spec fixed before this ran).

  python scripts/post2017_data_audit.py download [--refresh]     # yfinance bars for all 1,737 RSR NYSE tickers -> data/raw/yahoo_post2017/
  python scripts/post2017_data_audit.py analyze                  # R5 overlap test, identity, coverage 2018-2023, R8 survivor bias

HARD RULE: no model, strategy or portfolio return is computed for any date in 2018+. Price-derived statistics use the
2015-01-02..2017-12-08 overlap window only; for 2018+ only bar-availability booleans are used. The R8 check uses the
saved 2017 test predictions (read-only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd  # NOTE: bars.pkl.gz is written by this script itself (own local data); read_pickle is not applied to untrusted files

from hypershift.data import post2017 as P
from hypershift.data.rsr import load_rsr, read_ticker_file

RSR_ROOT = Path("data/raw/rsr/data")
RAW = Path("data/raw/yahoo_post2017")
OUT = Path("docs/phase1_5b")
START, END_EXCL = "2014-11-01", "2024-01-01"
OV_START, OV_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2017-12-08")
POST0, POST1 = pd.Timestamp("2018-01-02"), pd.Timestamp("2023-12-31")
CONVENTIONS = ("raw", "split_adj", "split_div_adj")      # predeclared candidates (a), (b), (c)
POSTHOC = "split_genuine"                                  # POST HOC (found after the audit; not eligible for the predeclared winner)
ALLC = CONVENTIONS + (POSTHOC,)


def candidates(t: str) -> list[str]:
    out = [t]
    if "." in t:
        out.append(t.replace(".", "-"))
    if "-" in t:
        root, suf = t.split("-", 1)
        if len(suf) == 1:
            out.append(f"{root}-P{suf}")
    return list(dict.fromkeys(out))


def download(refresh: bool) -> None:
    import yfinance as yf
    RAW.mkdir(parents=True, exist_ok=True)
    pkl = RAW / "bars.pkl.gz"
    if pkl.exists() and not refresh:
        print("exists:", pkl, "(use --refresh)")
        return
    tickers = read_ticker_file(RSR_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    syms = sorted({s for t in tickers for s in candidates(t)})
    print(len(tickers), "tickers,", len(syms), "symbols")
    bars, empty = {}, []
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=START, end=END_EXCL, auto_adjust=False, actions=True, group_by="ticker",
                                  threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001 (network)
                print("retry", i, repr(e)[:100])
                time.sleep(5 * (attempt + 1))
                raw = None
        if raw is None or raw.empty:
            empty += chunk
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([chunk[:1], raw.columns])
        have = set(raw.columns.get_level_values(0))
        for s in chunk:
            if s not in have:
                empty.append(s)
                continue
            d = raw[s].reindex(columns=["Close", "Adj Close", "Dividends", "Stock Splits"]).dropna(subset=["Close"])
            d[["Dividends", "Stock Splits"]] = d[["Dividends", "Stock Splits"]].fillna(0.0)
            if d.empty:
                empty.append(s)
                continue
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            bars[s] = d.astype("float64")
        print(f"{min(i + 50, len(syms))}/{len(syms)} symbols, {len(bars)} with data", flush=True)
    pd.to_pickle(bars, pkl, compression="gzip")
    (RAW / "download_meta.json").write_text(json.dumps(
        {"yfinance": yf.__version__, "downloaded_utc": pd.Timestamp.now('UTC').isoformat(), "start": START, "end_exclusive": END_EXCL,
         "n_symbols_requested": len(syms), "n_symbols_with_data": len(bars), "symbols_without_data": sorted(empty)}, indent=1))
    print("saved", pkl, len(bars), "symbols with data")


def is_simple_ratio(x: float, tol: float = 2e-3) -> bool:
    """True for genuine split ratios (integer, 1.5/2.5-type, thirds, or their reciprocals for reverse splits).
    Yahoo also books spin-offs/special distributions as non-integer 'splits' (e.g. 1.081, 1.841); RSR does not adjust for those."""
    if not x > 0:
        return False
    for v in (x, 1.0 / x):
        for q in (1, 2, 3):
            p_ = v * q
            if round(p_) >= 2 and abs(p_ - round(p_)) < tol * q:
                return True
    return False


def conv_series(d: pd.DataFrame) -> dict[str, pd.Series]:
    close = d["Close"]
    ratios = d["Stock Splits"].reindex(close.index).fillna(0.0).to_numpy()
    raw = pd.Series(P.undo_splits(close.to_numpy(), ratios), index=close.index)
    non_genuine = np.array([r if (r > 0 and not is_simple_ratio(float(r))) else 0.0 for r in ratios])
    # genuine splits stay adjusted (as in Yahoo Close), spin-off style pseudo-splits are undone
    return {"raw": raw, "split_adj": close, "split_div_adj": d["Adj Close"],
            POSTHOC: pd.Series(P.undo_splits(close.to_numpy(), non_genuine), index=close.index)}


def analyze() -> None:
    bars: dict[str, pd.DataFrame] = pd.read_pickle(RAW / "bars.pkl.gz", compression="gzip")
    meta = json.loads((RAW / "download_meta.json").read_text())
    tickers = read_ticker_file(RSR_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    rsr = load_rsr(RSR_ROOT, "NYSE", norm="paper")
    dates = pd.to_datetime(pd.read_csv(RSR_ROOT / "NYSE_aver_line_dates.csv", header=None)[0])
    cal_rsr = pd.DatetimeIndex(dates.iloc[29:].dt.normalize().to_numpy())
    assert len(cal_rsr) == rsr.num_steps == 1245
    ov = np.nonzero((cal_rsr >= OV_START) & (cal_rsr <= OV_END))[0]
    r_rsr = np.where((rsr.mask[:, 1:] > 0) & (rsr.mask[:, :-1] > 0), rsr.gt[:, 1:], np.nan)
    r_rsr = np.concatenate([np.full((rsr.num_nodes, 1), np.nan), r_rsr], axis=1)[:, ov]
    cal_ov = cal_rsr[ov]
    # the return of the first overlap day needs the previous calendar day: use the full RSR calendar, then slice
    all_days = set()
    for d in bars.values():
        all_days |= set(d.index)
    # --- calendars
    cnt = pd.Series(0, index=sorted(all_days))
    for d in bars.values():
        cnt.loc[d.index] += 1
    nbars = max(1, len(bars))
    ydays = cnt.index[cnt.to_numpy() >= 0.5 * np.median(cnt.to_numpy()[-500:])]
    ydays = ydays[(ydays >= pd.Timestamp(START)) & (ydays <= POST1)]
    rsr_not_y = sorted(set(cal_rsr[cal_rsr >= pd.Timestamp(START)]) - set(ydays))
    y_not_rsr = sorted(set(ydays[(ydays >= pd.Timestamp("2014-11-19")) & (ydays <= OV_END)]) - set(cal_rsr))
    # --- Yahoo split convention check on the first RSR symbol with a split on/after 2018-01-01 (Close must be split-adjusted)
    split_check = None
    for t in tickers:
        for sym in candidates(t):
            d = bars.get(sym)
            if d is None:
                continue
            ev = d.index[(d["Stock Splits"] >= 2) & (d.index >= pd.Timestamp("2018-01-01"))]
            if len(ev):
                e = ev[0]
                cs = conv_series(d)
                prev = d.index[d.index < e][-1]
                split_check = {"symbol": sym, "split_date": str(e.date()), "ratio": float(d.loc[e, "Stock Splits"]),
                               "Close_prev_day": float(cs["split_adj"].loc[prev]), "Close_split_day": float(cs["split_adj"].loc[e]),
                               "raw_prev_day_reconstructed": float(cs["raw"].loc[prev]), "raw_split_day": float(cs["raw"].loc[e])}
                break
        if split_check:
            break
    # --- R5 overlap per symbol candidate and convention (returns on the RSR calendar)
    rows, share_cache = [], {}
    for i, t in enumerate(tickers):
        for sym in candidates(t):
            d = bars.get(sym)
            if d is None:
                continue
            rets = {}
            for cv, ser in conv_series(d).items():
                r = P.returns_on_calendar(ser, cal_rsr)[ov]
                rets[cv] = r
            rows.append((i, t, sym, rets))
    # first symbol with >= 1 matched-eligible day in the overlap for each ticker (pooled R5 population, predeclared)
    first_sym = {}
    for i, t, sym, rets in rows:
        if i in first_sym:
            continue
        m, n = P.match_share(rets["split_div_adj"], r_rsr[i], P.TOL)
        _, n_any = P.match_share(rets["split_adj"], r_rsr[i], P.TOL)
        if n_any > 0:
            first_sym[i] = (t, sym, rets)
    pooled = {}
    for cv in ALLC:
        M = N = 0
        for i, (t, sym, rets) in first_sym.items():
            m, n = P.match_share(rets[cv], r_rsr[i], P.TOL)
            M, N = M + m, N + n
        pooled[cv] = {"matched": M, "total": N, "share": M / N if N else 0.0}
    winner = max(CONVENTIONS, key=lambda c: pooled[c]["share"])
    # --- identity under the winning convention: first candidate symbol that passes; else report first with data
    ident = {}
    per_ticker_share = {c: {} for c in ALLC}
    for i, t, sym, rets in rows:
        ok, share, n = P.identity_test(rets[winner], r_rsr[i])
        for cv in ALLC:
            m2, n2 = P.match_share(rets[cv], r_rsr[i], P.TOL)
            per_ticker_share[cv].setdefault(i, (sym, m2 / n2 if n2 else 0.0, n2))
        if i not in ident or (ok and not ident[i]["ok"]):
            ident[i] = {"ticker": t, "symbol": sym, "ok": ok, "share": share, "n": n}
    # subset of pooled shares restricted to identity-passing tickers (secondary statistic)
    sub = {}
    for cv in ALLC:
        M = N = 0
        for i, v in ident.items():
            if not v["ok"]:
                continue
            rets = next(r for (ii, tt, ss, r) in rows if ii == i and ss == v["symbol"])
            m, n = P.match_share(rets[cv], r_rsr[i], P.TOL)
            M, N = M + m, N + n
        sub[cv] = {"matched": M, "total": N, "share": M / N if N else 0.0}
    id_by_conv = {cv: len({i for (i, t, sym, rets) in rows if P.identity_test(rets[cv], r_rsr[i])[0]}) for cv in ALLC}
    reasons = {"no_yahoo_data": len(tickers) - len(ident),
               "fail_matched_days_lt_250": sum(1 for v in ident.values() if not v["ok"] and v["n"] < P.MIN_MATCHED),
               "fail_share_lt_0.95_with_n_ge_250": sum(1 for v in ident.values() if not v["ok"] and v["n"] >= P.MIN_MATCHED),
               "pass": sum(1 for v in ident.values() if v["ok"])}
    split_days = {"events_in_overlap": 0, "raw_matches_rsr": 0, "split_adj_matches_rsr": 0, "both_match": 0, "neither": 0}
    for i, v in ident.items():
        if not v["ok"]:
            continue
        d = bars[v["symbol"]]
        ev = d.index[(d["Stock Splits"] > 0) & (d.index >= OV_START) & (d.index <= OV_END)]
        rr = next(r for (ii, tt, ss, r) in rows if ii == i and ss == v["symbol"])
        for e in ev:
            k = np.nonzero(cal_ov == e)[0]
            if not len(k) or not np.isfinite(r_rsr[i, k[0]]):
                continue
            k = k[0]
            m_raw, m_adj = abs(rr["raw"][k] - r_rsr[i, k]) < P.TOL, abs(rr["split_adj"][k] - r_rsr[i, k]) < P.TOL
            split_days["events_in_overlap"] += 1
            split_days["raw_matches_rsr"] += int(m_raw)
            split_days["split_adj_matches_rsr"] += int(m_adj)
            split_days["both_match"] += int(m_raw and m_adj)
            split_days["neither"] += int(not m_raw and not m_adj)
    # --- post-2017 availability (bars only, no returns)
    ycal = pd.DatetimeIndex(ydays)
    post_cal = ycal[(ycal >= pd.Timestamp("2017-12-11")) & (ycal <= POST1)]
    obs_full = np.zeros((len(tickers), len(ycal)), dtype=bool)
    sel_symbol = [None] * len(tickers)
    for i, v in ident.items():
        d = bars[v["symbol"]]
        sel_symbol[i] = v["symbol"]
        obs_full[i] = ycal.isin(d.index)
    # continuity rule on the part after the overlap end (availability only)
    start_post = int(np.searchsorted(ycal, pd.Timestamp("2017-12-11")))
    avail_ident = np.zeros_like(obs_full)
    for i in range(len(tickers)):
        if i in ident and ident[i]["ok"]:
            o = obs_full[i].copy()
            full = o.copy()
            full[start_post - 1:] = P.truncate_gaps(o[start_post - 1:], P.GAP_DAYS)   # continuity rule, availability only
            avail_ident[i] = full
    elig = np.stack([P.availability_mask(a) for a in avail_ident])
    years = list(range(2018, 2024))
    cov = []
    for y in years:
        sel = np.nonzero(ycal.year == y)[0]
        sel = sel[ycal[sel] >= POST0]
        any_data = obs_full[:, sel].any(axis=1)
        any_ident = avail_ident[:, sel].any(axis=1)
        full_year = elig[:, sel].all(axis=1)
        cov.append({"year": y, "trading_days": int(len(sel)), "nodes_with_any_yahoo_bar": int(any_data.sum()),
                    "nodes_identity_pass_any_bar": int(any_ident.sum()), "nodes_eligible_all_year": int(full_year.sum()),
                    "mean_daily_eligible": float(elig[:, sel].sum(axis=0).mean()), "min_daily_eligible": int(elig[:, sel].sum(axis=0).min()),
                    "max_daily_eligible": int(elig[:, sel].sum(axis=0).max())})
    warm = np.searchsorted(ycal, POST0)
    n_warm = int(warm)
    elig_20180102 = int(elig[:, warm].sum())
    # --- R8 survivor bias with the saved 2017 test predictions (read-only)
    from hypershift.eval.forensics import hold_all, load_run, perf, portfolio
    id_ok = np.array([i in ident and ident[i]["ok"] for i in range(len(tickers))])
    last_cols = np.nonzero(ycal >= pd.Timestamp("2023-12-01"))[0]
    alive = id_ok & avail_ident[:, last_cols].any(axis=1)
    subsets = {"full_universe_1737": np.ones(len(tickers), bool), "identity_pass": id_ok, "identity_pass_alive_2023-12": alive}
    r8 = {}
    for name, sset in subsets.items():
        srs, has, ns = [], [], []
        for seed in range(5):
            run = load_run("R5_f2_alpha0_train", "HH", seed)
            mk = run.mask & sset[:, None]
            r, _ = portfolio(run.pred, run.gt, mk)
            srs.append(perf(r)["sr"])
            has.append(perf(hold_all(run.gt, mk))["sr"])
            ns.append(float(mk.sum(axis=0).mean()))
        r8[name] = {"nodes": int(sset.sum()), "mean_daily_eligible_2017": float(np.mean(ns)), "top5_sr_per_seed": [float(x) for x in srs],
                    "top5_sr_mean": float(np.mean(srs)), "hold_all_sr": float(np.mean(has))}
    # --- write outputs
    tab = []
    for i, t in enumerate(tickers):
        v = ident.get(i)
        row = {"idx": i, "ticker": t, "symbol": v["symbol"] if v else "", "has_data": bool(v), "identity_ok": bool(v and v["ok"]),
               "share_" + winner: v["share"] if v else np.nan, "n_matched_eligible": v["n"] if v else 0}
        for cv in ALLC:
            row["share_" + cv] = per_ticker_share[cv][i][1] if i in per_ticker_share[cv] else np.nan
        if i in ident:
            o = obs_full[i]
            row["first_bar"] = str(ycal[np.argmax(o)].date()) if o.any() else ""
            row["last_bar_through_2023"] = str(ycal[len(o) - 1 - np.argmax(o[::-1])].date()) if o.any() else ""
            row["gap_truncated"] = bool(row["identity_ok"] and (avail_ident[i] != obs_full[i])[start_post:].any())
        tab.append(row)
    tdf = pd.DataFrame(tab)
    tdf.to_csv(OUT / "post2017_audit_tickers.csv", index=False)
    n_data = int(tdf.has_data.sum())
    res = {"meta": meta, "n_tickers": len(tickers), "n_with_yahoo_data": n_data, "n_identity_pass": int(tdf.identity_ok.sum()),
           "overlap_window": [str(OV_START.date()), str(OV_END.date())], "overlap_days_in_rsr_calendar": int(len(cal_ov)),
           "r5_pooled_all_tickers_with_data": pooled, "winner": winner,
           "r5_pooled_identity_pass_subset": sub, "identity_reasons": reasons, "tickers_passing_identity_by_convention": id_by_conv, "split_days_in_overlap_identity_pass": split_days, "calendar_rsr_not_in_yahoo_days": [str(d.date()) for d in rsr_not_y][:20],
           "calendar_yahoo_not_in_rsr_overlap": [str(d.date()) for d in y_not_rsr][:20], "n_rsr_not_yahoo": len(rsr_not_y),
           "n_yahoo_not_rsr": len(y_not_rsr), "split_convention_check": split_check, "coverage_by_year": cov,
           "warmup_yahoo_days_before_2018-01-02": n_warm, "eligible_nodes_on_2018-01-02": elig_20180102,
           "median_per_ticker_share": {cv: float(np.nanmedian([v[1] for v in per_ticker_share[cv].values()])) for cv in ALLC},
           "r8_survivor_bias_2017_test_HH_seeds0-4": r8}
    (OUT / "post2017_audit_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("meta",)}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "analyze"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    download(a.refresh) if a.cmd == "download" else analyze()
