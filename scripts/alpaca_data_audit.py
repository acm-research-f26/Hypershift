"""Phase 1.5b data gate, second source: Alpaca SIP daily bars (CPU only). Contract: DATA_COMPATIBILITY.md Amendment A2.

  python scripts/alpaca_data_audit.py download [--refresh]   # raw/split/all bars for the 1,737 RSR NYSE tickers -> data/raw/alpaca_post2017/
  python scripts/alpaca_data_audit.py analyze                # identity on 2016-01-04..2017-12-08, convention table, coverage/attrition, R8 reverse check

HARD RULE: no model, strategy or portfolio return for any 2018+ date; only bar-availability booleans after 2017-12-08.
Credentials come from the Windows user environment at runtime and are never printed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd  # pickles are our own local files

from hypershift.data import alpaca as A
from hypershift.data import post2017 as P
from hypershift.data.rsr import load_rsr, read_ticker_file

RSR_ROOT = Path("data/raw/rsr/data")
RAW = Path("data/raw/alpaca_post2017")
OUT = Path("docs/phase1_5b")
START, END = "2016-01-04", "2023-12-31"
OV_START, OV_END = pd.Timestamp("2016-01-04"), pd.Timestamp("2017-12-08")
POST0, POST1 = pd.Timestamp("2018-01-02"), pd.Timestamp("2023-12-31")
ADJ = ("raw", "split", "all")
IDENT_ADJ = "split"      # A2 item 2


def symbols_for(tickers):
    return sorted({s for t in tickers for s in A.candidates(t) if "-" not in s})


def download(refresh: bool) -> None:
    pkl = RAW / "bars.pkl.gz"
    RAW.mkdir(parents=True, exist_ok=True)
    if pkl.exists() and not refresh:
        print("exists:", pkl)
        return
    tickers = read_ticker_file(RSR_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    syms = symbols_for(tickers)
    dash = sorted({s for t in tickers for s in A.candidates(t) if "-" in s})
    print(len(tickers), "tickers,", len(syms), "symbols,", len(dash), "dash symbols skipped (rejected by API)")
    raw_http = A.make_http()
    last = [0.0]

    def http(params):
        wait = 0.33 - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.time()
        return raw_http(params)

    bars = {a: {} for a in ADJ}
    for adj in ADJ:
        for i in range(0, len(syms), 100):
            bars[adj].update(A.fetch_bars(syms[i:i + 100], START, END, adj, http=http))
            print(adj, min(i + 100, len(syms)), "/", len(syms), "with data", len(bars[adj]), flush=True)
    pd.to_pickle(bars, pkl, compression="gzip")
    (RAW / "download_meta.json").write_text(json.dumps({
        "downloaded_utc": pd.Timestamp.now("UTC").isoformat(), "start": START, "end": END, "feed": "sip", "asof": A.ASOF,
        "n_symbols_requested": len(syms), "dash_symbols_skipped": dash,
        "n_symbols_with_data": {a: len(bars[a]) for a in ADJ},
        "symbols_without_data": sorted(set(syms) - set(bars["split"]))}, indent=1))
    print("saved", pkl)


def analyze() -> None:
    import post2017_data_audit as Y
    from hypershift.eval.forensics import hold_all, load_run, perf, portfolio
    bars = pd.read_pickle(RAW / "bars.pkl.gz", compression="gzip")
    meta = json.loads((RAW / "download_meta.json").read_text())
    tickers = read_ticker_file(RSR_ROOT / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
    rsr = load_rsr(RSR_ROOT, "NYSE", norm="paper")
    dates = pd.to_datetime(pd.read_csv(RSR_ROOT / "NYSE_aver_line_dates.csv", header=None)[0])
    cal_rsr = pd.DatetimeIndex(dates.iloc[29:].dt.normalize().to_numpy())
    ov = np.nonzero((cal_rsr >= OV_START) & (cal_rsr <= OV_END))[0]
    r_rsr = np.where((rsr.mask[:, 1:] > 0) & (rsr.mask[:, :-1] > 0), rsr.gt[:, 1:], np.nan)
    r_rsr = np.concatenate([np.full((rsr.num_nodes, 1), np.nan), r_rsr], axis=1)[:, ov]
    cal_ov = cal_rsr[ov]
    cnt = {}
    for d in bars["split"].values():
        for x in d.index:
            cnt[x] = cnt.get(x, 0) + 1
    cnt = pd.Series(cnt).sort_index()
    adays = cnt.index[cnt.to_numpy() >= 0.5 * np.median(cnt.to_numpy()[-500:])]
    adays = adays[(adays >= OV_START) & (adays <= POST1)]
    rsr_not_a = sorted(set(cal_ov) - set(adays))
    a_not_rsr = sorted(set(adays[(adays <= OV_END)]) - set(cal_ov))
    rows = []
    for i, t in enumerate(tickers):
        for sym in A.candidates(t):
            if sym not in bars["split"]:
                continue
            rets = {a: P.returns_on_calendar(bars[a][sym]["c"], cal_rsr)[ov] for a in ADJ if sym in bars[a]}
            if len(rets) == len(ADJ):
                rows.append((i, t, sym, rets))
    first_sym = {}
    for i, t, sym, rets in rows:
        if i not in first_sym and P.match_share(rets[IDENT_ADJ], r_rsr[i])[1] > 0:
            first_sym[i] = (sym, rets)
    pooled = {}
    for a in ADJ:
        M = N = 0
        for i, (sym, rets) in first_sym.items():
            m, n = P.match_share(rets[a], r_rsr[i])
            M, N = M + m, N + n
        pooled[a] = {"matched": M, "total": N, "share": M / N if N else 0.0}
    ident = {}
    for i, t, sym, rets in rows:
        ok, share, n = P.identity_test(rets[IDENT_ADJ], r_rsr[i])
        if i not in ident or (ok and not ident[i]["ok"]):
            ident[i] = {"ticker": t, "symbol": sym, "ok": ok, "share": share, "n": n, "rets": rets}
    id_by_adj = {a: len({i for (i, t, sym, rets) in rows if P.identity_test(rets[a], r_rsr[i])[0]}) for a in ADJ}
    sub = {}
    for a in ADJ:
        M = N = 0
        for i, v in ident.items():
            if v["ok"]:
                m, n = P.match_share(v["rets"][a], r_rsr[i])
                M, N = M + m, N + n
        sub[a] = {"matched": M, "total": N, "share": M / N if N else 0.0}
    reasons = {"no_alpaca_data": len(tickers) - len(ident),
               "fail_matched_days_lt_250": sum(1 for v in ident.values() if not v["ok"] and v["n"] < P.MIN_MATCHED),
               "fail_share_lt_0.95_with_n_ge_250": sum(1 for v in ident.values() if not v["ok"] and v["n"] >= P.MIN_MATCHED),
               "pass": sum(1 for v in ident.values() if v["ok"])}
    fail_shares = [v["share"] for v in ident.values() if not v["ok"] and v["n"] >= P.MIN_MATCHED]
    ev = {"events": 0, "raw_matches_rsr": 0, "split_matches_rsr": 0, "neither": 0}
    for i, v in ident.items():
        if not v["ok"]:
            continue
        rr, rs = v["rets"]["raw"], v["rets"]["split"]
        k = np.nonzero(np.isfinite(rr) & np.isfinite(rs) & np.isfinite(r_rsr[i]) & (np.abs(rr - rs) > 0.05))[0]   # genuine split-sized differences (smaller ones are 2-decimal price rounding)
        for j in k:
            mr, ms = abs(rr[j] - r_rsr[i, j]) < P.TOL, abs(rs[j] - r_rsr[i, j]) < P.TOL
            ev["events"] += 1
            ev["raw_matches_rsr"] += int(mr)
            ev["split_matches_rsr"] += int(ms)
            ev["neither"] += int(not mr and not ms)
    ybars = pd.read_pickle(Path("data/raw/yahoo_post2017/bars.pkl.gz"), compression="gzip")
    spin = {"yahoo_pseudo_split_events_on_alpaca_identity_pass": 0, "alpaca_split_return_matches_rsr": 0, "alpaca_raw_return_matches_rsr": 0}
    for i, v in ident.items():
        if not v["ok"]:
            continue
        for ysym in Y.candidates(v["ticker"]):
            d = ybars.get(ysym)
            if d is None:
                continue
            for e in d.index[(d["Stock Splits"] > 0) & (d.index >= OV_START) & (d.index <= OV_END)]:
                if Y.is_simple_ratio(float(d.loc[e, "Stock Splits"])):
                    continue
                k = np.nonzero(cal_ov == e)[0]
                if len(k) and np.isfinite(r_rsr[i, k[0]]) and np.isfinite(v["rets"]["split"][k[0]]):
                    spin["yahoo_pseudo_split_events_on_alpaca_identity_pass"] += 1
                    spin["alpaca_split_return_matches_rsr"] += int(abs(v["rets"]["split"][k[0]] - r_rsr[i, k[0]]) < P.TOL)
                    spin["alpaca_raw_return_matches_rsr"] += int(abs(v["rets"]["raw"][k[0]] - r_rsr[i, k[0]]) < P.TOL)
            break
    ytab = pd.read_csv(OUT / "post2017_audit_tickers.csv")
    yfail = ytab[(ytab.has_data) & (~ytab.identity_ok) & (ytab.n_matched_eligible >= P.MIN_MATCHED)].idx.tolist()
    ynodata = ytab[~ytab.has_data].idx.tolist()
    yshort = ytab[(ytab.has_data) & (ytab.n_matched_eligible < P.MIN_MATCHED)].idx.tolist()

    def npass(idx):
        return sum(1 for i in idx if i in ident and ident[i]["ok"])
    ycal = pd.DatetimeIndex(adays)
    n = len(tickers)
    obs = np.zeros((n, len(ycal)), bool)
    for i, v in ident.items():
        obs[i] = ycal.isin(bars["split"][v["symbol"]].index)
    start_post = int(np.searchsorted(ycal, pd.Timestamp("2017-12-11")))
    avail = np.zeros_like(obs)
    for i, v in ident.items():
        if v["ok"]:
            o = obs[i].copy()
            o[start_post - 1:] = P.truncate_gaps(o[start_post - 1:], P.GAP_DAYS)
            avail[i] = o
    elig = np.stack([P.availability_mask(a) for a in avail])
    last_idx = np.array([len(ycal) - 1 - int(np.argmax(a[::-1])) if a.any() else -1 for a in avail])
    cov = []
    for y in range(2018, 2024):
        sel = np.nonzero((ycal.year == y) & (ycal >= POST0))[0]
        s0, s1 = sel[0], sel[-1]
        sa, ea = elig[:, s0], elig[:, s1]
        cov.append({"year": y, "trading_days": int(len(sel)), "nodes_with_any_bar_identity_pass": int(avail[:, sel].any(axis=1).sum()),
                    "alive_at_year_start": int(sa.sum()), "alive_at_year_end": int(ea.sum()), "left_during_year": int((sa & ~ea).sum()),
                    "eligible_all_year": int(elig[:, sel].all(axis=1).sum()), "mean_daily_eligible": float(elig[:, sel].sum(axis=0).mean()),
                    "min_daily_eligible": int(elig[:, sel].sum(axis=0).min()), "max_daily_eligible": int(elig[:, sel].sum(axis=0).max())})
    id_ok = np.array([i in ident and ident[i]["ok"] for i in range(n)])
    last_day = len(ycal) - 1
    alive_end = id_ok & (last_idx >= last_day - 2)
    n_ended_before = int((id_ok & (last_idx < last_day - 2)).sum())
    warm = int(np.searchsorted(ycal, POST0))
    subsets = {"full_universe_1737": np.ones(n, bool), "alpaca_identity_pass": id_ok, "alpaca_pass_alive_2023-12": alive_end}
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
    tab = []
    for i, t in enumerate(tickers):
        v = ident.get(i)
        row = {"idx": i, "ticker": t, "symbol": v["symbol"] if v else "", "has_data": bool(v), "identity_ok": bool(v and v["ok"]),
               "share_split": v["share"] if v else np.nan, "n_matched_eligible": v["n"] if v else 0,
               "yahoo_identity_ok": bool(ytab.identity_ok[i])}
        if v and obs[i].any():
            row["first_bar"] = str(ycal[np.argmax(obs[i])].date())
            row["last_bar"] = str(ycal[len(ycal) - 1 - np.argmax(obs[i][::-1])].date())
        tab.append(row)
    pd.DataFrame(tab).to_csv(OUT / "alpaca_audit_tickers.csv", index=False)
    res = {"meta": meta, "n_tickers": n, "n_with_alpaca_data": len(ident), "n_identity_pass": int(id_ok.sum()),
           "overlap_window": [str(OV_START.date()), str(OV_END.date())], "overlap_days": int(len(cal_ov)),
           "r5_pooled_all_tickers_with_data": pooled, "r5_pooled_identity_pass_subset": sub,
           "tickers_passing_identity_by_adjustment": id_by_adj, "identity_reasons": reasons,
           "identity_fail_share_quantiles_n_ge_250": [float(x) for x in np.quantile(fail_shares, [0, .1, .5, .9, 1])] if fail_shares else [],
           "alpaca_raw_vs_split_event_days_identity_pass": ev, "spinoff_check_vs_yahoo_pseudo_splits": spin,
           "yahoo_identity_failures_n": len(yfail), "yahoo_identity_failures_passing_on_alpaca": npass(yfail),
           "yahoo_no_data_n": len(ynodata), "yahoo_no_data_passing_on_alpaca": npass(ynodata),
           "yahoo_short_n": len(yshort), "yahoo_short_passing_on_alpaca": npass(yshort),
           "n_rsr_days_not_in_alpaca_days": len(rsr_not_a), "n_alpaca_days_not_in_rsr_overlap": len(a_not_rsr),
           "rsr_not_alpaca_first": [str(d.date()) for d in rsr_not_a[:10]], "alpaca_not_rsr_first": [str(d.date()) for d in a_not_rsr[:10]],
           "coverage_by_year": cov, "warmup_alpaca_days_before_2018-01-02": warm, "eligible_nodes_on_2018-01-02": int(elig[:, warm].sum()),
           "identity_pass_with_last_bar_before_2023-12-29": n_ended_before, "identity_pass_alive_at_end": int(alive_end.sum()),
           "r8_reverse_survivor_2017_test_HH_seeds0-4": r8}
    (OUT / "alpaca_audit_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "analyze"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    download(a.refresh) if a.cmd == "download" else analyze()
