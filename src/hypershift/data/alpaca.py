"""Alpaca Market Data v2 daily bars (second source for the Phase 1.5b data gate; DATA_COMPATIBILITY.md Amendment A2).

Network is isolated behind an injectable `http(params) -> (status, json)`; credentials are read at runtime and never logged.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

import pandas as pd

URL = "https://data.alpaca.markets/v2/stocks/bars"
ASOF = "2017-12-08"


def candidates(t: str) -> list[str]:
    out = [t]
    if "-" in t:
        root, suf = t.split("-", 1)
        out.append(f"{root}.{suf}")
        if len(suf) == 1:
            out.append(f"{root}.PR{suf}")
    return list(dict.fromkeys(out))


def parse_bars(bars: list[dict]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(b["t"]).tz_convert("UTC").tz_localize(None).normalize() for b in bars])
    return pd.DataFrame({"c": [float(b["c"]) for b in bars]}, index=idx)


def _cred(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        v = subprocess.check_output(["powershell", "-NoProfile", "-c", f"[Environment]::GetEnvironmentVariable('{name}','User')"], text=True).strip()
    if not v:
        raise RuntimeError(f"{name} not set")
    return v


def make_http():
    import requests
    h = {"APCA-API-KEY-ID": _cred("APCA_API_KEY_ID"), "APCA-API-SECRET-KEY": _cred("APCA_API_SECRET_KEY")}

    def http(params):
        r = requests.get(URL, params=params, headers=h, timeout=60)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    return http


def fetch_bars(symbols, start, end, adjustment, http, sleep=time.sleep, asof=ASOF, return_invalid=False):
    """Fetch daily bars for `symbols` (<= ~100). 429 -> back off and retry; 400 'invalid symbol: X' -> drop X, retry."""
    syms, invalid, out = list(symbols), [], {}
    token = None
    tries = 0
    while syms:
        p = {"symbols": ",".join(syms), "timeframe": "1Day", "start": start, "end": end, "adjustment": adjustment,
             "feed": "sip", "limit": 10000, "asof": asof}
        if token:
            p["page_token"] = token
        status, js = http(p)
        if status == 429 or status >= 500:
            tries += 1
            if tries > 8:
                raise RuntimeError(f"alpaca http {status}")
            sleep(min(60, 5 * tries))
            continue
        if status == 400:
            m = re.search(r"invalid symbol[s]?:?\s*([A-Za-z0-9.\-,\s]+)", str(js.get("message", "")))
            bad = [s.strip() for s in m.group(1).split(",")] if m else []
            bad = [s for s in bad if s in syms]
            if not bad or token:
                raise RuntimeError(f"alpaca 400: {js.get('message')}")
            invalid += bad
            syms = [s for s in syms if s not in bad]
            continue
        if status != 200:
            raise RuntimeError(f"alpaca http {status}")
        tries = 0
        for s, b in (js.get("bars") or {}).items():
            out.setdefault(s, []).extend(b)
        token = js.get("next_page_token")
        if not token:
            break
    res = {s: parse_bars(b) for s, b in out.items() if b}
    return (res, invalid) if return_invalid else res
