"""Alpaca second source (DATA_COMPATIBILITY.md Amendment A2): parsing, pagination, retries, symbol candidates. No network."""
import pandas as pd

from hypershift.data import alpaca as A


def _bar(t, c):
    return {"t": t, "o": c, "h": c, "l": c, "c": c, "v": 1, "n": 1, "vw": c}


def test_candidates():
    assert A.candidates("AAPL") == ["AAPL"]
    assert A.candidates("BRK.B") == ["BRK.B"]
    assert A.candidates("AGO-B") == ["AGO-B", "AGO.B", "AGO.PRB"]


def test_parse_bars_dates_are_naive_utc_dates():
    d = A.parse_bars([_bar("2016-01-04T05:00:00Z", 10.0), _bar("2016-01-05T05:00:00Z", 11.0)])
    assert list(d.index) == [pd.Timestamp("2016-01-04"), pd.Timestamp("2016-01-05")]
    assert list(d["c"]) == [10.0, 11.0]


def test_fetch_paginates_and_merges():
    calls = []

    def http(params):
        calls.append(dict(params))
        if "page_token" not in params:
            return 200, {"bars": {"A": [_bar("2016-01-04T05:00:00Z", 1.0)]}, "next_page_token": "tok"}
        return 200, {"bars": {"A": [_bar("2016-01-05T05:00:00Z", 2.0)], "B": [_bar("2016-01-05T05:00:00Z", 3.0)]}, "next_page_token": None}

    out = A.fetch_bars(["A", "B"], "2016-01-04", "2023-12-31", "split", http=http, sleep=lambda s: None)
    assert len(calls) == 2 and calls[1]["page_token"] == "tok" and calls[0]["asof"] == "2017-12-08"
    assert list(out["A"]["c"]) == [1.0, 2.0] and list(out["B"]["c"]) == [3.0]


def test_fetch_retries_429_then_succeeds():
    seq = iter([(429, {}), (200, {"bars": {"A": [_bar("2016-01-04T05:00:00Z", 1.0)]}, "next_page_token": None})])
    slept = []
    out = A.fetch_bars(["A"], "2016-01-04", "2023-12-31", "raw", http=lambda p: next(seq), sleep=slept.append)
    assert slept and "A" in out


def test_fetch_invalid_symbol_400_drops_and_retries_rest():
    def http(params):
        syms = params["symbols"].split(",")
        if "BAD-X" in syms:
            return 400, {"message": "invalid symbol: BAD-X"}
        return 200, {"bars": {s: [_bar("2016-01-04T05:00:00Z", 1.0)] for s in syms}, "next_page_token": None}

    out, bad = A.fetch_bars(["A", "BAD-X", "B"], "2016-01-04", "2023-12-31", "raw", http=http, sleep=lambda s: None, return_invalid=True)
    assert set(out) == {"A", "B"} and bad == ["BAD-X"]
