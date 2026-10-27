"""Optional Yahoo Finance adjusted-close download for the tutorial."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


def save_metadata(path: Path, start: str, end: str):
    """Record provider and request settings beside the immutable price snapshot."""
    prices = pd.read_csv(path, index_col=0)
    metadata = {
        "provider": "Yahoo Finance via yfinance", "provider_url": "https://finance.yahoo.com/",
        "yfinance_version": yf.__version__, "auto_adjust": True, "interval": "1d",
        "requested_start_inclusive": start, "requested_end_exclusive": end,
        "tickers": prices.columns.tolist(), "rows": len(prices),
        "first_date": str(prices.index[0]), "last_date": str(prices.index[-1]),
        "snapshot_written_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "NVDA",
                        "JPM", "BAC", "GS", "MS", "XOM", "CVX", "COP", "SLB"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive end date.")
    parser.add_argument("--out", type=Path, default=Path("data/prices.csv"))
    args = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]
    if len(tickers) < 2 or len(set(tickers)) != len(tickers):
        parser.error("Provide at least two distinct tickers.")
    if args.out.exists():
        parser.error("Output already exists. Choose a new --out to preserve the data snapshot.")
    # Keep provider caches inside the workspace.
    yf.set_tz_cache_location(str(Path(".cache/yfinance").resolve()))
    raw = yf.download(tickers, start=args.start, end=args.end, interval="1d",
                      auto_adjust=True, group_by="column", multi_level_index=True,
                      threads=False, progress=False, timeout=20)
    if raw is None or raw.empty:
        raise RuntimeError("No data returned. Check network/provider availability, dates, and tickers.")
    prices = raw["Close"].reindex(columns=tickers).copy()
    prices = prices.sort_index()
    if prices.isna().any().any():
        counts = prices.isna().sum()
        raise ValueError(f"Incomplete prices; no rows were silently removed. Missing counts:\n{counts}\n"
                         "Choose a period with complete histories or repair the data at its source.")
    if not prices.map(lambda v: pd.notna(v) and 0 < v < float("inf")).all().all():
        raise ValueError("Provider returned nonpositive or nonfinite prices.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out, index_label="Date")
    save_metadata(args.out, args.start, args.end)
    print(f"Saved {len(prices)} sessions x {len(tickers)} stocks to {args.out.resolve()}")
    print("Adjusted-close teaching dataset; today's chosen tickers introduce survivorship bias.")


if __name__ == "__main__":
    main()
