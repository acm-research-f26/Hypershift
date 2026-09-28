"""Examples:
  python scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
  python scripts/fetch_fresh.py --source alpaca --kind hourly --start 2018-01-01 --end 2026-09-01 --name sp500_1h
  python scripts/fetch_fresh.py --source yf --kind hourly --name sp500_1h          (last ~730 days only)
"""
import argparse
from pathlib import Path

import pandas as pd

from hypershift.data.fresh import (
    NY, build_panel, daily_horizon, fetch_alpaca, fetch_yf, load_universe, save_panel, to_daily, to_rth_hourly,
)

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["yf", "alpaca"], required=True)
ap.add_argument("--kind", choices=["daily", "hourly"], required=True)
ap.add_argument("--start", default=None)
ap.add_argument("--end", default=None)
ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
ap.add_argument("--name", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N tickers only (debugging)")
ap.add_argument("--max-missing", type=float, default=0.05)
args = ap.parse_args()

root = Path("data/fresh")
uni = load_universe()
tickers = uni.ticker.tolist()[: args.limit or None]
print(f"universe: {len(tickers)} tickers (current S&P 500 -> survivorship-biased, see D17)")

if args.kind == "daily":
    if args.source != "yf":
        raise SystemExit("daily bars: use --source yf")
    bars = fetch_yf(tickers, "1d", start=args.start, end=args.end)
    (root / args.name).mkdir(parents=True, exist_ok=True)
    bars.to_csv(root / args.name / "bars.csv.gz", index=False)
    panel = build_panel(bars, max_missing=args.max_missing)
    save_panel(panel, root / args.name, uni)
    print(args.name, panel.features.shape, "valid/test index", panel.valid_index, panel.test_index)
else:
    if args.source == "alpaca":
        bars = fetch_alpaca(tickers, args.start, args.end, feed=args.feed)
    else:
        bars = to_rth_hourly(fetch_yf(tickers, "60m", period="730d"))
    raw_dir = root / f"{args.name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(raw_dir / "hourly_bars.csv.gz", index=False)
    h0 = build_panel(bars, max_missing=args.max_missing)
    if len(h0.tickers) < 300:
        print(f"WARNING D13: only {len(h0.tickers)} tickers survive; consider --max-missing 0.10")
    daily = build_panel(to_daily(bars), tickers=h0.tickers)
    ts_d = pd.to_datetime(daily.timestamps, utc=True).tz_convert(NY)
    hourly = build_panel(bars, tickers=daily.tickers, split_at=(ts_d[daily.valid_index], ts_d[daily.test_index]))
    save_panel(hourly, root / f"{args.name}_hourly", uni)
    save_panel(daily, root / f"{args.name}_daily", uni)
    save_panel(daily_horizon(hourly), root / f"{args.name}_hday", uni)
    for suffix, p in (("hourly", hourly), ("daily", daily)):
        print(suffix, p.features.shape, "valid/test", p.valid_index, p.test_index)
