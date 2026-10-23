"""Paper-trading target generator for the style-only book (docs/strategy_20261022/PAPER_TRADING.md).

  python scripts/paper_trade.py --dev-dryrun                  # as of 2023-12-29 (inspected data): proves the pipeline runs
  python scripts/paper_trade.py --panel P --ohlcv O --raw R --asof YYYY-MM-DD   # live use on an extended panel
Writes results/PAPER_TRADING/targets_<asof>.csv (ticker, weight, leg) and appends a line to log.csv.
Never reads or scores returns after the as-of date.
"""
import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.strategy import paper, engine as EN  # noqa: E402

OUT = Path('results/PAPER_TRADING')


def build_dev(asof):
    from hypershift.featbench.data import load_fb
    from fbaudit_refit import corrected_fb
    fb = corrected_fb(load_fb(2023, 5), 2023)
    t = int(np.searchsorted(fb.dates, np.datetime64(asof), side='right') - 1)
    return fb, t


def build_live(panel, ohlcv, raw, asof):
    from hypershift.fbaudit.prospective_lf import load_range_lf
    fb = load_range_lf(panel, ohlcv, raw, 5)
    t = int(np.searchsorted(fb.dates, np.datetime64(asof), side='right') - 1)
    return fb, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-dryrun', action='store_true')
    ap.add_argument('--panel'); ap.add_argument('--ohlcv'); ap.add_argument('--raw'); ap.add_argument('--asof')
    a = ap.parse_args()
    asof = '2023-12-29' if a.dev_dryrun else a.asof
    fb, t = build_dev(asof) if a.dev_dryrun else build_live(a.panel, a.ohlcv, a.raw, asof)
    train = np.arange(63, t - fb.horizon + 1)
    train = train[(fb.y_ok[:, train] & fb.eligible[:, train]).sum(0) >= 40]
    score = paper.style_scores(fb, train, t)
    tradable = fb.eligible[:, t]
    held_path = OUT / 'held.npz'
    held = np.load(held_path) if held_path.exists() and not a.dev_dryrun else None
    hl = held['long'] if held is not None else np.zeros(len(score), bool)
    hs = held['short'] if held is not None else np.zeros(len(score), bool)
    w, lo, sh = EN.ls_weights(np.where(tradable, score, np.nan), tradable, fb.sectors, fb.beta[:, t], hl, hs, .1, .3)
    OUT.mkdir(parents=True, exist_ok=True)
    tick = fb.panel.tickers
    name = f"targets_{asof}{'_DRYRUN' if a.dev_dryrun else ''}.csv"
    with open(OUT / name, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['ticker', 'weight', 'leg'])
        for i in np.flatnonzero(np.abs(w) > 0):
            wr.writerow([tick[i], f'{w[i]:.6f}', 'long' if i in set(lo) else 'short' if i in set(sh) else 'hedge'])
    if not a.dev_dryrun:
        nl, ns = np.zeros(len(w), bool), np.zeros(len(w), bool)
        nl[lo] = True; ns[sh] = True
        np.savez(held_path, long=nl, short=ns)
    digest = hashlib.sha256((OUT / name).read_bytes()).hexdigest()[:16]
    with open(OUT / 'log.csv', 'a', newline='') as fh:
        csv.writer(fh).writerow([datetime.now(timezone.utc).isoformat(), asof, name, digest, len(lo), len(sh),
                                 f'{w @ np.nan_to_num(fb.beta[:, t]):.2e}'])
    print(f'as of {asof}: {len(lo)} long, {len(sh)} short, gross {np.abs(w).sum():.3f}, '
          f'ex-ante beta {w @ np.nan_to_num(fb.beta[:, t]):+.2e}; wrote {OUT / name} (sha {digest})')


if __name__ == '__main__':
    main()
