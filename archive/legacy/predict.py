"""Apply a saved tutorial GNN to the latest complete row of a price CSV."""

import argparse

import torch

from stock_gnn import load_prices, predict_latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prices = load_prices(args.csv)
    print(f"Source: {checkpoint['source']}")
    print(f"Training labels through {checkpoint['train_last_target']}; "
          f"validation labels through {checkpoint['validation_last_target']}")
    horizon = checkpoint.get("horizon_sessions", 1)  # Original daily checkpoints remain usable.
    print(f"As of {prices.index[-1].date()}; total return forecast over the next {horizon} trading session(s):")
    print(predict_latest(checkpoint, prices).to_string())
    print("Units: percentage points. These are forecasts, not trade instructions.")


if __name__ == "__main__":
    main()
