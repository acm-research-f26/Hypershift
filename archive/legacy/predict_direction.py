"""Use a saved direction classifier on the latest close in a historical CSV."""
import argparse
import torch
from stock_gnn import load_prices
from train_direction import predict_direction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', default='data/prices.csv')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result = predict_direction(checkpoint, load_prices(args.csv))
    print(result.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nThese are model estimates at the last CSV date, not live prices or verified future outcomes.')
