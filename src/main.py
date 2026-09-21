import torch
from torch_geometric.loader import DataLoader

from data.data import (
    download_stock_data,
    prepare_graph_datasets,
    zero_return_baseline_mse,
)
from src.gnn import GNN
from src.train import evaluate, train

TICKERS = ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "WMT", "CAT")

def main():
    torch.manual_seed(727)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    close, volume = download_stock_data(TICKERS) # From Yahoo Finance
    splits = prepare_graph_datasets(close, volume) # Creates train, validation, test splits
    train_loader = DataLoader(splits.train, batch_size=32, shuffle=True) # supplies training batches
    validation_loader = DataLoader(splits.validation, batch_size=32, shuffle=False) # used to measure validation MSE,  MAE, amongst other statistics
    test_loader = DataLoader(splits.test, batch_size=32, shuffle=False)
    node_features = splits.train[0].x
    if not isinstance(node_features, torch.Tensor):
        raise TypeError("Each graph sample must contain tensor node features.")
    model = GNN(in_channels=node_features.shape[1])
    loss_fn = torch.nn.MSELoss()
    history = train(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        epochs=100,
        patience=15,
        tolerance=1e-6,
        device=device,
    )
    test_metrics = evaluate(model, test_loader, device)
    baseline_mse = zero_return_baseline_mse(splits.test)
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Stocks: {', '.join(splits.tickers)}")
    print(f"Samples: {len(splits.train)} train / "
        f"{len(splits.validation)} validation / "
        f"{len(splits.test)} test")
    print(f"Node features: {node_features.shape}")
    print(f"Undirected graph edges: {splits.edge_index.shape[1] // 2}")
    best_index = history["best_epoch"] - 1

    print("\nTraining")
    print("-" * 40)
    print(f"Epochs run: {history['epochs_ran']}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best train MSE: {history['train_loss'][best_index]:.8f}")
    print(f"Best validation MSE: {history['best_validation_mse']:.8f}")

    print("\nTest results")
    print("-" * 40)
    print(f"MSE: {test_metrics['mse']:.8f}")
    print(f"MAE: {test_metrics['mae']:.8f}")
    print(f"Directional accuracy: "
        f"{100 * test_metrics['directional_accuracy']:.2f}%")
    print(f"Zero-return baseline MSE: {baseline_mse:.8f}")

if __name__ == "__main__":
    main()
