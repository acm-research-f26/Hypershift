"""Tests for node-regression training and metric aggregation."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import src.train as train_module
from src.gnn import GNN


EMPTY_EDGES = torch.empty((2, 0), dtype=torch.long)


def _linear_samples(count: int = 20) -> list[Data]:
    generator = torch.Generator().manual_seed(23)
    samples: list[Data] = []
    for _ in range(count):
        x = torch.randn((3, 2), generator=generator)
        y = 0.7 * x[:, :1] - 0.25 * x[:, 1:2]
        samples.append(Data(x=x, edge_index=EMPTY_EDGES.clone(), y=y))
    return samples


def test_train_epoch_returns_finite_loss_and_updates_parameters() -> None:
    torch.manual_seed(29)
    model = GNN(in_channels=2, hidden_channels=8, out_channels=1, dropout=0.0)
    loader = DataLoader(_linear_samples(8), batch_size=4, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_module.train_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
    )

    assert math.isfinite(loss)
    assert loss >= 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


class _FirstFeatureModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        return x[:, :1]


def test_evaluate_aggregates_every_target_not_batch_means() -> None:
    samples = [
        Data(
            x=torch.tensor([[1.0], [-2.0]]),
            edge_index=EMPTY_EDGES.clone(),
            y=torch.tensor([[0.0], [-1.0]]),
        ),
        Data(
            x=torch.tensor([[3.0]]),
            edge_index=EMPTY_EDGES.clone(),
            y=torch.tensor([[1.0]]),
        ),
        Data(
            x=torch.tensor([[-4.0], [5.0], [6.0]]),
            edge_index=EMPTY_EDGES.clone(),
            y=torch.tensor([[-2.0], [-1.0], [8.0]]),
        ),
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False)

    metrics = train_module.evaluate(_FirstFeatureModel(), loader, "cpu")

    errors = torch.tensor([1.0, -1.0, 2.0, -2.0, 6.0, -2.0])
    expected_mse = float(errors.square().mean())
    expected_mae = float(errors.abs().mean())
    assert metrics["mse"] == pytest.approx(expected_mse)
    assert metrics["mae"] == pytest.approx(expected_mae)
    assert metrics["directional_accuracy"] == pytest.approx(5 / 6)
    assert all(parameter.grad is None for parameter in _FirstFeatureModel().parameters())


def test_train_restores_the_best_validation_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class ScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del edge_index, edge_weight
            return x[:, :1] * self.value

    validation_values = iter([3.0, 1.0, 2.0, 4.0])

    def fake_train_epoch(*args: object, **kwargs: object) -> float:
        model = args[0]
        assert isinstance(model, ScalarModel)
        with torch.no_grad():
            model.value.add_(1.0)
        return float(model.value.item())

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, float]:
        mse = next(validation_values)
        return {"mse": mse, "mae": mse, "directional_accuracy": 0.0}

    monkeypatch.setattr(train_module, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    model = ScalarModel()

    history = train_module.train(
        model,
        object(),
        object(),
        epochs=10,
        patience=2,
        device="cpu",
    )

    assert history["best_epoch"] == 2
    assert history["epochs_ran"] == 4
    assert model.value.item() == pytest.approx(2.0)


def test_end_to_end_training_reduces_validation_mse() -> None:
    torch.manual_seed(31)
    samples = _linear_samples(24)
    train_loader = DataLoader(samples[:18], batch_size=6, shuffle=True)
    validation_loader = DataLoader(samples[18:], batch_size=3, shuffle=False)
    model = GNN(in_channels=2, hidden_channels=10, out_channels=1, dropout=0.0)
    before = train_module.evaluate(model, validation_loader, "cpu")["mse"]

    history = train_module.train(
        model,
        train_loader,
        validation_loader,
        epochs=60,
        patience=15,
        learning_rate=0.02,
        weight_decay=0.0,
        device="cpu",
    )
    after = train_module.evaluate(model, validation_loader, "cpu")["mse"]

    assert history["best_epoch"] >= 1
    assert after < before
