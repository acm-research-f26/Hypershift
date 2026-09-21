"""Correctness contract for the Euclidean GCN."""

from __future__ import annotations

import torch
from torch import nn

from src.gnn import GNN


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [0.2, 1.0, -0.4],
            [1.2, -0.3, 0.7],
            [-0.8, 0.5, 1.1],
            [0.4, -1.0, 0.3],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def test_gnn_is_module_and_respects_embedding_and_output_shapes() -> None:
    torch.manual_seed(7)
    model = GNN(in_channels=3, hidden_channels=5, out_channels=1, dropout=0.0)
    assert isinstance(model, nn.Module)
    x, edge_index = _graph()

    embeddings = model.encode(x, edge_index)
    predictions = model(x, edge_index)

    assert embeddings.shape == (4, 5)
    assert predictions.shape == (4, 1)
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(predictions).all()


def test_gradients_reach_every_trainable_parameter() -> None:
    torch.manual_seed(11)
    model = GNN(in_channels=3, hidden_channels=6, out_channels=1, dropout=0.0)
    x, edge_index = _graph()

    model(x, edge_index).square().mean().backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
        if gradient is not None
    )


def test_unit_edge_weights_match_the_unweighted_graph() -> None:
    torch.manual_seed(13)
    model = GNN(in_channels=3, hidden_channels=5, out_channels=1, dropout=0.0)
    model.eval()
    x, edge_index = _graph()
    unit_weights = torch.ones(edge_index.shape[1])

    unweighted = model(x, edge_index)
    weighted = model(x, edge_index, edge_weight=unit_weights)

    torch.testing.assert_close(weighted, unweighted)


def test_node_permutation_only_permutes_predictions() -> None:
    """A GNN must not depend on the arbitrary numeric names of its nodes."""

    torch.manual_seed(17)
    model = GNN(in_channels=3, hidden_channels=7, out_channels=1, dropout=0.0)
    model.eval()
    x, edge_index = _graph()
    permutation = torch.tensor([2, 0, 3, 1])
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(len(permutation))

    original = model(x, edge_index)
    permuted = model(x[permutation], old_to_new[edge_index])

    torch.testing.assert_close(permuted, original[permutation], rtol=1e-5, atol=1e-6)


def test_reset_parameters_reinitializes_learned_weights() -> None:
    model = GNN(in_channels=3, hidden_channels=5, out_channels=1, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    model.reset_parameters()

    assert any(torch.count_nonzero(parameter).item() > 0 for parameter in model.parameters())
