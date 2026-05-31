# tests/test_training_smoke.py

import json
import math

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qgnn_hybrid.models import HybridGNN_MLP
from qgnn_hybrid.utils import train_model, evaluate_model


def make_toy_graph(label=0, num_nodes=5, in_channels=4):
    x = torch.randn(num_nodes, in_channels)

    # Simple directed ring graph.
    src = torch.arange(num_nodes)
    dst = torch.roll(src, shifts=-1)

    edge_index = torch.stack(
        [
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ],
        dim=0,
    )

    y = torch.tensor(label, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def make_loader(num_graphs=4, batch_size=2, in_channels=4):
    graphs = [
        make_toy_graph(label=i % 2, in_channels=in_channels)
        for i in range(num_graphs)
    ]
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


def test_train_and_evaluate_smoke(tmp_path):
    in_channels = 4

    train_loader = make_loader(num_graphs=4, batch_size=2, in_channels=in_channels)
    val_loader = make_loader(num_graphs=4, batch_size=2, in_channels=in_channels)
    test_loader = make_loader(num_graphs=4, batch_size=2, in_channels=in_channels)

    model = HybridGNN_MLP(
        in_channels=in_channels,
        hidden_dim=8,
        n_qubits=8,
        q_layers=1,
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-3,
        patience=1,
        save_every=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        model_name=tmp_path / "best_model",
        training_history=tmp_path / "training_history",
    )

    acc, auc = evaluate_model(model, test_loader)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 1

    assert 0.0 <= acc <= 1.0
    assert math.isnan(auc) or 0.0 <= auc <= 1.0

    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "training_history.json").exists()

    with open(tmp_path / "training_history.json", "r") as f:
        saved_history = json.load(f)

    assert "train_loss" in saved_history