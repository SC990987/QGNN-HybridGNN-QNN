# tests/test_models.py

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qgnn_hybrid.models import (
    HybridGNN_QNN_Basic_Torch,
    HybridGNN_QNN_Improved_Torch,
    HybridGNN_QNN_LegacyCRY_Torch,
    HybridGNN_MLP,
    JetGNN,
    ParticleNet,
)


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


def make_toy_batch(in_channels=4):
    graphs = [
        make_toy_graph(label=0, in_channels=in_channels),
        make_toy_graph(label=1, in_channels=in_channels),
    ]
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    return next(iter(loader))


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (
            HybridGNN_QNN_Basic_Torch,
            {"hidden_dim": 16, "n_qubits": 8, "q_layers": 1},
        ),
        (
            HybridGNN_QNN_Improved_Torch,
            {"hidden_dim": 16, "n_qubits": 8, "q_layers": 1},
        ),
        (
            HybridGNN_QNN_LegacyCRY_Torch,
            {"hidden_dim": 16, "n_qubits": 8, "q_layers": 1},
        ),
        (
            HybridGNN_MLP,
            {"hidden_dim": 16, "n_qubits": 8, "q_layers": 1},
        ),
        (
            JetGNN,
            {"hidden_dim": 16},
        ),
        (
            ParticleNet,
            {"hidden_channels": 16},
        ),
    ],
)
def test_model_forward_pass(model_cls, kwargs):
    in_channels = 4
    batch = make_toy_batch(in_channels=in_channels)

    model = model_cls(in_channels=in_channels, **kwargs)
    model.eval()

    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()