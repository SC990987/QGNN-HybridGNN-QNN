# models.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, EdgeConv, global_mean_pool

from .qnn import (
    qnn_circuit_basic,
    qnn_circuit_improved,
    qnn_torch_basic,
    qnn_torch_improved,
    qnn_torch_legacy_cry,
)


# =========================================================
# Shared GraphSAGE Encoder
# =========================================================
class GraphSAGEEncoder(torch.nn.Module):
    """
    Shared GraphSAGE encoder used by the hybrid GNN-QNN and GNN-MLP models.

    The encoder maps a particle graph to a fixed-dimensional graph-level
    representation, then projects it to the quantum feature dimension.
    """

    def __init__(self, in_channels, hidden_dim=64, out_dim=8):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin_proj = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin_proj(x)

        # Map latent features to a rotation-angle-like range.
        return torch.tanh(x) * torch.pi


# =========================================================
# Hybrid GNN + QNN: Basic PennyLane Reference
# =========================================================
class HybridGNN_QNN_Basic_PennyLane(torch.nn.Module):
    """
    Hybrid GNN-QNN model using the PennyLane basic reference circuit.

    Circuit:
    - RY angle encoding
    - Ring CNOT entanglement
    - Trainable RY rotations
    - Pauli-Z expectation values
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits)
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)

        q_out = qnn_circuit_basic(x, self.q_weights)

        # PennyLane returns a list/tuple of length n_qubits, each with shape (batch_size,).
        x = torch.stack(q_out, dim=1).to(x.device).float()

        return self.fc(x)


# =========================================================
# Hybrid GNN + QNN: Improved PennyLane Reference
# =========================================================
class HybridGNN_QNN_Improved_PennyLane(torch.nn.Module):
    """
    Hybrid GNN-QNN model using the PennyLane improved reference circuit.

    Circuit:
    - Data re-uploading
    - Trainable RX, RY, RZ rotations
    - Ring CNOT entanglement
    - Longer-range CNOT entanglement
    - Pauli-Z expectation values
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits, 3)
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)

        q_out = qnn_circuit_improved(x, self.q_weights)

        # PennyLane returns a list/tuple of length n_qubits, each with shape (batch_size,).
        x = torch.stack(q_out, dim=1).to(x.device).float()

        return self.fc(x)


# =========================================================
# Hybrid GNN + QNN: Basic Fast PyTorch Simulator
# =========================================================
class HybridGNN_QNN_Basic_Torch(torch.nn.Module):
    """
    Hybrid GNN-QNN model using the fast PyTorch basic circuit.

    This is the fast simulator equivalent of HybridGNN_QNN_Basic_PennyLane.
    It uses CNOT entanglement and should be used as the default fast QNN model.
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits)
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        x = qnn_torch_basic(x, self.q_weights, self.n_qubits)
        return self.fc(x)


# =========================================================
# Hybrid GNN + QNN: Improved Fast PyTorch Simulator
# =========================================================
class HybridGNN_QNN_Improved_Torch(torch.nn.Module):
    """
    Hybrid GNN-QNN model using the fast PyTorch improved circuit.

    This is the fast simulator equivalent of HybridGNN_QNN_Improved_PennyLane.
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits, 3)
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        x = qnn_torch_improved(x, self.q_weights, self.n_qubits)
        return self.fc(x)


# =========================================================
# Hybrid GNN + QNN: Legacy CRY Fast PyTorch Simulator
# =========================================================
class HybridGNN_QNN_LegacyCRY_Torch(torch.nn.Module):
    """
    Legacy hybrid GNN-QNN model used in early experiments.

    This model uses the legacy fast PyTorch QNN with CRY(pi/2) entanglement
    instead of CNOT entanglement.

    Important:
    This is not treated as the fast equivalent of the PennyLane CNOT-based
    reference circuit. It is retained for reproducibility and comparison with
    earlier experiments.
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits)
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        x = qnn_torch_legacy_cry(x, self.q_weights, self.n_qubits)
        return self.fc(x)


# =========================================================
# Hybrid GNN + MLP Classical Baseline
# =========================================================
class HybridGNN_MLP(torch.nn.Module):
    """
    Classical baseline using the same GraphSAGE encoder and projection layer,
    followed by a small MLP instead of a QNN.
    """

    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=2):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, n_qubits)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, n_qubits),
            torch.nn.ReLU(),
            torch.nn.Linear(n_qubits, n_qubits),
        )

        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        x = self.mlp(x)
        return self.fc(x)


# =========================================================
# GraphSAGE Baseline
# =========================================================
class JetGNN(torch.nn.Module):
    """
    Simple GraphSAGE baseline for quark/gluon jet classification.
    """

    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# =========================================================
# ParticleNet-style EdgeConv Baseline
# =========================================================
class ParticleNet(torch.nn.Module):
    """
    ParticleNet-style EdgeConv baseline.

    This is a lightweight EdgeConv-based model inspired by ParticleNet-like
    particle-cloud architectures.
    """

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.conv1 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(2 * in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        self.conv2 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        self.conv3 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

__all__ = [
    "GraphSAGEEncoder",
    "HybridGNN_QNN_Basic_PennyLane",
    "HybridGNN_QNN_Improved_PennyLane",
    "HybridGNN_QNN_Basic_Torch",
    "HybridGNN_QNN_Improved_Torch",
    "HybridGNN_QNN_LegacyCRY_Torch",
    "HybridGNN_MLP",
    "JetGNN",
    "ParticleNet",
    "HybridGNN_QNN_basic",
    "HybridGNN_QNN_improved",
    "HybridGNN_QNN_basic_torch",
    "HybridGNN_QNN_improved_torch",
    "HybridGNN_QNN_legacy_torch",
    "HybridGNN_QNN_legacy_cry_torch",
]