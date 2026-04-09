# models.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, EdgeConv, global_mean_pool

# 🔹 Import QNN circuits
from qnn import qnn_circuit_basic, qnn_circuit_improved


# =========================================================
# Hybrid GNN + QNN (Basic)
# =========================================================
class HybridGNN_QNN_basic(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()

        self.n_qubits = n_qubits

        # --- GNN Backbone ---
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin_proj = torch.nn.Linear(hidden_dim, n_qubits)

        # --- QNN Parameters ---
        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits)
        )

        # --- Final Classifier ---
        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        # GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        # Project to quantum space
        x = self.lin_proj(x)
        x = torch.tanh(x) * torch.pi

        # --- QNN ---
        device = x.device
        q_out = qnn_circuit_basic(x, self.q_weights)

        # Convert tuple -> tensor
        x = torch.stack(q_out, dim=1).to(device).float()

        return self.fc(x)


# =========================================================
# ParticleNet (EdgeConv-based GNN)
# =========================================================
class ParticleNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.conv1 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv2 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))

        self.conv3 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))

        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# =========================================================
# Hybrid GNN + QNN (Improved)
# =========================================================
class HybridGNN_QNN_improved(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=4):
        super().__init__()
        self.n_qubits = n_qubits

        ## GNN Backbone
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin_proj = torch.nn.Linear(hidden_dim, n_qubits)

        ## QNN Params
        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits, 3)
        )

        ## Final Classifier
        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin_proj(x)
        x = torch.tanh(x) * torch.pi

        # x is now shape (batch_size, n_qubits)
        device = x.device

        # --- VECTORIZED QNN CALL ---
        # No more list comprehension! Pass the whole batch.
        q_out = qnn_circuit_improved(x, self.q_weights) 

        # q_out is a tuple of length 8. Each element has shape (batch_size,)
        # Stack along dim=1 to reconstruct the (batch_size, 8) tensor
        x = torch.stack(q_out, dim=1).to(device).float()

        return self.fc(x)
    
# =========================================================
# Hybrid GNN + MLP
# =========================================================
class HybridGNN_MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, n_qubits=8, q_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        ## GNN Backbone
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin_proj = torch.nn.Linear(hidden_dim, n_qubits)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, n_qubits),
            torch.nn.ReLU(),
            torch.nn.Linear(n_qubits, n_qubits)
        )
        ## QNN Params
        self.q_weights = torch.nn.Parameter(
            0.01 * torch.randn(q_layers, n_qubits)
        )

        ## Final Classifier
        self.fc = torch.nn.Linear(n_qubits, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin_proj(x)
        x = torch.tanh(x) * torch.pi

        # # x is now shape (batch_size, n_qubits)
        # device = x.device

        # # --- VECTORIZED QNN CALL ---
        # # No more list comprehension! Pass the whole batch.
        # q_out = qnn_circuit(x, self.q_weights) 

        # # q_out is a tuple of length 8. Each element has shape (batch_size,)
        # # Stack along dim=1 to reconstruct the (batch_size, 8) tensor
        # x = torch.stack(q_out, dim=1).to(device).float()
        x = self.mlp(x)
        return self.fc(x)
    


# =========================================================
# Graph-Sage
# =========================================================   
class JetGNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 64)
        self.lin = torch.nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

