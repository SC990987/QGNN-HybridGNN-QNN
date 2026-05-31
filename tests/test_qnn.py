# tests/test_qnn.py

import torch

from qgnn_hybrid.qnn import (
    qnn_torch_basic,
    qnn_torch_improved,
    qnn_torch_legacy_cry,
)


def test_qnn_torch_basic_forward_shape():
    batch_size = 3
    n_qubits = 8
    q_layers = 2

    x = torch.randn(batch_size, n_qubits)
    weights = torch.randn(q_layers, n_qubits)

    out = qnn_torch_basic(x, weights, nq=n_qubits)

    assert out.shape == (batch_size, n_qubits)
    assert torch.isfinite(out).all()


def test_qnn_torch_improved_forward_shape():
    batch_size = 3
    n_qubits = 8
    q_layers = 2

    x = torch.randn(batch_size, n_qubits)
    weights = torch.randn(q_layers, n_qubits, 3)

    out = qnn_torch_improved(x, weights, nq=n_qubits)

    assert out.shape == (batch_size, n_qubits)
    assert torch.isfinite(out).all()


def test_qnn_torch_legacy_cry_forward_shape():
    batch_size = 3
    n_qubits = 8
    q_layers = 2

    x = torch.randn(batch_size, n_qubits)
    weights = torch.randn(q_layers, n_qubits)

    out = qnn_torch_legacy_cry(x, weights, nq=n_qubits)

    assert out.shape == (batch_size, n_qubits)
    assert torch.isfinite(out).all()