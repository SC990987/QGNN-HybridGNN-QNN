# qnn.py

import pennylane as qml
import torch

from .qnn_fast import apply_ry, apply_rx, apply_rz, apply_cnot, apply_cry


N_QUBITS = 8

dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def qnn_circuit_basic(x, weights):
    """
    PennyLane reference circuit.

    x: shape (batch_size, n_qubits)
    weights: shape (n_layers, n_qubits)
    """

    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")

    for layer in range(weights.shape[0]):
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

        for q in range(N_QUBITS):
            qml.RY(weights[layer, q], wires=q)

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def qnn_circuit_improved(x, weights):
    """
    PennyLane reference circuit with data re-uploading.

    x: shape (batch_size, n_qubits)
    weights: shape (n_layers, n_qubits, 3)
    """

    n_layers = weights.shape[0]

    for layer in range(n_layers):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")

        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)

        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

        for q in range(0, N_QUBITS, 2):
            qml.CNOT(wires=[q, (q + 2) % N_QUBITS])

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

def _initial_zero_state(batch_size, nq, device, dtype):
    """Create batched |00...0> state."""
    re = torch.zeros(batch_size, *([2] * nq), device=device, dtype=dtype)
    im = torch.zeros_like(re)
    re[:, (0,) * nq] = 1.0
    return re, im


def _pauli_z_expectations(re, im, nq):
    """Return Pauli-Z expectation values for each qubit."""
    batch_size = re.shape[0]
    probs = (re**2 + im**2).reshape(batch_size, -1)
    probs_view = probs.view(batch_size, *([2] * nq))

    outputs = []
    for q in range(nq):
        dims_to_sum = tuple(dim for dim in range(1, nq + 1) if dim != q + 1)
        marginal = probs_view.sum(dim=dims_to_sum)
        z_expectation = marginal[:, 0] - marginal[:, 1]
        outputs.append(z_expectation)

    return torch.stack(outputs, dim=1)


def qnn_torch_basic(x, weights, nq=N_QUBITS):
    """
    Fast PyTorch equivalent of qnn_circuit_basic.

    Circuit:
    1. RY angle encoding
    2. Ring CNOT entanglement
    3. Trainable RY rotations
    4. Pauli-Z expectation values

    x: shape (batch_size, nq)
    weights: shape (n_layers, nq)
    """
    batch_size = x.shape[0]
    re, im = _initial_zero_state(batch_size, nq, x.device, x.dtype)

    # Equivalent to qml.AngleEmbedding(x, rotation="Y")
    for q in range(nq):
        re, im = apply_ry(re, im, x[:, q], q, nq)

    for layer in range(weights.shape[0]):
        # Equivalent to ring CNOT layer
        for q in range(nq):
            re, im = apply_cnot(re, im, q, (q + 1) % nq, nq)

        # Equivalent to trainable RY rotations
        for q in range(nq):
            re, im = apply_ry(re, im, weights[layer, q], q, nq)

    return _pauli_z_expectations(re, im, nq)


def qnn_torch_improved(x, weights, nq=N_QUBITS):
    """
    Fast PyTorch equivalent of qnn_circuit_improved.

    Circuit per layer:
    1. RY data re-uploading
    2. Trainable RX, RY, RZ rotations
    3. Ring CNOT entanglement
    4. Longer-range CNOT entanglement

    x: shape (batch_size, nq)
    weights: shape (n_layers, nq, 3)
    """
    batch_size = x.shape[0]
    re, im = _initial_zero_state(batch_size, nq, x.device, x.dtype)

    n_layers = weights.shape[0]

    for layer in range(n_layers):
        # Equivalent to qml.AngleEmbedding(x, rotation="Y")
        for q in range(nq):
            re, im = apply_ry(re, im, x[:, q], q, nq)

        # Equivalent to RX, RY, RZ trainable rotations
        for q in range(nq):
            re, im = apply_rx(re, im, weights[layer, q, 0], q, nq)
            re, im = apply_ry(re, im, weights[layer, q, 1], q, nq)
            re, im = apply_rz(re, im, weights[layer, q, 2], q, nq)

        # Equivalent to ring CNOT layer
        for q in range(nq):
            re, im = apply_cnot(re, im, q, (q + 1) % nq, nq)

        # Equivalent to longer-range CNOT layer
        for q in range(0, nq, 2):
            re, im = apply_cnot(re, im, q, (q + 2) % nq, nq)

    return _pauli_z_expectations(re, im, nq)

def qnn_torch_legacy_cry(x, weights, nq=N_QUBITS):
    """
    Legacy fast PyTorch QNN used in early experiments.

    This circuit is similar to qnn_torch_basic, but uses a ring of
    controlled-RY rotations with theta = pi/2 instead of CNOT gates.

    Important:
    This is NOT exactly equivalent to qnn_circuit_basic, because
    qnn_circuit_basic uses CNOT entanglement. This function is kept
    for reproducibility of earlier experiments.

    Circuit:
    1. RY angle encoding
    2. Ring CRY(pi/2) entanglement
    3. Trainable RY rotations
    4. Pauli-Z expectation values

    x: shape (batch_size, nq)
    weights: shape (n_layers, nq)
    """
    batch_size = x.shape[0]
    re, im = _initial_zero_state(batch_size, nq, x.device, x.dtype)

    # RY angle encoding
    for q in range(nq):
        re, im = apply_ry(re, im, x[:, q], q, nq)

    for layer in range(weights.shape[0]):
        # Legacy CRY(pi/2) entanglement ring
        theta = torch.full(
            (),
            torch.pi / 2,
            device=x.device,
            dtype=x.dtype,
        )

        for q in range(nq):
            re, im = apply_cry(re, im, theta, q, (q + 1) % nq, nq)

        # Trainable RY rotations
        for q in range(nq):
            re, im = apply_ry(re, im, weights[layer, q], q, nq)

    return _pauli_z_expectations(re, im, nq)
