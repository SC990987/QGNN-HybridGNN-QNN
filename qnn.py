# qnn.py

import pennylane as qml

# Global config (single source of truth)
N_QUBITS = 8

# Device
dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def qnn_circuit_basic(x, weights):
    """
    x: shape (batch_size, n_qubits)
    weights: shape (n_layers, n_qubits)
    """

    # Data encoding (batched)
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='Y')

    # Variational layers
    for l in range(weights.shape[0]):

        # Entanglement ring
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

        # Trainable rotations
        for i in range(N_QUBITS):
            qml.RY(weights[l, i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def qnn_circuit_improved(x, weights):
    """
    x: shape (batch_size, n_qubits)
    weights: shape (n_layers, n_qubits, 3)
    """

    n_layers = weights.shape[0]

    for l in range(n_layers):

        # Data re-uploading
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='Y')

        # Expressive rotations
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        # Entanglement (ring)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

        # Longer-range entanglement
        for i in range(0, N_QUBITS, 2):
            qml.CNOT(wires=[i, (i + 2) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]