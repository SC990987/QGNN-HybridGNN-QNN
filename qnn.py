# qnn.py

import pennylane as qml
import torch 
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

##### Adding Custom QC Simulation code for GPUs
def _bc(val, ndim):
    if val.dim() == 0:
        return val
    return val.reshape(-1, *([1] * ndim))


def apply_ry(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([c * re0 - s * re1, s * re0 + c * re1], dim=-1)
    new_im = torch.stack([c * im0 - s * im1, s * im0 + c * im1], dim=-1)
    return new_re.movedim(-1, qubit + 1), new_im.movedim(-1, qubit + 1)


def apply_rz(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    ct = _bc(torch.cos(theta / 2), nq - 1)
    st = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = ct * re0 + st * im0
    new_im0 = ct * im0 - st * re0
    new_re1 = ct * re1 - st * im1
    new_im1 = ct * im1 + st * re1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_rx(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = c * re0 + s * im1
    new_im0 = c * im0 - s * re1
    new_re1 = s * im0 + c * re1
    new_im1 = -s * re0 + c * im1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_izz(re, im, theta, q0, q1, nq):
    z = torch.tensor([1.0, -1.0], dtype=re.dtype, device=re.device)
    s0 = [1] * (nq + 1); s0[q0 + 1] = 2
    s1 = [1] * (nq + 1); s1[q1 + 1] = 2
    zz = z.reshape(s0) * z.reshape(s1)
    angle = _bc(theta / 2, nq) * zz
    ct = torch.cos(angle)
    st = torch.sin(angle)
    new_re = ct * re + st * im
    new_im = ct * im - st * re
    return new_re, new_im


def apply_cry(re, im, theta, control, target, nq):
    re = re.movedim(control + 1, -2)
    im = im.movedim(control + 1, -2)
    t_ax = (target + 1) if target < control else target
    re = re.movedim(t_ax, -1)
    im = im.movedim(t_ax, -1)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_c0, re_c1 = re[..., 0, :], re[..., 1, :]
    im_c0, im_c1 = im[..., 0, :], im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_c1 = torch.stack([c * re_t0 - s * re_t1, s * re_t0 + c * re_t1], dim=-1)
    new_im_c1 = torch.stack([c * im_t0 - s * im_t1, s * im_t0 + c * im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    out_re = out_re.movedim(-1, t_ax).movedim(-1, control + 1)
    out_im = out_im.movedim(-1, t_ax).movedim(-1, control + 1)
    return out_re, out_im

def apply_cnot(re, im, control, target, nq):
    ## Move Control qubit to second to last axis
    re = re.movedim(control + 1, -2)
    im = im.movedim(control + 1, -2)

    ## Adjust target index after move
    t_ax = (target + 1) if target < control else target

    ## Move Target to last axis
    re = re.movedim(t_ax, -1)
    im = im.movedim(t_ax, -1)

    ## Split control states
    re_c0, re_c1 = re[..., 0, :], re[..., 1, :]
    im_c0, im_c1 = im[..., 0, :], im[..., 1, :]

    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]

    ## Flip target when control = 1
    new_re_c1 = torch.stack([re_t1, re_t0], dim=-1)
    new_im_c1 = torch.stack([im_t1, im_t0], dim=-1)

    ## Recombine
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)

    ## Move axes back
    out_re = out_re.movedim(-1, t_ax).movedim(-1, control + 1)
    out_im = out_im.movedim(-1, t_ax).movedim(-1, control + 1)
    return out_re, out_im


def _amp_to_state_real(amp, nq):
    B = amp.shape[0]
    norm = amp.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normed = (amp / norm).reshape(B, *([2] * nq))
    return normed, torch.zeros_like(normed)


def _state_to_probs_real(re, im):
    B = re.shape[0]
    return (re * re + im * im).reshape(B, -1)


def _renormalize_real(re, im, nq):
    B = re.shape[0]
    norm_sq = (re * re + im * im).reshape(B, -1).sum(dim=1)
    norm = torch.sqrt(norm_sq.clamp(min=1e-24)).reshape(-1, *([1] * nq))
    return re / norm, im / norm


def qnn_torch(x, weights, nq):
    """
    x: (batch_size, nq)
    weights: (n_layers, nq)
    """

    B = x.shape[0]

    # Start in |0...0>
    re = torch.zeros(B, *([2]*nq), device=x.device)
    im = torch.zeros_like(re)
    re[:, (0,) * nq] = 1.0  # |000...0>

    # --- Encoding ---
    for q in range(nq):
        re, im = apply_ry(re, im, x[:, q], q, nq)

    # --- Variational layers ---
    for l in range(weights.shape[0]):

        # Entanglement ring (use CRY as proxy for CNOT-like behavior)
        for q in range(nq):
            re, im = apply_cry(re, im, torch.tensor(torch.pi/2, device=re.device), q, (q+1)%nq, nq)

        # Trainable rotations
        for q in range(nq):
            re, im = apply_ry(re, im, weights[l, q], q, nq)

    # --- Measurement ---
    probs = (re**2 + im**2).reshape(B, -1)

    # Expectation Z per qubit
    outputs = []
    for q in range(nq):
        axis = list(range(1, nq+1))
        axis.remove(q+1)

        p0 = probs.view(B, *([2]*nq)).sum(dim=axis)
        z = p0[:, 0] - p0[:, 1]
        outputs.append(z)

    return torch.stack(outputs, dim=1)