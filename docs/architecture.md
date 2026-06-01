# Architecture

## Purpose

This repository implements and benchmarks hybrid graph neural network / quantum neural network models for quark/gluon jet classification. The project is organized around a central question:

> Given a graph representation of a jet, does a quantum circuit head provide useful classification power beyond matched classical heads?

The current architecture separates the problem into three layers:

1. **Physics-informed graph construction**
2. **Graph-level representation learning with a GNN encoder**
3. **Classifier heads: linear, MLP, or QNN**

This separation makes it possible to distinguish between performance coming from the graph encoder and performance coming from the final classifier head.

---

## Data representation

The input dataset consists of simulated quark and gluon jets. Each jet is represented as a variable-length list of particles/constituents. Each particle is converted into a graph node.

### Node features

The graph node features are built from particle-level kinematic quantities:

- transverse momentum, `pT`
- pseudorapidity, `eta`
- azimuthal angle, `phi`

The preprocessing step:

1. removes padded particles,
2. applies a log transform to `pT`,
3. centers `eta` and `phi` within each jet,
4. wraps `phi` back into `[-pi, pi]`,
5. standardizes `log(pT)` and `eta`,
6. represents `phi` using `sin(phi)` and `cos(phi)`.

The final node feature vector is:

```text
[standardized_log_pT, standardized_eta, sin(phi), cos(phi)]
```

This avoids the discontinuity at `phi = +/- pi` and gives the graph model a smooth angular representation.

---

## Graph construction

Each jet is converted into a k-nearest-neighbor graph in detector-coordinate space.

For each particle, nearest neighbors are selected using angular distance in the `eta-phi` plane:

```text
Delta R = sqrt((Delta eta)^2 + (Delta phi)^2)
```

where `Delta phi` is wrapped into `[-pi, pi]`.

The default graph-building choice is:

```text
k = 16
min_particles = 16
```

This means jets with fewer than 16 non-padded particles are skipped. This preserves the original benchmark behavior used during early experiments.

### Important cache detail

The processed graph cache filename includes both `k` and `min_particles`:

```text
data/processed/qg_jets_k16_min16.pt
```

This is important because changing the filtering rule changes the graph dataset. A cache built with `min_particles=2` should not be reused for a benchmark using `min_particles=16`.

---

## Model groups

The repository contains two kinds of models:

1. **Matched-head ablations**
2. **Broader classical baselines**

These should be interpreted separately.

---

## Shared encoder for matched-head models

The matched-head models use the same `GraphSAGEEncoder`:

```text
SAGEConv
→ ReLU
→ SAGEConv
→ ReLU
→ global_mean_pool
→ Linear(hidden_dim -> n_qubits)
→ tanh(x) * pi
```

The output is an `n_qubits`-dimensional graph representation. In the default benchmark:

```text
hidden_dim = 64
n_qubits = 8
q_layers = 4
```

This shared encoder is used by:

- `HybridGNN_LinearHead`
- `HybridGNN_MLP`
- `HybridGNN_QNN_Basic_Torch`
- `HybridGNN_QNN_Improved_Torch`
- `HybridGNN_QNN_LegacyCRY_Torch`
- PennyLane reference QNN models

This makes these models the fairest comparison for studying the classifier head.

---

## Matched bottleneck ablation: linear head

`HybridGNN_LinearHead` uses the shared GraphSAGE encoder and the same 8-dimensional bottleneck, then applies only a linear classifier:

```text
GraphSAGEEncoder -> Linear(n_qubits -> 2)
```

This is the most important ablation for answering:

> Does the MLP or QNN head add useful modeling capacity beyond the shared graph encoder and bottleneck?

If the linear-head model performs the same as the MLP or QNN models, then most of the classification power is coming from the encoder rather than the head.

---

## Matched classical head: GNN-MLP

`HybridGNN_MLP` uses the same shared encoder and bottleneck, followed by a small classical MLP head:

```text
GraphSAGEEncoder
→ Linear(n_qubits -> n_qubits)
→ ReLU
→ Linear(n_qubits -> n_qubits)
→ Linear(n_qubits -> 2)
```

This is the main matched classical baseline for the QNN models.

---

## QNN heads

The QNN models use the same graph encoder and bottleneck as the matched linear and MLP heads. The bottleneck output is interpreted as rotation angles for the quantum circuit.

### Basic QNN

The basic QNN uses:

```text
RY angle encoding
→ ring CNOT entanglement
→ trainable RY rotations
→ Pauli-Z expectation values
→ Linear(n_qubits -> 2)
```

This is the simplest QNN head and has trainable weights of shape:

```text
(q_layers, n_qubits)
```

### Improved QNN

The improved QNN uses a more expressive circuit with trainable rotations in multiple axes and additional entanglement structure. Its weights have shape:

```text
(q_layers, n_qubits, 3)
```

Because this has more QNN parameters than the basic circuit, it should be described as an architectural comparison rather than a parameter-matched comparison.

### Legacy CRY QNN

The legacy CRY model is retained for reproducibility with earlier experiments. It uses a controlled-RY entanglement pattern and should not be treated as the exact fast equivalent of the CNOT-based PennyLane reference circuit.

---

## PennyLane and fast PyTorch backends

The repository includes two QNN backends:

1. **PennyLane reference circuits**
2. **Fast PyTorch simulator circuits**

The PennyLane models are reference implementations. The fast PyTorch models are the practical training/benchmarking implementations.

The fast simulator implementation is based on Eric A. F. Reinhardt's `fastQML` work and is used here with permission.

Optional `torch.compile` experiments are being tested on a separate branch. The main branch currently uses eager PyTorch execution for stability, especially on Apple Silicon/MPS.

---

## Broader classical baselines

### JetGNN / GraphSAGE baseline

`JetGNN` is a simple GraphSAGE baseline:

```text
SAGEConv
→ ReLU
→ SAGEConv
→ ReLU
→ global_mean_pool
→ Linear(hidden_dim -> 2)
```

This is useful as a simple classical GNN baseline. It is not a strict matched-head ablation because it does not use the shared 8-dimensional bottleneck.

### ParticleNet-style EdgeConv baseline

`ParticleNet` is a stronger classical graph baseline using stacked EdgeConv layers:

```text
EdgeConv
→ EdgeConv
→ EdgeConv
→ global_mean_pool
→ MLP classifier
```

This model is included to contextualize hybrid-model performance against a more expressive classical particle-cloud architecture. It is not a matched ablation of the hybrid QNN architecture.

---

## Training pipeline

Training is config-driven through YAML files in `configs/`.

A typical command is:

```bash
python3 -m qgnn_hybrid.train --config configs/gnn_mlp.yaml --seed 42
```

The training script:

1. loads the dataset,
2. builds or loads cached processed graphs,
3. splits the data into train/validation/test sets,
4. builds the selected model,
5. trains with early stopping based on validation loss,
6. saves the best model checkpoint,
7. evaluates on the test set,
8. writes final metrics to JSON.

Outputs are written to model-specific directories under:

```text
outputs/
```

---

## Reproducibility notes

Important reproducibility controls:

- fixed random seed,
- deterministic processed-cache filenames,
- explicit graph-construction settings,
- YAML config files,
- saved training history,
- saved final metrics,
- pytest smoke tests,
- separate stable and experimental branches.

For final reported results, use multi-seed averages rather than a single seed.

Recommended seeds:

```text
42, 43, 44, 45, 46
```

