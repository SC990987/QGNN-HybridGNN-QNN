# Hybrid GNN–QNN for Quark/Gluon Jet Classification

A reproducible machine learning research project exploring hybrid quantum-classical graph neural networks for quark/gluon jet classification in high energy physics.

This repository implements and benchmarks classical Graph Neural Network models and hybrid Graph Neural Network–Quantum Neural Network models for particle jet tagging. Each jet is represented as a graph, where particles are nodes and edges encode angular relationships between particles. A classical GNN learns a graph-level jet representation, which is then processed by either a classical MLP head or a variational quantum circuit.

## Project Overview

Quark/gluon discrimination is an important jet-tagging task in high energy physics. When quarks and gluons hadronize, they produce collimated sprays of particles known as jets. Identifying whether a jet originated from a quark or a gluon can improve signal-background separation in collider analyses.

Graph neural networks are well suited to this task because jets naturally have relational structure. Particle-level constituents can be treated as nodes in a graph, while edges connect nearby particles in angular space. This allows the model to learn local and global jet substructure through message passing.

This project investigates whether a quantum neural network layer can be integrated into a graph-based jet-classification pipeline and achieve competitive performance with classical baselines.

## Goals

The main goals of this project are:

* Build a reproducible quark/gluon jet classification pipeline.
* Represent particle jets as graphs using particle-level kinematic features.
* Train classical GNN baselines for comparison.
* Implement hybrid GNN-QNN models using PyTorch, PyTorch Geometric, and PennyLane.
* Add a faster PyTorch-based QNN simulator backend for training experiments.
* Compare classical and hybrid models using accuracy and ROC-AUC.
* Document model design, training workflow, limitations, and future directions.

## Repository Structure

```text
QGNN-HybridGNN-QNN/
│
├── configs/                 # YAML configuration files for training runs
├── data/                    # Local data directory; raw/processed data are not committed
├── docs/                    # Additional project documentation
├── notebooks/               # Summary and visualization notebooks
├── results/                 # Final metrics, plots, and result summaries
├── scripts/                 # Optional helper scripts
├── src/
│   └── qgnn_hybrid/          # Main Python package
│       ├── __init__.py
│       ├── data.py           # Dataset loading, downloading, graph construction, caching
│       ├── models.py         # Classical and hybrid GNN architectures
│       ├── qnn.py            # PennyLane QNNs and fast PyTorch QNN wrappers
│       ├── qnn_fast.py       # Fast PyTorch quantum circuit simulation utilities
│       ├── train.py          # Command-line training entry point
│       └── utils.py          # Preprocessing, graph utilities, training, evaluation
├── tests/                   # Unit and smoke tests
├── pyproject.toml           # Package configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Dataset

This project uses the quark/gluon jet dataset introduced in:

> P. T. Komiske, E. M. Metodiev, and J. Thaler, “Energy Flow Networks: Deep Sets for Particle Jets,” Journal of High Energy Physics, 2019.

Each jet contains particle-level information such as transverse momentum, rapidity, azimuthal angle, and particle identity. In this project, jets are converted into graph objects for use with PyTorch Geometric.

Raw datasets are not committed to this repository.

Expected local structure:

```text
data/
├── raw/
│   └── QG_jets.npz
└── processed/
    └── qg_jets_k16.pt
```

The dataset is downloaded automatically by `qgnn_hybrid.data.download_dataset()` if it is not already present. Processed graph objects are cached under `data/processed/` so repeated training runs do not need to rebuild the graph dataset from scratch.

## Graph Construction

Each jet is represented as a graph:

* Nodes represent particles.
* Node features are derived from particle-level kinematics.
* Edges connect nearby particles in angular space using k-nearest neighbors.
* Edge features encode angular separation information.
* The graph is passed into a GNN to learn a jet-level representation.

Current graph construction details:

```text
k-nearest neighbors: 16
node features: standardized log(pt), standardized eta, sin(phi), cos(phi)
edge features: delta_eta, delta_phi, delta_R
default split: 70% train, 15% validation, 15% test
processed graph cache: data/processed/qg_jets_k16.pt
```

## Models

This repository includes classical GNN baselines, hybrid GNN-QNN models, and fast PyTorch QNN simulator variants.

### Classical Baselines

Implemented classical models:

* `graphsage` — simple GraphSAGE baseline.
* `particlenet` — lightweight ParticleNet-style EdgeConv baseline.
* `gnn_mlp` — GraphSAGE encoder followed by a classical MLP head.

These baselines establish the classical performance level needed to evaluate whether the quantum component provides useful representational power.

### Hybrid GNN-QNN Models

The hybrid models use a classical GraphSAGE encoder followed by a quantum neural network head.

The general pipeline is:

```text
Particle jet
   ↓
Graph construction
   ↓
GraphSAGE message passing
   ↓
Global mean pooling
   ↓
Linear projection to quantum feature space
   ↓
Quantum neural network
   ↓
Pauli-Z expectation values
   ↓
Final classifier
   ↓
Quark/gluon prediction
```

Implemented hybrid models:

* `qnn_basic_pennylane` — PennyLane reference implementation with RY encoding, ring CNOT entanglement, and trainable RY rotations.
* `qnn_improved_pennylane` — PennyLane reference implementation with data re-uploading, RX/RY/RZ rotations, ring CNOTs, and longer-range CNOTs.
* `qnn_basic_torch` — fast PyTorch simulator equivalent of the basic PennyLane CNOT-based circuit.
* `qnn_improved_torch` — fast PyTorch simulator equivalent of the improved PennyLane CNOT-based circuit.
* `qnn_legacy_cry_torch` — legacy fast PyTorch QNN using CRY(pi/2) entanglement, retained for reproducibility of early experiments.

### Quantum Circuit Backends

This repository contains two QNN backends:

1. **PennyLane reference circuits** in `qnn.py`
2. **Fast PyTorch simulator circuits** using functions from `qnn_fast.py`

The PennyLane circuits are kept as clear reference implementations. The fast PyTorch simulator is used for faster differentiable QNN training experiments.

The default fast QNN models use CNOT entanglement to match the PennyLane reference circuits. The legacy CRY model is kept separately and is not treated as equivalent to the CNOT-based reference circuit.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/QGNN-HybridGNN-QNN.git
cd QGNN-HybridGNN-QNN
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `yaml` is missing, install PyYAML:

```bash
python -m pip install pyyaml
```

PyTorch Geometric installation may depend on your local PyTorch, CUDA, or Apple Silicon configuration. See the official PyTorch Geometric installation guide if needed.

## Usage

Run commands from the repository root:

```bash
cd QGNN-HybridGNN-QNN
```

### Debug run

Use this to test that the pipeline works before launching a full training run:

```bash
python -m qgnn_hybrid.train --config configs/debug_gnn_mlp.yaml
```

### Train classical baselines

```bash
python -m qgnn_hybrid.train --config configs/graphsage.yaml
python -m qgnn_hybrid.train --config configs/particlenet.yaml
python -m qgnn_hybrid.train --config configs/gnn_mlp.yaml
```

### Train hybrid QNN models

```bash
python -m qgnn_hybrid.train --config configs/qnn_basic_torch.yaml
python -m qgnn_hybrid.train --config configs/qnn_improved_torch.yaml
python -m qgnn_hybrid.train --config configs/qnn_legacy_cry_torch.yaml
```

### Override config values from the command line

```bash
python -m qgnn_hybrid.train --config configs/qnn_basic_torch.yaml --seed 43
python -m qgnn_hybrid.train --config configs/qnn_basic_torch.yaml --epochs 10 --batch-size 16
```

### Available model names

```text
graphsage
particlenet
gnn_mlp
qnn_basic_torch
qnn_improved_torch
qnn_legacy_cry_torch
qnn_basic_pennylane
qnn_improved_pennylane
```

## Configuration Files

Training runs are controlled with YAML files in `configs/`.

Example:

```yaml
model: qnn_basic_torch

batch_size: 32
epochs: 50
patience: 5
seed: 42
lr: 0.001

hidden_dim: 64
n_qubits: 8
q_layers: 4

output_dir: outputs/qnn_basic_torch
save_every: 1
save_history_every_epoch: false
```

The command-line arguments override values from the YAML config.

## Outputs

Training outputs are saved under the configured `output_dir`, for example:

```text
outputs/qnn_basic_torch/
├── qnn_basic_torch_seed_42_best_model.pt
├── qnn_basic_torch_seed_42_checkpoint.pt
├── qnn_basic_torch_seed_42_training_history.json
└── qnn_basic_torch_seed_42_metrics.json
```

The `outputs/` directory is ignored by Git because it contains generated checkpoints and experiment artifacts.

## Results

Current results are preliminary and will be updated as the cleaned multi-seed evaluation pipeline is finalized.

| Model                      | Test AUC | Notes                                   |
| -------------------------- | -------: | --------------------------------------- |
| GraphSAGE                  |   0.8515 | Classical GNN baseline                  |
| ParticleNet-style EdgeConv |   0.8596 | Stronger classical graph baseline       |
| Hybrid GNN-QNN             |   0.8604 | GNN backbone with quantum circuit layer |
| GNN-MLP                    |   0.8633 | Best current classical baseline         |

A final results table with mean and standard deviation across multiple random seeds is coming soon.

Planned final result files:

```text
results/
├── seed_results.csv
├── metrics_summary.csv
└── figures/
    ├── auc_comparison.png
    ├── roc_curves.png
    └── training_curves.png
```

## Interpretation

The hybrid GNN-QNN model achieves competitive performance with classical graph neural network baselines, but the current classical GNN-MLP model remains slightly stronger in terms of test AUC. This project should therefore be interpreted as a reproducible benchmark of hybrid quantum-classical architectures for jet classification, rather than a claim of quantum advantage.

The results suggest that hybrid quantum-classical models can be integrated into graph-based particle physics workflows, but further work is needed to understand when quantum circuits provide practical advantages over well-tuned classical architectures.
## Reproducibility

This project is organized to support reproducible experimentation through:

* Config-driven training
* Fixed random seeds
* Cached graph preprocessing
* Multi-seed evaluation
* Saved metrics summaries
* Version-controlled source code
* Lightweight result files
* Ignored raw data, processed data, checkpoints, and generated outputs
* Pytest smoke tests for imports, QNN forward passes, model forward passes, and a minimal training/evaluation loop

Current reproducibility checklist:

```text
[x] Package-style source layout
[x] Editable install with pyproject.toml
[x] YAML training configs
[x] Fixed random seeds
[x] Cached graph preprocessing
[x] Multiple model variants
[x] Fast PyTorch QNN backend
[x] Pytest smoke tests
[ ] Final multi-seed summary script
[ ] Final results CSV files
[ ] Final plots
```

## Testing

This repository includes a small pytest smoke-test suite to verify that the main package components work after code changes.

Run the tests from the repository root:

```bash
python -m pytest -q
```

The current smoke tests check:

* Package and module imports
* Fast QNN forward passes
* Hybrid and classical model forward passes on synthetic PyTorch Geometric graphs
* A one-epoch synthetic training/evaluation loop
* Checkpoint, best-model, and training-history file creation

Example successful output:

```text
10 passed
```

These tests do not download the full quark/gluon dataset. They use small synthetic graph batches so that the test suite runs quickly and can be used as a lightweight sanity check during development.

Some warnings may appear from PyTorch or PyTorch Geometric, especially on newer Python versions. These warnings are external dependency warnings and do not indicate test failures.


## Documentation

Additional documentation is planned in the `docs/` directory:

```text
docs/
├── architecture.md
├── experiment_notes.md
└── results_interpretation.md
```

These documents will describe graph construction, model architecture, quantum circuit design, fast simulator validation, and interpretation of the results.

## Acknowledgments

This project uses a custom PyTorch-based quantum circuit simulator to accelerate hybrid quantum-classical model training. The original simulator implementation was developed by Eric Reinhardt and is used here with permission.

The simulator provides differentiable implementations of common quantum gates and circuit operations, enabling faster experimentation with quantum machine learning models compared with relying only on the PennyLane reference implementation.

All model integration with the graph neural network pipeline, benchmarking, and quark/gluon jet classification experiments were performed as part of this project.

## Limitations

This project currently has several important limitations:

* Quantum circuit simulation remains computationally expensive compared with fully classical neural network layers.
* The QNN layer operates on a reduced graph-level latent representation rather than directly on the full particle graph.
* The hybrid model has not yet shown clear performance improvement over the best classical baseline.
* Results depend on dataset size, graph construction choices, circuit depth, random seed, and quantum circuit design.
* The current implementation is designed for research benchmarking, not production deployment.

## Future Work

Future improvements may include:

* Multi-seed evaluation with mean and standard deviation
* Hyperparameter scans over the number of qubits and circuit depth
* Alternative quantum feature maps
* Alternative entanglement patterns
* Runtime and parameter-efficiency comparisons
* Validation of fast PyTorch circuits against PennyLane reference circuits
* More expressive GNN backbones
* Comparison against additional published jet-tagging baselines
* Extension to other graph-learning datasets, such as molecular property prediction or materials science benchmarks

## Skills Demonstrated

This project demonstrates experience with:

* PyTorch
* PyTorch Geometric
* PennyLane
* Graph neural networks
* Quantum machine learning
* Variational quantum circuits
* Scientific machine learning
* Particle physics data analysis
* Model benchmarking
* Config-driven training
* Reproducible ML project organization

## References

1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. “Quantum Graph Neural Networks.” arXiv:1909.12264, 2019.

2. Beer, K., Khosla, M., Köhler, J., Osborne, T. J., & Zhao, T. “Quantum machine learning of graph-structured data.” Physical Review A, 108(1), 012410, 2023.

3. Komiske, P. T., Metodiev, E. M., & Thaler, J. “Energy Flow Networks: Deep Sets for Particle Jets.” Journal of High Energy Physics, 2019.

4. Qu, H., & Gouskos, L. “Jet tagging via particle clouds.” Physical Review D, 101(5), 056019, 2020.

## Status

This repository is being cleaned and polished as a professional research portfolio project. Core model development is complete, and the current focus is reproducibility, documentation, tests, final multi-seed result summaries, and clean GitHub presentation.
