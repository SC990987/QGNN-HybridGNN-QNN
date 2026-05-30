# Hybrid GNN–QNN for Quark/Gluon Jet Classification

A reproducible machine learning research project exploring hybrid quantum-classical graph neural networks for quark/gluon jet classification in high energy physics.

This repository implements and benchmarks classical Graph Neural Network models and a hybrid Graph Neural Network–Quantum Neural Network architecture for particle jet tagging. Each jet is represented as a graph, where particles are nodes and edges encode angular relationships between particles. A classical GNN learns a graph-level jet representation, which is then processed by either a classical multilayer perceptron or a variational quantum circuit for final classification.

## Project Overview

Quark/gluon discrimination is an important jet-tagging task in high energy physics. When quarks and gluons hadronize, they produce collimated sprays of particles known as jets. Identifying whether a jet originated from a quark or a gluon can improve signal-background separation in collider analyses.

Graph neural networks are well suited to this task because jets naturally have relational structure. Particle-level constituents can be treated as nodes in a graph, while edges connect nearby particles in angular space. This allows the model to learn local and global jet substructure through message passing.

This project investigates whether a quantum neural network layer can be integrated into a graph-based jet-classification pipeline and achieve competitive performance with classical baselines.

## Goals

The main goals of this project are:

* Build a reproducible quark/gluon jet classification pipeline.
* Represent particle jets as graphs using particle-level kinematic features.
* Train classical GNN baselines for comparison.
* Implement a hybrid GNN-QNN model using PyTorch, PyTorch Geometric, and PennyLane.
* Compare classical and hybrid models using accuracy and ROC-AUC.
* Document the model design, training workflow, limitations, and future directions.

## Repository Structure

```text
QGNN-HybridGNN-QNN/
│
├── configs/                 # YAML configuration files for experiments
├── data/                    # Local data directory; raw data is not committed
├── docs/                    # Additional project documentation
├── notebooks/               # Summary and visualization notebooks
├── results/                 # Final metrics, plots, and result summaries
├── scripts/                 # Command-line scripts for training/evaluation
├── src/
│   └── qgnn_hybrid/          # Main Python package
│       ├── data.py           # Dataset loading and graph construction
│       ├── models.py         # Classical and hybrid GNN architectures
│       ├── qnn.py            # PennyLane quantum circuit components
│       ├── train.py          # Training utilities
│       └── utils.py          # Plotting and helper functions
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
└── raw/
    └── QG_jets.npz
```

Instructions for downloading or preparing the dataset are coming soon.

## Graph Construction

Each jet is represented as a graph:

* Nodes represent particles.
* Node features include particle-level kinematic information.
* Edges connect nearby particles in angular space.
* The graph is passed into a GNN to learn a jet-level representation.

Current graph construction details:

```text
k-nearest neighbors: Coming soon
node features: Coming soon
edge features: Coming soon
maximum particles per jet: Coming soon
train/validation/test split: Coming soon
```

## Models

### Classical GNN Baselines

This repository includes classical graph-based baselines for comparison.

Implemented or planned baselines include:

* GraphSAGE
* ParticleNet-style EdgeConv model
* GNN with classical MLP classification head

These baselines establish the classical performance level needed to evaluate whether the quantum component provides useful representational power.

### Hybrid GNN-QNN Model

The hybrid model uses a classical GNN backbone followed by a variational quantum circuit.

The pipeline is:

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
Variational quantum circuit
   ↓
Pauli-Z expectation values
   ↓
Final classifier
   ↓
Quark/gluon prediction
```

The quantum circuit is implemented using PennyLane. Classical features are encoded into the circuit using rotation gates, followed by entangling CNOT layers and trainable single-qubit rotations. The final quantum state is measured using Pauli-Z expectation values, producing a vector of quantum features for classification.

Current quantum circuit settings:

```text
number of qubits: 8
encoding: RY angle embedding
entanglement: ring CNOT pattern
trainable gates: RY rotations
measurement: Pauli-Z expectation values
circuit depth: Coming soon
```

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

PyTorch Geometric installation may depend on your local PyTorch, CUDA, or Apple Silicon configuration. See the official PyTorch Geometric installation guide if needed.

## Usage

### Train a classical GNN-MLP model

```bash
python scripts/train.py --config configs/gnn_mlp.yaml
```

### Train the hybrid GNN-QNN model

```bash
python scripts/train.py --config configs/gnn_qnn.yaml
```

### Evaluate trained models

```bash
python scripts/evaluate.py --config configs/gnn_mlp.yaml
python scripts/evaluate.py --config configs/gnn_qnn.yaml
```

### Generate plots and result summaries

```bash
python scripts/make_plots.py
```

Command-line scripts are under active cleanup. Final reproducible commands are coming soon.

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

This project is being organized to support reproducible experimentation through:

* Config-driven training
* Fixed random seeds
* Multi-seed evaluation
* Saved metrics summaries
* Version-controlled source code
* Lightweight result files
* Ignored raw data and model checkpoints

Planned reproducibility checklist:

```text
[ ] Finalize dataset download/preparation script
[ ] Finalize YAML configs
[ ] Add exact package versions
[ ] Add multi-seed summary script
[ ] Add final results CSV files
[ ] Add final plots
[ ] Add smoke tests
```

## Testing

Run tests with:

```bash
pytest
```

Current tests are coming soon.

Planned tests include:

* Package import test
* Data-loading smoke test
* Model forward-pass test
* Training-loop smoke test on a small batch

## Documentation

Additional documentation is planned in the `docs/` directory:

```text
docs/
├── architecture.md
├── experiment_notes.md
└── results_interpretation.md
```

These documents will describe the graph construction, model architecture, quantum circuit design, and interpretation of the results.

## Limitations

This project currently has several important limitations:

* Quantum circuit simulation is computationally expensive.
* The current QNN layer operates on a reduced latent representation rather than the full particle graph.
* The hybrid model has not yet shown clear performance improvement over the best classical baseline.
* Results depend on dataset size, graph construction choices, circuit depth, and random seed.
* The current implementation is designed for research benchmarking, not production deployment.

## Future Work

Future improvements may include:

* Hyperparameter scans over the number of qubits and circuit depth
* Alternative quantum feature maps
* Alternative entanglement patterns
* More expressive GNN backbones
* Larger multi-seed experiments
* Runtime and parameter-efficiency comparisons
* Comparison against additional published jet-tagging baselines
* Deployment of a lightweight inference demo

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
* Reproducible ML project organization

## References

1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. “Quantum Graph Neural Networks.” arXiv:1909.12264, 2019.

2. Beer, K., Khosla, M., Köhler, J., Osborne, T. J., & Zhao, T. “Quantum machine learning of graph-structured data.” Physical Review A, 108(1), 012410, 2023.

3. Komiske, P. T., Metodiev, E. M., & Thaler, J. “Energy Flow Networks: Deep Sets for Particle Jets.” Journal of High Energy Physics, 2019.

4. Qu, H., & Gouskos, L. “Jet tagging via particle clouds.” Physical Review D, 101(5), 056019, 2020.

## Status

This repository is currently being cleaned and polished as a professional research portfolio project. Core model development is complete, while documentation, scripts, tests, and final reproducibility improvements are in progress.
