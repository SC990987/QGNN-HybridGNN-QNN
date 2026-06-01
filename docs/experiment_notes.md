# Experiment Notes

## Current status

The repository has been reorganized into a professional, package-style project for benchmarking hybrid GNN-QNN models on quark/gluon jet classification.

Major pieces now in place:

- package layout under `src/qgnn_hybrid/`,
- YAML-driven training configs,
- cleaned model registry,
- fast PyTorch QNN simulator backend,
- PennyLane reference circuits,
- pytest smoke tests,
- processed graph caching,
- model-specific output directories,
- README documentation updates,
- fastQML citation and acknowledgment,
- matched linear-head ablation.

---

## Key debugging work completed

### 1. Import and package cleanup

The original repo used root-level files such as:

```text
data.py
models.py
qnn.py
train.py
notebook_tools.py
```

These were moved into:

```text
src/qgnn_hybrid/
```

The import style was changed to package imports such as:

```python
from qgnn_hybrid.data import get_dataloaders
from qgnn_hybrid.models import HybridGNN_MLP
```

This made the project more suitable for installation, testing, and job-application portfolio review.

---

## 2. Config-driven training

Training is now controlled through YAML configs rather than hard-coded model choices.

Example:

```bash
python3 -m qgnn_hybrid.train --config configs/gnn_mlp.yaml --seed 42
```

This makes it easier to reproduce experiments across:

- model type,
- seed,
- batch size,
- number of epochs,
- learning rate,
- hidden dimension,
- number of qubits,
- number of QNN layers,
- output directory.

---

## 3. Processed graph cache issue

A major performance regression was traced to the new `data.py`.

The newer data pipeline originally allowed graphs with as few as two non-padded particles:

```text
min_particles = 2
```

while the original benchmark behavior skipped jets with fewer than `k` particles:

```text
min_particles = k = 16
```

This changed the graph dataset and also caused cache ambiguity.

### Fix

The cleaned `data.py` now defaults to:

```text
min_particles = k
```

and the processed graph cache filename includes both `k` and `min_particles`:

```text
data/processed/qg_jets_k16_min16.pt
```

This restored the original benchmark behavior and brought GNN-MLP training time back in line with the old branch.

### Lesson

Processed datasets must include all graph-building choices in the cache filename. Otherwise, stale caches can silently invalidate comparisons.

---

## 4. MPS device checks

Training was confirmed to use Apple Silicon MPS:

```text
Using device: mps
First model parameter device: mps:0
Batch x device: mps:0
Batch edge_index device: mps:0
Batch y device: mps:0
```

The slowdown was not caused by accidental CPU training.

---

## 5. Training metric overhead

The old training loop only computed training loss during the training phase. The newer loop originally computed additional training metrics, including train accuracy and train AUC.

Train AUC required collecting predictions and labels on CPU, which can cause overhead on MPS/CUDA due to repeated device-to-CPU transfers. Train accuracy also introduced extra synchronization through `.item()` calls.

### Final training behavior

The training loop now computes:

```text
training phase:
- train loss only

validation phase:
- validation loss
- validation accuracy
- validation AUC

test phase:
- test loss
- test accuracy
- test AUC
```

This matches the original training behavior more closely while keeping validation and test metrics.

---

## 6. `torch.compile` experiments

An experimental branch was created for optional QNN-only `torch.compile` support.

The motivation was that compiling the QNN simulator function could potentially speed up repeated circuit simulation calls.

However, on Apple Silicon/MPS, `torch.compile` produced backend/compiler issues. For stability, these changes were excluded from `main`.

### Current branch policy

```text
main:
    stable eager PyTorch implementation

torch.compile branch:
    experimental QNN optimization work
```

The README notes that `torch.compile` support is experimental and not part of the stable benchmark path yet.

---

## 7. fastQML attribution

The fast PyTorch QNN simulator work is based on Eric A. F. Reinhardt's `fastQML` project.

Reference added:

```bibtex
@software{Reinhardt_fastQML_2026,
  author  = {Reinhardt, Eric A. F.},
  title   = {{fastQML}: Fast {PyTorch} quantum-circuit primitives via {K}ronecker-product layer fusion},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ereinha/fastQML}
}
```

The README now gives explicit credit for the simulator work while distinguishing it from the model integration and HEP application work in this repository.

---

## 8. Matched linear-head ablation

A new model was added:

```text
gnn_linear_ablation
```

This model uses the same GraphSAGE encoder and 8-dimensional bottleneck as the GNN-MLP and QNN models, but replaces the MLP/QNN head with a single linear classifier.

Purpose:

> Test whether the MLP or QNN head adds predictive power beyond the shared graph encoder and bottleneck representation.

This ablation is critical because the initial results showed similar performance for GraphSAGE, GNN-MLP, and QNN-improved models.

---

## Current single-seed benchmark results

All results below are for seed 42 on the processed dataset:

```text
processed dataset: data/processed/qg_jets_k16_min16.pt
total graphs: 98,070
train: 68,649
validation: 14,710
test: 14,711
batch size: 32
hidden dim: 64
n_qubits: 8
q_layers: 4
learning rate: 0.001
patience: 5
```

| Model | Test accuracy | Test AUC |
|---|---:|---:|
| GraphSAGE | 0.7838 | 0.8613 |
| ParticleNet-style EdgeConv | 0.7900 | 0.8666 |
| GNN-MLP | 0.7857 | 0.8613 |
| GNN Linear Ablation | 0.7851 | 0.8610 |
| QNN Basic Torch | 0.7791 | 0.8536 |
| QNN Improved Torch | 0.7887 | 0.8613 |
| QNN Legacy CRY Torch | 0.7843 | 0.8596 |

---

## Current interpretation

The seed-42 results suggest that most of the classification power comes from the GraphSAGE encoder. The matched linear-head ablation reaches almost the same AUC as the GNN-MLP and improved QNN head.

This implies that, in the current setup, the MLP and QNN heads are not adding a clear measurable performance improvement beyond the shared graph encoder and 8-dimensional bottleneck.

This does not mean the heads are not learning. It means the current benchmark does not show that they improve generalization relative to a simple linear bottleneck classifier.

---

## Next experiments

### Required for final reporting

Run the full benchmark over multiple seeds:

```text
42, 43, 44, 45, 46
```

Report:

- mean test AUC,
- standard deviation of test AUC,
- mean test accuracy,
- standard deviation of test accuracy,
- trainable parameter count,
- average epoch time.

### Important ablations

1. **Vary bottleneck dimension**
   - `n_qubits = 2`
   - `n_qubits = 4`
   - `n_qubits = 8`
   - `n_qubits = 16`

2. **Vary QNN depth**
   - `q_layers = 1`
   - `q_layers = 2`
   - `q_layers = 4`
   - `q_layers = 6`

3. **Freeze the encoder**
   - train only the head,
   - compare linear, MLP, and QNN heads on the same fixed representation.

4. **Train heads on saved embeddings**
   - precompute graph embeddings from the shared encoder,
   - benchmark heads independently.

5. **Random/frozen QNN ablation**
   - compare trainable QNN vs fixed random QNN plus trainable final classifier.

---

## README updates still worth adding

Add a concise physics/graph representation section:

- what a jet is,
- detector coordinates `eta` and `phi`,
- transverse momentum `pT`,
- definition of `Delta R`,
- graph construction with k-nearest neighbors,
- node features and edge construction.

Suggested diagrams:

1. particles in the `eta-phi` plane,
2. k-nearest-neighbor graph construction,
3. model pipeline from jet constituents to graph to GNN to classifier head.

