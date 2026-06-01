# Results Interpretation

## Summary

The current benchmark shows that hybrid GNN-QNN models can be integrated into a graph-based quark/gluon jet classifier and achieve competitive performance with classical graph baselines.

However, the current single-seed results do **not** show a clear advantage from the QNN head. The matched linear-head ablation performs almost identically to the GNN-MLP and improved QNN head, suggesting that most of the classification power is coming from the shared GraphSAGE encoder.

These results should be interpreted as a careful benchmark and ablation study, not as evidence of quantum advantage.

---

## Dataset and training setup

The seed-42 benchmark used the processed graph dataset:

```text
data/processed/qg_jets_k16_min16.pt
```

Dataset split:

| Split | Number of graphs |
|---|---:|
| Train | 68,649 |
| Validation | 14,710 |
| Test | 14,711 |
| Total | 98,070 |

Training configuration:

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Hidden dimension | 64 |
| Qubits / bottleneck dimension | 8 |
| QNN layers | 4 |
| Learning rate | 0.001 |
| Early stopping patience | 5 |
| Seed | 42 |

The graph construction used:

```text
k = 16
min_particles = 16
```

This preserves the original benchmark behavior where jets with fewer than 16 valid particles are skipped.

---

## Single-seed results

| Model | Test accuracy | Test AUC | Interpretation |
|---|---:|---:|---|
| GraphSAGE | 0.7838 | 0.8613 | Simple classical GNN baseline |
| ParticleNet-style EdgeConv | **0.7900** | **0.8666** | Strongest overall single-seed model |
| GNN-MLP | 0.7857 | 0.8613 | Matched classical nonlinear head |
| GNN Linear Ablation | 0.7851 | 0.8610 | Matched linear bottleneck ablation |
| QNN Basic Torch | 0.7791 | 0.8536 | Basic QNN head underperforms |
| QNN Improved Torch | 0.7887 | 0.8613 | Best QNN; competitive with matched baselines |
| QNN Legacy CRY Torch | 0.7843 | 0.8596 | Legacy QNN; close but below improved QNN |

---

## Main conclusions

### 1. ParticleNet is the strongest model in the current benchmark

The ParticleNet-style EdgeConv baseline achieved:

```text
Test accuracy = 0.7900
Test AUC      = 0.8666
```

This is the best single-seed result. This is expected because ParticleNet-style EdgeConv is a more expressive particle-cloud architecture and is not constrained by the same 8-dimensional bottleneck used by the matched GNN-MLP and GNN-QNN models.

ParticleNet should therefore be interpreted as a strong classical performance baseline, not as a matched ablation of the QNN head.

---

## 2. The improved QNN is competitive, but not better than classical matched heads

The improved QNN reached:

```text
Test accuracy = 0.7887
Test AUC      = 0.8613
```

This is competitive with:

```text
GraphSAGE:  Test AUC = 0.8613
GNN-MLP:    Test AUC = 0.8613
```

This means the improved QNN head does not introduce a large performance penalty relative to classical baselines. However, it also does not clearly improve performance over the matched classical head.

A careful statement is:

> The improved QNN head is competitive with the matched GNN-MLP head in this seed-42 benchmark, but it does not demonstrate an advantage over classical heads.

---

## 3. The linear-head ablation is the most important result

The matched linear-head ablation reached:

```text
Test accuracy = 0.7851
Test AUC      = 0.8610
```

This is almost identical to the GNN-MLP and improved QNN models.

This suggests:

> The shared GraphSAGE encoder and 8-dimensional bottleneck representation capture most of the useful classification information. In the current setup, the MLP and QNN heads do not add a clear measurable improvement beyond a simple linear classifier.

This does not mean the MLP or QNN parameters are not learning. It means their learned transformations do not measurably improve test AUC relative to a linear head after the same encoder.

---

## 4. The basic QNN is weaker

The basic QNN reached:

```text
Test accuracy = 0.7791
Test AUC      = 0.8536
```

This is below the matched GNN-MLP, linear-head ablation, GraphSAGE baseline, and improved QNN.

Possible explanations:

- the basic circuit is not expressive enough,
- the ring CNOT + RY-only structure is too restrictive,
- the circuit is harder to optimize,
- the GraphSAGE encoder already produces a representation where a simpler head is sufficient,
- the QNN transformation may distort useful features.

---

## 5. The legacy CRY QNN is close but not leading

The legacy CRY model reached:

```text
Test accuracy = 0.7843
Test AUC      = 0.8596
```

This is close to the stronger matched baselines but still slightly below the improved QNN and GNN-MLP.

This model should be retained for reproducibility but not treated as the main QNN architecture.

---

## Runtime interpretation

Approximate steady-state epoch times from the seed-42 logs:

| Model | Approximate steady epoch time |
|---|---:|
| GraphSAGE | 20-22 s |
| GNN Linear Ablation | 20-23 s |
| GNN-MLP | 21-23 s |
| ParticleNet-style EdgeConv | 70-72 s |
| QNN Basic Torch | 79-84 s |
| QNN Legacy CRY Torch | 82-90 s |
| QNN Improved Torch | 170-180 s |

The QNN models are significantly slower than the matched classical heads. The improved QNN is especially expensive due to its more expressive circuit structure.

This matters for interpretation:

> The improved QNN achieves competitive AUC, but the current implementation is much slower than the matched classical linear and MLP heads.

---

## Recommended README wording

A good concise README interpretation would be:

```text
In the seed-42 benchmark, the improved GNN-QNN model achieved competitive performance with the matched GNN-MLP baseline, reaching a test ROC-AUC of 0.8613. However, a matched linear-head ablation achieved a similar test ROC-AUC of 0.8610, suggesting that most of the classification power comes from the shared GraphSAGE encoder and 8-dimensional bottleneck representation. The strongest overall model was the ParticleNet-style EdgeConv baseline with a test ROC-AUC of 0.8666. These results indicate that the QNN head can be integrated into a graph-based jet classifier, but the current setup does not demonstrate a clear performance advantage over classical heads.
```

---

## What should be claimed

Appropriate claims:

- The repository implements a reproducible hybrid GNN-QNN benchmark for quark/gluon jet classification.
- The improved QNN head is competitive with matched classical heads.
- The ParticleNet-style model remains the strongest classical baseline.
- A matched linear-head ablation suggests the encoder is responsible for most of the performance.
- The current results do not show quantum advantage.
- Further ablations are needed to determine when the QNN head provides unique benefit.

Avoid claiming:

- quantum advantage,
- QNN superiority over classical baselines,
- that the QNN is definitely learning more useful features,
- that the QNN is better because it ties classical baselines,
- that a single seed is enough for final conclusions.

---

## Next steps before finalizing results

### 1. Multi-seed benchmark

Run:

```text
seeds = 42, 43, 44, 45, 46
```

Report:

```text
mean test AUC ± std
mean test accuracy ± std
average epoch time
trainable parameter count
```

### 2. Bottleneck-size study

Run the matched heads with:

```text
n_qubits = 2, 4, 8, 16
```

This will test whether the head matters more when the bottleneck is tighter.

### 3. Frozen encoder study

Train the shared encoder once, freeze it, and compare:

```text
linear head
MLP head
basic QNN head
improved QNN head
legacy QNN head
```

This isolates the head from encoder adaptation.

### 4. Fixed/random QNN ablation

Compare:

```text
trainable QNN
random fixed QNN + trainable final linear layer
identity/pass-through head
```

This tests whether the QNN transformation itself is contributing useful learned structure.

### 5. Parameter-count table

Add trainable parameter counts for all models. This is important because the improved QNN has more circuit parameters than the basic QNN, and ParticleNet likely has a different model capacity than the matched-head models.

---

## Final interpretation

The current project is strongest when framed as:

> A reproducible, carefully controlled benchmark of hybrid graph neural network / quantum neural network models for jet tagging, including matched classical ablations and stronger classical baselines.

The most scientifically honest conclusion is:

> The QNN head is competitive with classical matched heads, but the present results suggest that the GraphSAGE encoder dominates performance. The current setup does not show a clear QNN advantage, motivating further ablations with tighter bottlenecks, frozen encoders, and multi-seed evaluation.

