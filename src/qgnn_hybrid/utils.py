# utils.py

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


# =========================================================
# General utilities
# =========================================================
def get_device(device=None):
    """
    Return the best available torch device unless one is explicitly provided.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_parent_dir(path):
    """
    Create the parent directory for a file path if it does not already exist.
    """
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def add_suffix_if_missing(path, suffix):
    """
    Add a file suffix if the path does not already have it.
    """
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(suffix)
    return path


def safe_roc_auc(labels, scores):
    """
    Compute ROC-AUC safely.

    roc_auc_score fails if only one class is present in labels.
    This can happen for very small validation/test splits.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(np.unique(labels)) < 2:
        return float("nan")

    return float(roc_auc_score(labels, scores))


def json_safe(obj):
    """
    Convert NumPy/Torch values into JSON-serializable Python objects.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(json_safe(value) for value in obj)

    return obj


def count_parameters(model):
    """
    Count trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# Graph construction utilities
# =========================================================
def delta_phi(phi1, phi2):
    """
    Compute wrapped angular difference in the range [-pi, pi].
    """
    dphi = phi1 - phi2
    return (dphi + torch.pi) % (2 * torch.pi) - torch.pi


def preprocess(particles):
    """
    Preprocess particle-level jet data into node features.

    Expected input columns:
        particles[:, 0] = pt
        particles[:, 1] = eta
        particles[:, 2] = phi

    Returns
    -------
    node_features:
        Tensor with columns [log_pt_standardized, eta_standardized, sin(phi), cos(phi)]
    eta:
        Centered eta values.
    phi:
        Centered and wrapped phi values.
    """
    mask = particles[:, 0] > 0
    particles = particles[mask]

    if particles.shape[0] == 0:
        raise ValueError("No non-padded particles found after preprocessing.")

    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    pt = torch.log(pt + 1e-6)

    eta = eta - eta.mean()
    phi = phi - phi.mean()

    phi = (phi + torch.pi) % (2 * torch.pi) - torch.pi

    pt = (pt - pt.mean()) / (pt.std(unbiased=False) + 1e-6)
    eta = (eta - eta.mean()) / (eta.std(unbiased=False) + 1e-6)

    phi_sin = torch.sin(phi)
    phi_cos = torch.cos(phi)

    node_features = torch.stack([pt, eta, phi_sin, phi_cos], dim=1)

    return node_features, eta, phi


def build_edge_index(eta, phi, k):
    """
    Build directed k-nearest-neighbor edges using angular distance dR.
    """
    num_nodes = eta.shape[0]

    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=eta.device)

    eta_i = eta.view(num_nodes, 1)
    eta_j = eta.view(1, num_nodes)

    phi_i = phi.view(num_nodes, 1)
    phi_j = phi.view(1, num_nodes)

    d_eta = eta_i - eta_j
    d_phi = delta_phi(phi_i, phi_j)
    dR = torch.sqrt(d_eta**2 + d_phi**2)

    k_eff = min(k, num_nodes - 1)

    knn = dR.topk(k=k_eff + 1, largest=False).indices[:, 1:]

    row = torch.arange(num_nodes, device=eta.device).repeat_interleave(k_eff)
    col = knn.reshape(-1)

    edge_index = torch.stack([row, col], dim=0).long()
    return edge_index.contiguous()


def build_edge_features(edge_index, eta, phi):
    """
    Build edge features [delta_eta, delta_phi, delta_R].
    """
    if edge_index.numel() == 0:
        return torch.empty((0, 3), dtype=eta.dtype, device=eta.device)

    row, col = edge_index

    deta = eta[row] - eta[col]
    dphi = delta_phi(phi[row], phi[col])
    dR = torch.sqrt(deta**2 + dphi**2)

    return torch.stack([deta, dphi, dR], dim=1)


# =========================================================
# Epoch helper
# =========================================================
def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    compute_auc=True,
    compute_accuracy=True,
):
    """
    Run one training or evaluation epoch.

    If optimizer is provided, the model is trained.
    If optimizer is None, the model is evaluated.

    Parameters
    ----------
    compute_auc:
        If True, collect prediction scores and labels to compute ROC-AUC.
        For training epochs, this should usually be False to avoid expensive
        device-to-CPU transfers every batch.

    compute_accuracy:
        If True, compute classification accuracy.
        For training epochs on MPS/CUDA, this can be False to avoid an extra
        device synchronization every batch.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_scores = []
    all_labels = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)

            if is_training:
                optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.long()

            loss = F.cross_entropy(out, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            if compute_accuracy:
                preds = torch.argmax(out, dim=1)
                total_correct += (preds == labels).sum().item()

            if compute_auc:
                probs = torch.softmax(out, dim=1)[:, 1]
                all_scores.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_examples, 1)

    if compute_accuracy:
        accuracy = total_correct / max(total_examples, 1)
    else:
        accuracy = float("nan")

    if compute_auc and all_scores and all_labels:
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        auc = safe_roc_auc(all_labels, all_scores)
    else:
        auc = float("nan")

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "auc": float(auc),
    }


# =========================================================
# Main training utility
# =========================================================
def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=0.001,
    device=None,
    checkpoint_path="checkpoint.pt",
    patience=5,
    save_every=1,
    save_history_every_epoch=False,
    history_dir="history_logs",
    model_name="best_model",
    training_history="training_history",
):
    """
    Train a model with optional validation, checkpointing, and early stopping.

    During training, only train loss is computed. Train accuracy and train AUC
    are intentionally skipped to avoid extra device synchronization overhead,
    especially on MPS/CUDA.

    Validation AUC and test AUC are still computed.
    """
    device = get_device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = Path(checkpoint_path)
    best_model_path = add_suffix_if_missing(model_name, ".pt")
    history_path = add_suffix_if_missing(training_history, ".json")

    ensure_parent_dir(checkpoint_path)
    ensure_parent_dir(best_model_path)
    ensure_parent_dir(history_path)

    if save_history_every_epoch:
        history_dir = Path(history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "epoch_time": [],
    }

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint.get("epoch", -1) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)

        if "history" in checkpoint:
            history = checkpoint["history"]

            # Backward compatibility with older checkpoints that stored
            # training metrics we no longer track.
            history.pop("train_auc", None)
            history.pop("train_acc", None)

        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            compute_auc=False,
            compute_accuracy=False,
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                compute_auc=True,
                compute_accuracy=True,
            )
        else:
            val_metrics = {
                "loss": train_metrics["loss"],
                "accuracy": float("nan"),
                "auc": float("nan"),
            }

        epoch_time = time.time() - epoch_start_time

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["epoch_time"].append(float(epoch_time))

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if save_history_every_epoch:
            epoch_file = history_dir / f"epoch_{epoch + 1}.json"

            with open(epoch_file, "w") as f:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["accuracy"],
                        "val_auc": val_metrics["auc"],
                        "epoch_time": float(epoch_time),
                    },
                    f,
                    indent=4,
                )

        if save_every is not None and save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                    "history": json_safe(history),
                },
                checkpoint_path,
            )

        current_loss = val_metrics["loss"]

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    with open(history_path, "w") as f:
        json.dump(json_safe(history), f, indent=4)

    print(f"Saved full training history to {history_path}")

    return model, history


# =========================================================
# Evaluation utility
# =========================================================
def evaluate_model(model, test_loader, device=None):
    """
    Evaluate a trained model on a test DataLoader.

    Returns
    -------
    acc:
        Test accuracy.
    auc:
        Test ROC-AUC.
    """
    device = get_device(device)
    model = model.to(device)

    metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        compute_auc=True,
        compute_accuracy=True,
    )

    acc = metrics["accuracy"]
    auc = metrics["auc"]

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    return acc, auc