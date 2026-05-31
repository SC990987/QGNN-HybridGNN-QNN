# utils.py

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score


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
    # Remove padded particles
    mask = particles[:, 0] > 0
    particles = particles[mask]

    if particles.shape[0] == 0:
        raise ValueError("No non-padded particles found after preprocessing.")

    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    # Log-scale pt
    pt = torch.log(pt + 1e-6)

    # Center jet coordinates
    eta = eta - eta.mean()
    phi = phi - phi.mean()

    # Wrap phi back into [-pi, pi]
    phi = (phi + torch.pi) % (2 * torch.pi) - torch.pi

    # Standardize pt and eta
    pt = (pt - pt.mean()) / (pt.std(unbiased=False) + 1e-6)
    eta = (eta - eta.mean()) / (eta.std(unbiased=False) + 1e-6)

    # Encode phi continuously
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

    # Do not ask for more neighbors than available nodes
    k_eff = min(k, num_nodes - 1)

    # Nearest neighbors, skipping self
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
# Evaluation helper
# =========================================================
def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
):
    """
    Run one training or evaluation epoch.

    If optimizer is provided, the model is trained.
    If optimizer is None, the model is evaluated.
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

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            total_correct += (preds == labels).sum().item()
            total_examples += batch_size

            all_scores.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    all_scores = np.concatenate(all_scores) if all_scores else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    auc = safe_roc_auc(all_labels, all_scores) if len(all_labels) > 0 else float("nan")

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

    This function is compatible with the current train.py script, which passes:

        model_name=str(output_dir / f"{run_name}_best_model")
        training_history=str(output_dir / f"{run_name}_training_history")
        checkpoint_path=str(output_dir / f"{run_name}_checkpoint")

    Parameters
    ----------
    model:
        PyTorch model.
    train_loader:
        Training DataLoader.
    val_loader:
        Optional validation DataLoader.
    epochs:
        Maximum number of epochs.
    lr:
        Adam learning rate.
    device:
        Optional torch device or device string.
    checkpoint_path:
        Path for resumable checkpoint.
    patience:
        Early stopping patience based on validation loss.
    save_every:
        Save checkpoint every N epochs.
    save_history_every_epoch:
        If True, save one JSON file per epoch.
    history_dir:
        Directory for per-epoch history files.
    model_name:
        Output path prefix for best model. ".pt" is added if missing.
    training_history:
        Output path prefix for full training history. ".json" is added if missing.

    Returns
    -------
    model:
        Best model loaded at the end of training.
    history:
        Dictionary of training and validation metrics.
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
        "train_acc": [],
        "train_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "epoch_time": [],
    }

    # Resume checkpoint if available
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

        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
            )
        else:
            val_metrics = {
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "auc": train_metrics["auc"],
            }

        epoch_time = time.time() - epoch_start_time

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_auc"].append(train_metrics["auc"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        history["epoch_time"].append(float(epoch_time))

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save per-epoch history if requested
        if save_history_every_epoch:
            epoch_file = history_dir / f"epoch_{epoch + 1}.json"

            with open(epoch_file, "w") as f:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["loss"],
                        "train_acc": train_metrics["accuracy"],
                        "train_auc": train_metrics["auc"],
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["accuracy"],
                        "val_auc": val_metrics["auc"],
                        "epoch_time": float(epoch_time),
                    },
                    f,
                    indent=4,
                )

        # Save checkpoint
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

        # Save best model based on validation loss
        current_loss = val_metrics["loss"]

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience is not None and patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model before returning
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save full history
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
    )

    acc = metrics["accuracy"]
    auc = metrics["auc"]

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    return acc, auc