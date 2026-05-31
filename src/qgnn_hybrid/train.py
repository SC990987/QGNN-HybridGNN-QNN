# train.py

import argparse
import json
import random
from pathlib import Path
import yaml

import numpy as np
import torch

from qgnn_hybrid.data import get_dataloaders
from qgnn_hybrid.models import (
    HybridGNN_QNN_Basic_Torch,
    HybridGNN_QNN_Improved_Torch,
    HybridGNN_QNN_LegacyCRY_Torch,
    HybridGNN_QNN_Basic_PennyLane,
    HybridGNN_QNN_Improved_PennyLane,
    HybridGNN_MLP,
    JetGNN,
    ParticleNet,
)
import qgnn_hybrid.utils as nbtools


# =========================================================
# Available models
# =========================================================
MODEL_REGISTRY = {
    "qnn_basic_torch": HybridGNN_QNN_Basic_Torch,
    "qnn_improved_torch": HybridGNN_QNN_Improved_Torch,
    "qnn_legacy_cry_torch": HybridGNN_QNN_LegacyCRY_Torch,
    "qnn_basic_pennylane": HybridGNN_QNN_Basic_PennyLane,
    "qnn_improved_pennylane": HybridGNN_QNN_Improved_PennyLane,
    "gnn_mlp": HybridGNN_MLP,
    "graphsage": JetGNN,
    "particlenet": ParticleNet,
}


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    in_channels: int,
    hidden_dim: int,
    n_qubits: int,
    q_layers: int,
):
    """Create a model from the model registry."""
    model_cls = MODEL_REGISTRY[model_name]

    if model_name == "graphsage":
        return model_cls(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

    if model_name == "particlenet":
        return model_cls(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
        )

    return model_cls(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        q_layers=q_layers,
    )


def load_config(config_path):
    """Load YAML config file."""
    if config_path is None:
        return {}

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GNN, GNN-MLP, and hybrid GNN-QNN models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config file.",
    )

    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--q-layers", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)

    parser.add_argument(
        "--save-history-every-epoch",
        action="store_true",
        help="Save one JSON history file per epoch.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    defaults = {
        "model": "qnn_basic_torch",
        "batch_size": 32,
        "epochs": 50,
        "patience": 5,
        "seed": 42,
        "lr": 1e-3,
        "hidden_dim": 64,
        "n_qubits": 8,
        "q_layers": 4,
        "output_dir": "outputs",
        "save_every": 1,
        "save_history_every_epoch": False,
    }

    # Start from defaults, overwrite with config, then overwrite with CLI args.
    merged = defaults | config

    for key, value in vars(args).items():
        if key == "config":
            continue

        # For regular args, CLI overrides config only if explicitly provided.
        if value is not None:
            merged[key] = value

    # Special case: argparse store_true defaults to False, so preserve config unless CLI flag is used.
    if args.save_history_every_epoch:
        merged["save_history_every_epoch"] = True

    return argparse.Namespace(**merged)


# =========================================================
# Main training loop
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.model}_seed_{args.seed}"

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )

    in_channels = train_loader.dataset[0].x.shape[1]

    # --- Model ---
    model = build_model(
        model_name=args.model,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        n_qubits=args.n_qubits,
        q_layers=args.q_layers,
    )

    print("=" * 70)
    print(f"Training model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Input channels: {in_channels}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Qubits: {args.n_qubits}")
    print(f"Q layers: {args.q_layers}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # --- Train ---
    model, history = nbtools.train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_every=args.save_every,
        save_history_every_epoch=args.save_history_every_epoch,
        history_dir=str(output_dir / f"{run_name}_epoch_history"),
        model_name=str(output_dir / f"{run_name}_best_model"),
        training_history=str(output_dir / f"{run_name}_training_history"),
        checkpoint_path=str(output_dir / f"{run_name}_checkpoint.pt"),
    )

    # --- Evaluate ---
    acc, auc = nbtools.evaluate_model(model, test_loader)

    # --- Save final metrics ---
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "test_accuracy": float(acc),
        "test_auc": float(auc),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_qubits": args.n_qubits,
        "q_layers": args.q_layers,
        "in_channels": int(in_channels),
        "history": history,
    }

    metrics_path = output_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(nbtools.json_safe(metrics), f, indent=4)

    print("=" * 70)
    print("Training complete.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print("=" * 70)


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()