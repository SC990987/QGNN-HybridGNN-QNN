## Quantum Computing imports
import pennylane as qml
import numpy as np
from math import pi

## GNN imports
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, EdgeConv
from sklearn.metrics import accuracy_score, roc_auc_score

## Plotting imports
import matplotlib.pyplot as plt

import os
import time
from datetime import datetime


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    ## wraps the distance back to the range [-pi, pi]
    return (dphi + torch.pi) % (2 * torch.pi) - torch.pi

def preprocess(particles):
    ## Remove padding
    mask = particles[:, 0] > 0
    particles = particles[mask]

    pt  = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    ## Log scale the pt
    ## adding 1e-6 to avoid log(0)
    pt = torch.log(pt + 1e-6)

    ## Center Jet around 0
    ## makes the network translation invariant
    eta = eta - eta.mean()
    phi = phi - phi.mean()

    ## wraps the distance back to the range [-pi, pi]
    phi = (phi + torch.pi) % (2 * torch.pi) - torch.pi

    ## Standardize pt and eta
    ## sets the mean to 1 and a variance of 0
    pt  = (pt - pt.mean()) / (pt.std() + 1e-6)
    eta = (eta - eta.mean()) / (eta.std() + 1e-6)

    ## Encode phi
    ## avoids discontinuity at +- pi
    phi_sin = torch.sin(phi)
    phi_cos = torch.cos(phi)

    node_features = torch.stack([pt, eta, phi_sin, phi_cos], dim=1)

    return node_features, eta, phi

def build_edge_index(eta, phi, k):
    ## Builds the edges in our graph which is the dR between the pairs of particles
    N = eta.shape[0]

    eta_i = eta.view(N, 1)
    eta_j = eta.view(1, N)

    phi_i = phi.view(N, 1)
    phi_j = phi.view(1, N)

    d_eta = eta_i - eta_j
    d_phi = delta_phi(phi_i, phi_j)
    dR = torch.sqrt(d_eta**2 + d_phi**2)

    ## Ensures we don't ask for more neighbors than nodes
    k_eff = min(k, N - 1)

    ## Get the nearest neighbors
    knn = dR.topk(k=k_eff + 1, largest=False).indices[:, 1:]  # skip self

    edge_index = []
    for i in range(N):
        for j in knn[i]:
            edge_index.append([i, j.item()])

    return torch.tensor(edge_index).t().contiguous()

def build_edge_features(edge_index, eta, phi):
    ## Assigns features to our edges
    row, col = edge_index

    deta = eta[row] - eta[col]
    dphi = delta_phi(phi[row], phi[col])
    dR = torch.sqrt(deta**2 + dphi**2)

    return torch.stack([deta, dphi, dR], dim=1)


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=0.001,
    device=None,
    checkpoint_path="checkpoint.pt",
    patience=5,
    save_every=1
):
    # Device setup
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    # 📊 History for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "epoch_time": []
    }

    # Resume from checkpoint
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        val_loss = train_loss
        val_acc = 0
        val_auc = 0

        if val_loader is not None:
            model.eval()
            val_loss = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)

                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = F.cross_entropy(out, batch.y)
                    val_loss += loss.item()

                    probs = torch.softmax(out, dim=1)[:, 1]  # binary class prob
                    preds = torch.argmax(out, dim=1)

                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())

                    val_acc += (preds == batch.y).sum().item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)

            # AUC (safe check)
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.0

        # ⏱️ Timing
        epoch_time = time.time() - epoch_start_time
        history["epoch_time"].append(epoch_time)

        # 📊 Store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # 🖨️ Logging with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(
            f"[{timestamp}] Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss
            }, checkpoint_path)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pt")
            print("✅ Saved best model")

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("⏹ Early stopping triggered")
            if os.path.exists("best_model.pt"):
                print("Loading best model before returning...")
                model.load_state_dict(torch.load("best_model.pt", map_location=device))
            break

    return model, history

def evaluate_model(model, test_loader, device=None):
    ## Device Setup
    device = device or torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )

    model.eval()

    all_preds = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)

            ## Predictions for accuracy
            preds = out.argmax(dim=1).cpu().numpy()

            ## Probabilities for AUC
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

            labels = batch.y.cpu().numpy()

            all_preds.append(preds)
            all_scores.append(probs)
            all_labels.append(labels)

    ## Concatenate
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    ## Metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    return acc, auc

