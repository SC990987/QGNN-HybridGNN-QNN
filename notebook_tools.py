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

    # ✅ Resume from checkpoint if exists
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

        # ✅ Validation loss (for early stopping)
        if val_loader is not None:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = F.cross_entropy(out, batch.y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
        else:
            val_loss = train_loss

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ✅ Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss
            }, checkpoint_path)

        # ✅ Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pt")
            print("✅ Saved best model")

        else:
            patience_counter += 1

        # ✅ Early stopping
        if patience_counter >= patience:
            print("⏹ Early stopping triggered")
            # ✅ Load best model before returning
            if os.path.exists("best_model.pt"):
                print("Loading best model before returning...")
                model.load_state_dict(torch.load("best_model.pt", map_location=device))
            break

    return model

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

