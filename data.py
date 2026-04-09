# data.py

import os
import urllib.request
import numpy as np
import torch
import random

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import notebook_tools as nbtools  # your preprocessing module


def download_dataset(path="QG_jets.npz"):
    url = "https://zenodo.org/record/3164691/files/QG_jets.npz"

    if not os.path.exists(path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print("Dataset already exists.")


def load_raw_data(files=["QG_jets.npz"]):
    X_list, y_list = [], []

    for f in files:
        data = np.load(f)
        X_list.append(data["X"])
        y_list.append(data["y"])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


def build_graphs(X, y, k=16):
    graph_list = []

    for i in range(X.shape[0]):
        particles = torch.tensor(X[i], dtype=torch.float)

        # preprocess
        node_features, eta, phi = nbtools.preprocess(particles)

        if node_features.shape[0] < k:
            continue

        edge_index = nbtools.build_edge_index(eta, phi, k)
        edge_attr = nbtools.build_edge_features(edge_index, eta, phi)

        label = torch.tensor(y[i], dtype=torch.long)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label
        )

        graph_list.append(graph)

    return graph_list


def split_data(graph_list, seed=42):
    random.Random(seed).shuffle(graph_list)

    n = len(graph_list)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    train_data = graph_list[:train_end]
    val_data   = graph_list[train_end:val_end]
    test_data  = graph_list[val_end:]

    return train_data, val_data, test_data


def get_dataloaders(batch_size=32, k=16):
    download_dataset()

    X, y = load_raw_data()
    graph_list = build_graphs(X, y, k=k)

    train_data, val_data, test_data = split_data(graph_list)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size)
    test_loader  = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader