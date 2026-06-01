# data.py

from pathlib import Path
import shutil
import urllib.request
import random

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import qgnn_hybrid.utils as nbtools


# =========================================================
# Paths and dataset config
# =========================================================
DATA_URL = "https://zenodo.org/record/3164691/files/QG_jets.npz"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"

DEFAULT_DATA_PATH = DEFAULT_RAW_DIR / "QG_jets.npz"
LEGACY_DATA_PATH = Path("QG_jets.npz")


# =========================================================
# Download and loading utilities
# =========================================================
def download_dataset(path=DEFAULT_DATA_PATH, url=DATA_URL):
    """
    Download the QG jets dataset if it does not already exist.

    If a legacy root-level QG_jets.npz file exists, copy it into data/raw/
    instead of downloading again.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Dataset already exists at {path}")
        return path

    if LEGACY_DATA_PATH.exists():
        print(f"Found legacy dataset at {LEGACY_DATA_PATH}")
        print(f"Copying dataset to {path}")
        shutil.copy2(LEGACY_DATA_PATH, path)
        return path

    print(f"Downloading dataset to {path}...")
    urllib.request.urlretrieve(url, path)
    print("Download complete.")

    return path


def load_raw_data(files=None):
    """
    Load raw QG jet arrays from one or more .npz files.

    Parameters
    ----------
    files:
        Optional list of dataset files. If None, uses data/raw/QG_jets.npz.

    Returns
    -------
    X:
        Jet constituent array.
    y:
        Label array.
    """
    if files is None:
        files = [DEFAULT_DATA_PATH]

    X_list, y_list = [], []

    for file_path in files:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}. "
                "Run download_dataset() first or provide the correct path."
            )

        data = np.load(file_path)
        X_list.append(data["X"])
        y_list.append(data["y"])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# =========================================================
# Graph construction
# =========================================================
def build_graphs(
    X,
    y,
    k=16,
    max_events=None,
    min_particles=None,
):
    """
    Convert raw jet arrays into PyTorch Geometric Data objects.

    By default, min_particles is set equal to k. This preserves the original
    benchmark behavior, where jets with fewer than k valid particles were
    skipped.

    Parameters
    ----------
    X:
        Raw particle-level input array.
    y:
        Labels.
    k:
        Number of nearest neighbors for graph construction.
    max_events:
        Optional maximum number of events to process. Useful for debugging.
    min_particles:
        Minimum number of non-padded particles required to build a graph.
        If None, defaults to k.

    Returns
    -------
    graph_list:
        List of PyTorch Geometric Data objects.
    """
    if min_particles is None:
        min_particles = k

    graph_list = []

    n_events = X.shape[0]
    if max_events is not None:
        n_events = min(n_events, max_events)

    skipped = 0

    for i in range(n_events):
        particles = torch.tensor(X[i], dtype=torch.float32)

        try:
            node_features, eta, phi = nbtools.preprocess(particles)
        except ValueError:
            skipped += 1
            continue

        # Preserve original benchmark behavior:
        # skip jets with fewer than k valid particles by default.
        if node_features.shape[0] < min_particles:
            skipped += 1
            continue

        edge_index = nbtools.build_edge_index(eta, phi, k)
        edge_attr = nbtools.build_edge_features(edge_index, eta, phi)

        label = torch.tensor(y[i], dtype=torch.long)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label,
        )

        graph_list.append(graph)

    print(f"Built {len(graph_list)} graphs.")
    if skipped > 0:
        print(f"Skipped {skipped} events with fewer than {min_particles} valid particles.")

    return graph_list


# =========================================================
# Processed graph cache
# =========================================================
def save_processed_graphs(graph_list, path):
    """
    Save processed graph objects to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(graph_list, path)
    print(f"Saved processed graphs to {path}")


def load_processed_graphs(path):
    """
    Load processed graph objects from disk.

    Note:
    PyTorch 2.6 changed torch.load() to default to weights_only=True.
    PyTorch Geometric Data objects require weights_only=False because they are
    full Python objects, not just tensor state dictionaries.

    Only use weights_only=False for trusted local files generated by this repo.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed graph file not found: {path}")

    try:
        graph_list = torch.load(path, weights_only=False)
    except TypeError:
        # For older PyTorch versions that do not support weights_only.
        graph_list = torch.load(path)

    print(f"Loaded {len(graph_list)} processed graphs from {path}")

    return graph_list


def get_processed_path(k=16, max_events=None, min_particles=None):
    """
    Build a deterministic processed-cache filename.

    min_particles is included in the filename because changing the filtering
    rule changes the graph dataset.
    """
    if min_particles is None:
        min_particles = k

    if max_events is None:
        return DEFAULT_PROCESSED_DIR / f"qg_jets_k{k}_min{min_particles}.pt"

    return DEFAULT_PROCESSED_DIR / f"qg_jets_k{k}_min{min_particles}_max{max_events}.pt"


# =========================================================
# Splitting and dataloaders
# =========================================================
def split_data(
    graph_list,
    seed=42,
    train_frac=0.70,
    val_frac=0.15,
):
    """
    Split graph list into train, validation, and test sets.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")

    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")

    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")

    graph_list = list(graph_list)

    rng = random.Random(seed)
    rng.shuffle(graph_list)

    n_graphs = len(graph_list)

    train_end = int(train_frac * n_graphs)
    val_end = train_end + int(val_frac * n_graphs)

    train_data = graph_list[:train_end]
    val_data = graph_list[train_end:val_end]
    test_data = graph_list[val_end:]

    print(
        f"Split sizes: "
        f"train={len(train_data)}, "
        f"val={len(val_data)}, "
        f"test={len(test_data)}"
    )

    return train_data, val_data, test_data


def get_dataloaders(
    batch_size=32,
    k=16,
    seed=42,
    files=None,
    download=True,
    use_cache=True,
    max_events=None,
    min_particles=None,
    train_frac=0.70,
    val_frac=0.15,
    num_workers=0,
):
    """
    Build train, validation, and test DataLoaders.

    This function remains compatible with:

        get_dataloaders(batch_size=32)

    Default behavior:
        min_particles defaults to k, preserving the original benchmark dataset
        construction where jets with fewer than k valid particles are skipped.

    Parameters
    ----------
    batch_size:
        DataLoader batch size.
    k:
        Number of nearest neighbors for graph construction.
    seed:
        Random seed for split and train shuffling.
    files:
        Optional list of raw .npz files.
    download:
        If True, download the default dataset when files is None.
    use_cache:
        If True, save/load processed graph objects from data/processed.
    max_events:
        Optional maximum number of events for debugging.
    min_particles:
        Minimum number of non-padded particles required. If None, defaults to k.
    train_frac:
        Fraction of data used for training.
    val_frac:
        Fraction of data used for validation.
    num_workers:
        DataLoader workers.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    if min_particles is None:
        min_particles = k

    if files is None:
        if download:
            download_dataset(DEFAULT_DATA_PATH)
        files = [DEFAULT_DATA_PATH]

    processed_path = get_processed_path(
        k=k,
        max_events=max_events,
        min_particles=min_particles,
    )

    if use_cache and processed_path.exists():
        graph_list = load_processed_graphs(processed_path)
    else:
        X, y = load_raw_data(files)

        graph_list = build_graphs(
            X,
            y,
            k=k,
            max_events=max_events,
            min_particles=min_particles,
        )

        if use_cache:
            save_processed_graphs(graph_list, processed_path)

    train_data, val_data, test_data = split_data(
        graph_list,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader