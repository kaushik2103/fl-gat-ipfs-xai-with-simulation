#!/usr/bin/env python3
# ======================================================
# GRAPH BUILDER FOR FEDERATED GAT
# Clients (client_0 → client_7) + Global Test
# FL-safe • Non-IID robust • Mask-correct
# ======================================================

import numpy as np
import torch
import random
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

# ======================================================
# CONFIG
# ======================================================

CLIENT_IDS = list(range(0, 8))   # 🔥 client_0 → client_7
K = 10
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parents[1]
FED_DATA_DIR = BASE_DIR / "dataset" / "fed_dataset_safe"
GRAPH_SAVE_DIR = BASE_DIR / "saved_graphs"
GRAPH_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# UTILS
# ======================================================

def validate_labels(y: np.ndarray, scope: str):
    unique = np.unique(y)
    if unique.min() < 0:
        raise ValueError(f"[ERROR] {scope}: Negative labels found")
    print(f"[INFO] {scope}: Classes present → {unique.tolist()}")

def normalize_features(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

def build_knn_edge_index(X: np.ndarray, k: int) -> torch.Tensor:
    adj = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=-1,
    )

    adj = adj.maximum(adj.T)

    rows, cols = adj.nonzero()
    edge_index = torch.tensor(
        np.vstack((rows, cols)),
        dtype=torch.long,
    )

    # self-loops
    n = X.shape[0]
    self_loops = torch.arange(n)
    self_loops = torch.stack([self_loops, self_loops])
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return coalesce(edge_index)

def add_train_test_masks(data: Data, train_ratio: float):
    n = data.num_nodes
    perm = torch.randperm(n)
    split = int(train_ratio * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[perm[:split]] = True
    data.test_mask[perm[split:]] = True
    return data

# ======================================================
# BUILD CLIENT GRAPH
# ======================================================

def build_client_graph(client_id: int):
    client_dir = FED_DATA_DIR / f"client_{client_id}"
    x_path = client_dir / "tabular.npy"
    y_path = client_dir / "labels.npy"

    if not x_path.exists() or not y_path.exists():
        print(f"[SKIP] client_{client_id} not found")
        return

    print(f"[CLIENT {client_id}] Building graph")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    validate_labels(y, f"CLIENT {client_id}")
    X = normalize_features(X)

    data = Data(
        x=torch.from_numpy(X),
        edge_index=build_knn_edge_index(X, K),
        y=torch.from_numpy(y),
    )

    data = add_train_test_masks(data, TRAIN_RATIO)

    save_path = GRAPH_SAVE_DIR / f"client_{client_id}_graph.pt"
    torch.save(data, save_path)

    print(
        f"[CLIENT {client_id}] Saved → {save_path} | "
        f"Nodes={data.num_nodes}, Edges={data.num_edges}"
    )

# ======================================================
# BUILD GLOBAL TEST GRAPH
# ======================================================

def build_global_test_graph():
    print("\n[GLOBAL TEST] Building graph")

    test_dir = FED_DATA_DIR / "global_test"
    x_path = test_dir / "tabular.npy"
    y_path = test_dir / "labels.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("global_test data missing")

    X = normalize_features(np.load(x_path).astype(np.float32))
    y = np.load(y_path).astype(np.int64)

    validate_labels(y, "GLOBAL TEST")

    data = Data(
        x=torch.from_numpy(X),
        edge_index=build_knn_edge_index(X, K),
        y=torch.from_numpy(y),
    )

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.ones(data.num_nodes, dtype=torch.bool)

    save_path = GRAPH_SAVE_DIR / "global_test_graph.pt"
    torch.save(data, save_path)

    print(
        f"[GLOBAL TEST] Saved → {save_path} | "
        f"Nodes={data.num_nodes}, Edges={data.num_edges}"
    )

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("=== BUILDING CLIENT GRAPHS ===")
    for cid in CLIENT_IDS:
        build_client_graph(cid)

    print("\n=== BUILDING GLOBAL TEST GRAPH ===")
    build_global_test_graph()

    print("\n✅ ALL GRAPHS BUILT SUCCESSFULLY")
