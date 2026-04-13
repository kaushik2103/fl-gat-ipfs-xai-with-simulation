# ============================================================
# GRAPH BUILDER FOR DEMO APPLICATION
# Used for inference + GNNExplainer
# Compatible with FL-GAT-IPFS trained model
# ============================================================

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DEFAULT_K = 10  # Must match training graph construction


# ============================================================
# BUILD KNN EDGE INDEX
# ============================================================

def build_knn_edge_index(X: np.ndarray, k: int = DEFAULT_K) -> torch.Tensor:
    """
    Construct symmetric KNN graph with self-loops.

    Handles small datasets safely.
    """

    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be numpy array")

    if X.ndim != 2:
        raise ValueError("Input must be 2D feature matrix")

    num_nodes = X.shape[0]

    # Prevent crash if dataset smaller than K
    k = min(k, max(1, num_nodes - 1))

    # Build adjacency matrix
    adj = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=-1,
    )

    # Make graph symmetric
    adj = adj.maximum(adj.T)

    rows, cols = adj.nonzero()

    edge_index = torch.tensor(
        np.vstack((rows, cols)),
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Add self-loops
    # --------------------------------------------------------
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)

    edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Remove duplicates and sort
    edge_index = coalesce(edge_index)

    return edge_index


# ============================================================
# BUILD GRAPH FROM FEATURES (INFERENCE)
# ============================================================

def build_graph_from_features(X: np.ndarray) -> Data:
    """
    Convert preprocessed feature matrix into PyG Data object.
    Used for model inference.
    """

    if not isinstance(X, np.ndarray):
        raise ValueError("Input features must be numpy array")

    if X.ndim != 2:
        raise ValueError("Feature input must be 2D")

    X = X.astype(np.float32)

    edge_index = build_knn_edge_index(X)

    data = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
    )

    return data


# ============================================================
# DEVICE TRANSFER
# ============================================================

def move_to_device(data: Data, device: str = "cpu") -> Data:
    """
    Move graph object to specified device.
    """

    if not isinstance(data, Data):
        raise ValueError("Input must be torch_geometric Data object")

    return data.to(device)
