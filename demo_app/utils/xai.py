# ============================================================
# XAI MODULE FOR FL-GAT-IPFS DEMO
# GNNExplainer (Correct Feature Extraction + Clean Graph)
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import to_networkx


# ============================================================
# INITIALIZE EXPLAINER
# ============================================================

def get_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=150),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


# ============================================================
# EXPLAIN SINGLE NODE (FIXED FEATURE EXTRACTION)
# ============================================================

def explain_node(model, data, node_idx, device="cpu"):

    model.eval()
    model.to(device)
    data = data.to(device)

    explainer = get_explainer(model)

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=int(node_idx),
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE (FIXED — NO FLATTEN BUG)
    # --------------------------------------------------------
    if explanation.node_mask is not None:
        node_mask = explanation.node_mask.detach().cpu().numpy()

        # PyG returns (num_nodes, num_features)
        if node_mask.ndim == 2:
            feature_importance = node_mask[int(node_idx)]
        else:
            feature_importance = node_mask.flatten()

    else:
        feature_importance = np.zeros(data.x.size(1))

    # --------------------------------------------------------
    # EDGE IMPORTANCE
    # --------------------------------------------------------
    if explanation.edge_mask is not None:
        edge_importance = explanation.edge_mask.detach().cpu().numpy()
    else:
        edge_importance = np.zeros(data.edge_index.size(1))

    return explanation, feature_importance, edge_importance


# ============================================================
# GET TOP FEATURES
# ============================================================

def get_top_features(feature_importance, feature_names=None, top_k=5):

    if feature_importance is None or len(feature_importance) == 0:
        return []

    scores = np.array(feature_importance)
    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in indices:
        if feature_names is not None and idx < len(feature_names):
            name = feature_names[idx]
        else:
            name = f"Feature_{idx}"

        results.append((name, float(scores[idx])))

    return results


# ============================================================
# FEATURE IMPORTANCE PLOT (CLEAN)
# ============================================================

def plot_feature_importance(top_features):

    fig, ax = plt.subplots(figsize=(7, 4))

    if not top_features:
        ax.text(0.5, 0.5, "No feature importance available",
                ha="center", va="center")
        ax.axis("off")
        return fig

    names = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]

    ax.barh(names[::-1], scores[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Contributing Features")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig


# ============================================================
# VISUALIZE LOCAL EXPLANATION GRAPH
# ============================================================

def visualize_explanation_graph(data, edge_importance, node_idx, threshold=0.5):

    fig, ax = plt.subplots(figsize=(6, 6))

    if edge_importance is None or len(edge_importance) == 0:
        ax.text(0.5, 0.5, "No edge importance available",
                ha="center", va="center")
        ax.axis("off")
        return fig

    G = to_networkx(data, to_undirected=True)

    edge_index_np = data.edge_index.t().cpu().numpy()

    # Select important edges
    important_edges = []
    for i, (u, v) in enumerate(edge_index_np):
        if i < len(edge_importance) and edge_importance[i] > threshold:
            important_edges.append((u, v))

    # Only show local neighborhood of selected node
    neighbors = list(G.neighbors(node_idx))
    sub_nodes = [node_idx] + neighbors
    subgraph = G.subgraph(sub_nodes)

    pos = nx.spring_layout(subgraph, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_size=300,
        node_color="lightblue",
        ax=ax
    )

    # Highlight selected node
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=[node_idx],
        node_color="red",
        node_size=400,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edge_color="gray",
        alpha=0.5,
        ax=ax
    )

    # Highlight important edges
    important_local = [
        e for e in important_edges
        if e[0] in sub_nodes and e[1] in sub_nodes
    ]

    if important_local:
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=important_local,
            edge_color="red",
            width=2,
            ax=ax
        )

    ax.set_title("Local Explanation Graph (Highlighted Edges)")
    ax.axis("off")

    plt.tight_layout()
    return fig


# ============================================================
# TEXT EXPLANATION
# ============================================================

def generate_text_explanation(pred_label, confidence, top_features):

    explanation = (
        f"The model predicts '{pred_label}' "
        f"with a confidence of {confidence:.2f}%. "
    )

    if top_features:
        feature_text = ", ".join(
            [f"{name} ({score:.2f})" for name, score in top_features]
        )
        explanation += (
            f"This decision is primarily influenced by: {feature_text}."
        )

    return explanation
