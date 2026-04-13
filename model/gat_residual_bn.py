#!/usr/bin/env python3
# ============================================================
# SIMPLE RESIDUAL GAT + BATCH NORM
# Fast • FL-safe • Non-IID robust
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge


class StrongResidualGAT(torch.nn.Module):
    """
    Simple Residual GAT for Federated Intrusion Detection
    ----------------------------------------------------
    ✔ Single GAT layer (FAST)
    ✔ Residual projection
    ✔ BatchNorm (better convergence than LayerNorm)
    ✔ Edge dropout
    ✔ FedProx + FL safe
    ✔ High accuracy & macro-F1
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.15,
        attn_dropout: float = 0.2,
        edge_dropout: float = 0.1,
    ):
        super().__init__()

        self.edge_dropout = edge_dropout
        self.dropout = Dropout(dropout)

        # -------- GAT Layer --------
        self.gat = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
        )

        out_dim = hidden_channels * heads

        # -------- Residual Projection --------
        self.res_proj = Linear(in_channels, out_dim)

        # -------- Normalization --------
        self.bn = BatchNorm1d(out_dim)

        # -------- Classifier --------
        self.classifier = Linear(out_dim, num_classes)

        self.reset_parameters()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def reset_parameters(self):
        self.gat.reset_parameters()
        torch.nn.init.xavier_uniform_(self.res_proj.weight)
        torch.nn.init.zeros_(self.res_proj.bias)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x, edge_index):
        # ---- Edge Dropout ----
        edge_index, _ = dropout_edge(
            edge_index,
            p=self.edge_dropout,
            training=self.training,
        )

        # ---- GAT ----
        h = self.gat(x, edge_index)

        # ---- Residual ----
        h = h + self.res_proj(x)

        # ---- BatchNorm + Activation ----
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)

        # ---- Output ----
        logits = self.classifier(h)
        return logits
