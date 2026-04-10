import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, Sequential, ReLU
from torch_geometric.nn import GINConv


class ResidualGIN(torch.nn.Module):
    """
    Deep Residual GIN with:
    - GINConv layers (learnable epsilon)
    - MLP inside each GINConv
    - Batch Normalization
    - Residual connections
    - Dropout
    Suitable for large-scale node classification (FL + NeighborLoader).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert num_layers >= 3, "num_layers must be >= 3"

        self.num_layers = num_layers
        self.dropout = Dropout(p=dropout)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # -------- Input layer --------
        mlp_in = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp_in, train_eps=True))
        self.bns.append(BatchNorm1d(hidden_channels))

        # -------- Hidden layers --------
        for _ in range(num_layers - 2):
            mlp_hidden = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp_hidden, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))

        # -------- Output layer (no BN, no ReLU) --------
        mlp_out = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, num_classes),
        )
        self.convs.append(GINConv(mlp_out, train_eps=True))

    def forward(self, x, edge_index):
        """
        x: [num_nodes, num_features]
        edge_index: [2, num_edges]
        """

        residual = None

        # Hidden layers
        for i in range(self.num_layers - 1):
            out = self.convs[i](x, edge_index)
            out = self.bns[i](out)
            out = F.relu(out)

            # Residual connection (shape-safe)
            if residual is not None:
                out = out + residual

            out = self.dropout(out)

            residual = out
            x = out

        # Final layer (logits only)
        x = self.convs[-1](x, edge_index)
        return x


# ----------------- OPTIONAL TEST -----------------
if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.randn(100, 57)          # 100 nodes, 57 features
    edge_index = torch.randint(0, 100, (2, 400))
    y = torch.randint(0, 8, (100,))

    data = Data(x=x, edge_index=edge_index, y=y)

    model = ResidualGIN(
        in_channels=57,
        hidden_channels=128,
        num_classes=8,
        num_layers=6,
        dropout=0.5,
    )

    out = model(data.x, data.edge_index)
    print("Output shape:", out.shape)  # should be [100, 8]
