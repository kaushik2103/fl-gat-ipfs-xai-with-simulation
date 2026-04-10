import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import SAGEConv


class ResidualGSAGE(torch.nn.Module):
    """
    Deep GraphSAGE with:
    - SAGEConv layers
    - Batch Normalization
    - Residual connections
    - Dropout
    Designed for large-scale cyber threat detection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"

        self.num_layers = num_layers
        self.dropout = Dropout(p=dropout)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # -------- Input layer --------
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # -------- Hidden layers --------
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # -------- Output layer (no BN) --------
        self.convs.append(SAGEConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, num_features]
        edge_index: Graph edges [2, num_edges]
        """
        residual = None

        # Hidden layers
        for i in range(self.num_layers - 1):
            out = self.convs[i](x, edge_index)
            out = self.bns[i](out)
            out = F.relu(out)

            # Residual connection
            if residual is not None:
                out = out + residual

            out = self.dropout(out)

            residual = out
            x = out

        # Output layer (logits)
        x = self.convs[-1](x, edge_index)
        return x


# ----------------- OPTIONAL TEST -----------------
if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.randn(100, 57)
    edge_index = torch.randint(0, 100, (2, 400))
    y = torch.randint(0, 8, (100,))

    data = Data(x=x, edge_index=edge_index, y=y)

    model = ResidualGSAGE(
        in_channels=57,
        hidden_channels=128,
        num_classes=8,
        num_layers=4,
        dropout=0.3,
    )

    out = model(data.x, data.edge_index)
    print("Output shape:", out.shape)  # [100, 8]
