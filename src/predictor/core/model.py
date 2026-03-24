"""
PanGAT — Graph Attention Network for PAN Concentration Prediction
==================================================================
A 2-layer Graph Attention Network (GAT) that predicts PAN concentration
from lightning, fire, and CO precursor features at each spatial grid cell.

Why GAT over simpler GNN variants
-----------------------------------
The thesis GTWR used a fixed Gaussian kernel to weight spatial neighbours.
GAT replaces that with *learned* attention weights conditioned on node
features — a node with high lightning activity can receive more weight from
its upwind neighbours than from downwind ones, without the model being
explicitly told wind direction.

The attention weight α_{ij} between nodes i and j is:

    α_{ij} = softmax( LeakyReLU( a^T [Wh_i || Wh_j] ) )

where W is a shared linear transformation and a is a learnable attention
vector.  Multi-head attention (heads=4) runs this in parallel with
different weight matrices and concatenates the results, allowing the model
to attend to neighbours for different reasons simultaneously.

Architecture
------------
    Input (5 features)
        ↓
    GATConv(5 → 32, heads=4)  → 128 channels + ELU
        ↓
    GATConv(128 → 32, heads=1) → 32 channels + ELU
        ↓
    Linear(32 → 1)             → predicted PAN (ppbv)

Dropout (p=0.1) before each GAT layer regularises the small training set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv


class PanGAT(torch.nn.Module):
    """
    Graph Attention Network predicting PAN concentration (ppbv).

    Args:
        in_channels:     Number of input node features (default 5).
        hidden_channels: Channels per attention head in layer 1 (default 32).
        heads:           Number of attention heads in layer 1 (default 4).
        dropout:         Dropout probability applied before each conv (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention — output is hidden_channels * heads
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
        )

        # Layer 2: single head, concatenation disabled → output is hidden_channels
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        # Regression head
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          Node feature matrix  (N, in_channels).
            edge_index: Graph connectivity    (2, E).

        Returns:
            Predicted PAN values             (N,).
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))

        return self.lin(x).squeeze(-1)
