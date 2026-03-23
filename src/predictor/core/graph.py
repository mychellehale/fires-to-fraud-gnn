"""
Spatial Graph Construction
==========================
Converts the fused feature table into PyTorch Geometric Data objects,
one graph snapshot per calendar day.

Graph definition
----------------
- Node   : a 0.5° × 0.5° grid cell that has at least one PAN observation
           on a given day.
- Edge   : directed pair (i → j) where cell j is among cell i's 8 nearest
           spatial neighbours.  Using k=8 NN guarantees every node has
           the same degree regardless of how sparse the observations are on
           a given day, making training numerically stable.
- Node features (x):
    [0] lightning_3d_count  — strike count in preceding 72 h
    [1] fire_3d_count        — fire event count in preceding 72 h
    [2] mean_co              — mean CO (ppbv) on the observation day
    [3] lat                  — grid-cell latitude  (used as a spatial prior)
    [4] lon                  — grid-cell longitude
- Target (y) : mean_pan — mean PAN concentration (ppbv)

Why k=8 nearest neighbours
----------------------------
Radius-based edges leave isolated nodes when observations are sparse.
k-NN guarantees connectivity.  k=8 mirrors the 8-cell Moore neighbourhood
of a regular grid — the same spatial context a convolution would see —
while still allowing the GAT attention mechanism to up- or down-weight
individual neighbours based on their features.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import BallTree
from torch_geometric.data import Data

FEATURE_COLS = ["lightning_3d_count", "fire_3d_count", "mean_co", "lat", "lon"]
TARGET_COL = "mean_pan"
KNN = 8  # Moore neighbourhood equivalent on a 0.5° grid


def _knn_edges(lats: np.ndarray, lons: np.ndarray, k: int) -> torch.Tensor:
    """
    Returns a (2, E) edge_index tensor connecting each node to its k nearest
    spatial neighbours using the haversine metric.

    Self-loops are excluded.  Edges are directed: for each pair (i, j) where
    j ∈ kNN(i), we add both i→j and j→i so message passing is bidirectional.
    """
    coords_rad = np.radians(np.column_stack([lats, lons]))
    tree = BallTree(coords_rad, metric="haversine")

    # query k+1 because the nearest neighbour of a point is itself
    distances, indices = tree.query(coords_rad, k=min(k + 1, len(lats)))

    src, dst = [], []
    for i, neighbours in enumerate(indices):
        for j in neighbours:
            if i != j:
                src.append(i)
                dst.append(j)

    return torch.tensor([src, dst], dtype=torch.long)


def build_static_graph(features: pl.DataFrame, k: int = KNN) -> Data:
    """
    Builds a single spatial graph from time-aggregated features.

    Used in transductive learning where the whole graph is available during
    message passing but the loss is computed only on masked (train/test) nodes.
    This matches the thesis evaluation setup: one row per unique grid cell,
    features averaged/summed over the full study period.

    Args:
        features: DataFrame with FEATURE_COLS + 'mean_pan' (one row per cell).
        k:        k-NN neighbours per node.

    Returns:
        PyG Data with x, edge_index, y populated.
    """
    lats = features["lat"].to_numpy()
    lons = features["lon"].to_numpy()
    x = torch.tensor(features.select(FEATURE_COLS).to_numpy(), dtype=torch.float)
    y = torch.tensor(features["mean_pan"].to_numpy(), dtype=torch.float)
    edge_index = _knn_edges(lats, lons, k=k)
    return Data(x=x, edge_index=edge_index, y=y)


def build_daily_graphs(
    features: pl.DataFrame,
    k: int = KNN,
) -> list[tuple[str, Data]]:
    """
    Builds one PyG Data object per calendar day in the feature table.

    Args:
        features: Output of build_graph_features.py — must contain
                  FEATURE_COLS + [TARGET_COL, 'date'].
        k:        Number of nearest neighbours per node (default 8).

    Returns:
        List of (date_str, Data) tuples sorted chronologically.
        Data attributes:
            x           FloatTensor  (N, 5)  — node features (unscaled)
            edge_index  LongTensor   (2, E)  — k-NN edges
            y           FloatTensor  (N,)    — PAN target
    """
    graphs: list[tuple[str, Data]] = []

    for date_val, day_df in features.sort("date").group_by("date", maintain_order=True):
        date_str = str(date_val)

        if day_df.height < 2:
            # Need at least 2 nodes to form an edge
            continue

        lats = day_df["lat"].to_numpy()
        lons = day_df["lon"].to_numpy()

        x = torch.tensor(
            day_df.select(FEATURE_COLS).to_numpy(), dtype=torch.float
        )
        y = torch.tensor(day_df[TARGET_COL].to_numpy(), dtype=torch.float)
        edge_index = _knn_edges(lats, lons, k=k)

        graphs.append((date_str, Data(x=x, edge_index=edge_index, y=y)))

    return graphs
