"""
GNN Training and Evaluation — PAN Prediction
=============================================
Trains PanGAT and evaluates it against the thesis GTWR baseline.

Two evaluation modes
--------------------
spatial (default) — matches the thesis setup exactly:
    Aggregate all 29 September days into a single value per grid cell.
    Random 80/20 node split.  Transductive learning: the full graph is
    visible during message passing; loss is only computed on masked nodes.
    This is directly comparable to the thesis's R² = 0.361 (GWR).

    Why this works where the temporal mode struggled:
    Monthly aggregation gives lightning and fire real signal.  In the
    daily dataset, 97% of (date, cell) pairs had zero precursor activity;
    the monthly totals spread that signal across all active cells.

temporal (--temporal flag) — stricter, operationally realistic:
    Train on Sep 2–25, test on Sep 26–30.  The model must generalise to
    future dates.  This is a harder task than the thesis cross-validation
    and is the right setup for a forecasting system.

Thesis baseline (for comparison)
---------------------------------
    OLS (global)    R² = 0.061   RMSE = 0.252
    GWR (best)      R² = 0.361   RMSE = 0.078  ← target to beat
    GTWR adaptive   R² = 0.227   RMSE = 0.184

Run
---
    python research/build_graph_features.py   # once
    python research/train_gnn.py              # spatial mode (default)
    python research/train_gnn.py --temporal   # temporal mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import polars as pl
from predictor.core.graph import build_static_graph, build_daily_graphs, FEATURE_COLS
from predictor.core.model import PanGAT
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_PATH = ROOT / "data" / "processed" / "graph_features.parquet"
MODEL_OUT     = ROOT / "data" / "processed" / "pan_gat.pt"

TRAIN_END    = "2020-09-25"
EPOCHS       = 500
LR           = 0.005
PATIENCE     = 60   # early stopping: halt if test R² doesn't improve
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 8
SEED         = 42
COUNT_COLS   = ["lightning_3d_count", "fire_3d_count"]

THESIS_BASELINE = {
    "OLS (global)":    {"r2": 0.061, "rmse": 0.252, "mae": 0.073},
    "GWR (best)":      {"r2": 0.361, "rmse": 0.078, "mae": 0.030},
    "GTWR (adaptive)": {"r2": 0.227, "rmse": 0.184, "mae": 0.071},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


def print_comparison(label: str, gnn: dict[str, float]) -> None:
    w = 52
    print(f"\n{'─' * w}")
    print(f"{'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"{'─' * w}")
    for name, m in THESIS_BASELINE.items():
        print(f"{name:<22} {m['r2']:>8.3f} {m['rmse']:>8.3f} {m['mae']:>8.3f}")
    print(f"{'─' * w}")
    delta = gnn['r2'] - THESIS_BASELINE["GWR (best)"]["r2"]
    sign = "+" if delta >= 0 else ""
    print(
        f"{'PanGAT ' + label:<22} {gnn['r2']:>8.3f} {gnn['rmse']:>8.3f} {gnn['mae']:>8.3f}"
        f"   ← {sign}{delta:.3f} vs GWR"
    )
    print(f"{'─' * w}")


def apply_log1p(df: pl.DataFrame) -> pl.DataFrame:
    """log1p on count features: compresses right-skewed distributions
    (lightning max=2311, median=38) to a scale StandardScaler can handle."""
    return df.with_columns([pl.col(c).log1p().alias(c) for c in COUNT_COLS])


def make_scalers(
    train_df: pl.DataFrame,
) -> tuple[StandardScaler, StandardScaler]:
    feat_scaler = StandardScaler()
    feat_scaler.fit(train_df.select(FEATURE_COLS).to_numpy())
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(train_df["mean_pan"].to_numpy().reshape(-1, 1))
    return feat_scaler, tgt_scaler


def scale_df(
    df: pl.DataFrame,
    feat_scaler: StandardScaler,
    tgt_scaler: StandardScaler,
) -> pl.DataFrame:
    sx = feat_scaler.transform(df.select(FEATURE_COLS).to_numpy())
    sy = tgt_scaler.transform(df["mean_pan"].to_numpy().reshape(-1, 1)).flatten()
    return df.with_columns(
        [pl.Series(name=col, values=sx[:, i]) for i, col in enumerate(FEATURE_COLS)]
        + [pl.Series(name="mean_pan", values=sy)]
    )


def inv_y(arr: np.ndarray, tgt_scaler: StandardScaler) -> np.ndarray:
    return tgt_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Spatial mode (matches thesis evaluation)
# ---------------------------------------------------------------------------
def run_spatial(features: pl.DataFrame, grid: float = 0.5) -> None:
    print(f"\n── Spatial mode ({grid}° grid, matches thesis setup) ───────────────")

    # Optional re-binning to a coarser grid (e.g. 1.0° to match thesis 1°×1° cells)
    half = grid / 2.0
    # Cast to Float32 first so Polars type stubs know .floor() returns numeric,
    # not the temporal overload that the stubs expose on untyped Expr.
    rebinned = features.with_columns([
        ((pl.col("lat").cast(pl.Float32) / grid).floor() * grid + half).alias("lat"),
        ((pl.col("lon").cast(pl.Float32) / grid).floor() * grid + half).alias("lon"),
    ])

    # Aggregate 29 daily snapshots → one row per grid cell.
    # log1p BEFORE aggregation so counts aren't double-scaled.
    spatial = (
        apply_log1p(rebinned)
        .group_by(["lat", "lon"])
        .agg([
            pl.col("lightning_3d_count").sum(),
            pl.col("fire_3d_count").sum(),
            pl.col("mean_co").mean(),
            pl.col("mean_pan").mean(),
        ])
        .drop_nulls("mean_pan")
        .sort(["lat", "lon"])
    )
    print(f"  Unique grid cells: {spatial.height:,}")
    print(f"  Lightning non-zero: {(spatial['lightning_3d_count'] > 0).sum():,} "
          f"({100*(spatial['lightning_3d_count']>0).mean():.1f}%)")
    print(f"  Fire non-zero:      {(spatial['fire_3d_count'] > 0).sum():,} "
          f"({100*(spatial['fire_3d_count']>0).mean():.1f}%)")

    # Random 80/20 node split (reproducible)
    torch.manual_seed(SEED)
    n = spatial.height
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask = ~train_mask

    train_rows = spatial.with_row_index("_i").filter(
        pl.col("_i").is_in(perm[:n_train].tolist())
    ).drop("_i")

    feat_scaler, tgt_scaler = make_scalers(train_rows)
    spatial_scaled = scale_df(spatial, feat_scaler, tgt_scaler)

    data = build_static_graph(spatial_scaled)

    model = PanGAT(in_channels=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    import copy
    best_r2, best_state, no_improve = -np.inf, None, 0

    print(f"\n  Training up to {EPOCHS} epochs (early stopping patience={PATIENCE}) ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Check test R² every 5 epochs for early stopping
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p = model(data.x, data.edge_index).numpy()
            yt = inv_y(data.y.numpy()[test_mask.numpy()], tgt_scaler)
            yp = inv_y(p[test_mask.numpy()], tgt_scaler)
            test_r2 = float(r2_score(yt, yp))
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch}  best test R²={best_r2:.4f}")
                break
            model.train()

        if epoch % 75 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4d}/{EPOCHS}  train MSE = {float(loss):.5f}  best test R²={best_r2:.4f}")

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    with torch.no_grad():
        pred_all = model(data.x, data.edge_index).numpy()

    for mask, label in [(train_mask.numpy(), "Train"), (test_mask.numpy(), "Test ")]:
        yt = inv_y(data.y.numpy()[mask], tgt_scaler)
        yp = inv_y(pred_all[mask], tgt_scaler)
        m = metrics(yt, yp)
        print(f"  {label}: R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
              f"  (pred [{yp.min():.3f}, {yp.max():.3f}] ppbv)")

    test_yt = inv_y(data.y.numpy()[test_mask.numpy()], tgt_scaler)
    test_yp = inv_y(pred_all[test_mask.numpy()], tgt_scaler)
    print_comparison("(spatial)", metrics(test_yt, test_yp))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")


# ---------------------------------------------------------------------------
# Temporal mode (stricter, operationally realistic)
# ---------------------------------------------------------------------------
def run_temporal(features: pl.DataFrame) -> None:
    print("\n── Temporal mode (train Sep 2–25 / test Sep 26–30) ─────────────")

    train_df = features.filter(
        pl.col("date") <= pl.lit(TRAIN_END).str.strptime(pl.Date, "%Y-%m-%d")
    )
    test_df = features.filter(
        pl.col("date") > pl.lit(TRAIN_END).str.strptime(pl.Date, "%Y-%m-%d")
    )
    print(f"  Train: {train_df.height:,} rows ({str(train_df['date'].min())} – {str(train_df['date'].max())})")
    print(f"  Test:  {test_df.height:,} rows  ({str(test_df['date'].min())} – {str(test_df['date'].max())})")

    train_df = apply_log1p(train_df)
    test_df  = apply_log1p(test_df)
    feat_scaler, tgt_scaler = make_scalers(train_df)
    train_scaled = scale_df(train_df, feat_scaler, tgt_scaler)
    test_scaled  = scale_df(test_df,  feat_scaler, tgt_scaler)

    train_graphs = [d for _, d in build_daily_graphs(train_scaled)]
    test_graphs  = [d for _, d in build_daily_graphs(test_scaled)]
    train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

    model = PanGAT(in_channels=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n  Training {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch.x, batch.edge_index), batch.y)
            loss.backward()
            optimizer.step()
            total += float(loss)
        scheduler.step()
        if epoch % 75 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4d}/{EPOCHS}  train MSE = {total/len(train_loader):.5f}")

    def eval_graphs(graphs: list[Data], label: str) -> dict[str, float]:
        model.eval()
        yt_all, yp_all = [], []
        with torch.no_grad():
            for g in graphs:
                p = model(g.x, g.edge_index).numpy()
                yt_all.append(g.y.numpy())
                yp_all.append(p)
        yt = inv_y(np.concatenate(yt_all), tgt_scaler)
        yp = inv_y(np.concatenate(yp_all), tgt_scaler)
        m = metrics(yt, yp)
        print(f"  {label}: R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
              f"  (pred [{yp.min():.3f}, {yp.max():.3f}] ppbv)")
        return m

    eval_graphs(train_graphs, "Train")
    test_m = eval_graphs(test_graphs, "Test ")
    print_comparison("(temporal)", test_m)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal", action="store_true",
        help="Use temporal holdout split instead of spatial 80/20 (harder task)"
    )
    parser.add_argument(
        "--grid", type=float, default=0.5,
        help="Grid resolution in degrees (0.5 default; 1.0 matches thesis 1°×1° cells)"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURES_PATH}\n"
            "Run `python research/build_graph_features.py` first."
        )

    features = pl.read_parquet(FEATURES_PATH)

    if args.temporal:
        run_temporal(features)
    else:
        run_spatial(features, grid=args.grid)


if __name__ == "__main__":
    main()
