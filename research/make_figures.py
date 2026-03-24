"""
make_figures.py — Generate all Medium article visuals
======================================================
Produces six publication-quality figures in figures/.

    Fig 1  — Model comparison bar chart
    Fig 2  — Feature sparsity (daily vs monthly coverage)
    Fig 3  — Log1p transform distribution
    Fig 4  — GNN architecture schematic
    Fig 5  — Predicted vs actual scatter (test set)
    Fig 6  — CONUS maps: lightning → fire → PAN causal chain

Run:
    python research/make_figures.py

Requires graph_features.parquet and pan_gat.pt in data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictor.core.graph import build_static_graph, FEATURE_COLS
from predictor.core.model import PanGAT

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FEATURES_PATH = ROOT / "data" / "processed" / "graph_features.parquet"
MODEL_PATH    = ROOT / "data" / "processed" / "pan_gat.pt"

SEED = 42

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PALETTE = {
    "ols":    "#9e9e9e",
    "gtwr":   "#78909c",
    "gwr":    "#42a5f5",
    "pangat": "#ef5350",
    "bg":     "#0d1117",
    "text":   "#e6edf3",
    "grid":   "#21262d",
    "accent": "#f78166",
}

def dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "font.size":         12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Figure 1 — Model comparison
# ---------------------------------------------------------------------------
def fig_model_comparison() -> None:
    dark_style()

    models = ["OLS\n(Global)", "GTWR\n(Adaptive)", "GWR\n(Non-adaptive)", "PanGAT\n(Ours)"]
    r2    = [0.061, 0.227, 0.361, 0.323]
    rmse  = [0.252, 0.184, 0.078, 0.069]
    colors = [PALETTE["ols"], PALETTE["gtwr"], PALETTE["gwr"], PALETTE["pangat"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    x = np.arange(len(models))
    w = 0.6

    # R² (higher is better)
    bars1 = ax1.bar(x, r2, width=w, color=colors, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylabel("R²", fontsize=13)
    ax1.set_title("Explained Variance  (higher → better)", fontsize=13, pad=12)
    ax1.set_ylim(0, 0.45)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, r2):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                 color=PALETTE["text"], fontweight="bold")

    # Arrow + annotation on PanGAT bar for R²
    ax1.annotate("", xy=(3, r2[3] + 0.002), xytext=(2, r2[2] + 0.002),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["accent"],
                                 lw=1.5, connectionstyle="arc3,rad=0.2"))

    # RMSE (lower is better)
    bars2 = ax2.bar(x, rmse, width=w, color=colors, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylabel("RMSE (ppbv)", fontsize=13)
    ax2.set_title("Prediction Error  (lower → better)", fontsize=13, pad=12)
    ax2.set_ylim(0, 0.30)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, rmse):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                 color=PALETTE["text"], fontweight="bold")

    # Star annotation on PanGAT RMSE bar
    ax2.text(3, rmse[3] + 0.018, "Best RMSE ✓",
             ha="center", color=PALETTE["pangat"], fontsize=11, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=PALETTE["ols"],    label="Thesis baseline"),
        mpatches.Patch(color=PALETTE["gtwr"],   label="Thesis spatiotemporal"),
        mpatches.Patch(color=PALETTE["gwr"],    label="Thesis best (GWR)"),
        mpatches.Patch(color=PALETTE["pangat"], label="PanGAT — this work"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("PanGAT vs Thesis Baselines — September 2020 CONUS",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_model_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Sparsity: daily vs monthly coverage
# ---------------------------------------------------------------------------
def fig_sparsity() -> None:
    dark_style()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    categories  = ["Daily snapshots\n(temporal split)", "Monthly aggregation\n(spatial split)"]
    zero_pct    = [100 - 1.3,  100 - 19.6]
    nonzero_pct = [1.3, 19.6]

    x = np.arange(len(categories))
    w = 0.45

    ax.bar(x, zero_pct,    width=w, color="#37474f", label="Zero lightning signal",  zorder=3)
    ax.bar(x, nonzero_pct, width=w, color=PALETTE["pangat"],
           label="Non-zero signal",  zorder=3, bottom=zero_pct)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("% of grid cells", fontsize=13)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Why the temporal split failed:\n97% of daily cells had zero precursor signal",
                 fontsize=13, pad=12)

    for i, (z, nz) in enumerate(zip(zero_pct, nonzero_pct)):
        ax.text(i, nz + z + 2, f"{nz:.1f}%", ha="center", fontsize=13,
                color=PALETTE["pangat"], fontweight="bold")
        ax.text(i, z / 2, f"{z:.1f}%", ha="center", fontsize=11,
                color="#90a4ae")

    ax.annotate("R² = 0.002\n(model collapsed)", xy=(0, 105), ha="center",
                fontsize=11, color=PALETTE["accent"], fontweight="bold")
    ax.annotate("R² = 0.323\n(✓ model learns)", xy=(1, 105), ha="center",
                fontsize=11, color=PALETTE["pangat"], fontweight="bold")

    ax.legend(loc="upper right", frameon=False, fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_sparsity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Log1p transform on real lightning counts
# ---------------------------------------------------------------------------
def fig_log1p_transform() -> None:
    dark_style()

    features = pl.read_parquet(FEATURES_PATH)
    raw = (
        features.group_by(["lat", "lon"])
        .agg(pl.col("lightning_3d_count").sum())
        ["lightning_3d_count"]
        .to_numpy()
    )
    raw_nz   = raw[raw > 0]
    log_nz   = np.log1p(raw_nz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    kw = dict(bins=50, color=PALETTE["pangat"], alpha=0.85, edgecolor="none", zorder=3)
    ax1.hist(raw_nz,  **kw)
    ax2.hist(log_nz,  **kw)

    ax1.set_title("Raw lightning count\n(monthly sum, non-zero cells)", fontsize=13)
    ax2.set_title("After log1p transform\n(same data)", fontsize=13)
    ax1.set_xlabel("Count", fontsize=12)
    ax2.set_xlabel("log1p(count)", fontsize=12)
    ax1.set_ylabel("Number of cells", fontsize=12)
    ax2.set_ylabel("Number of cells", fontsize=12)

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    stats1 = f"median={np.median(raw_nz):.0f}  max={raw_nz.max():.0f}"
    stats2 = f"median={np.median(log_nz):.1f}  max={log_nz.max():.1f}"
    ax1.text(0.97, 0.95, stats1, transform=ax1.transAxes,
             ha="right", va="top", fontsize=10, color=PALETTE["text"],
             bbox=dict(facecolor=PALETTE["grid"], alpha=0.7, edgecolor="none", pad=4))
    ax2.text(0.97, 0.95, stats2, transform=ax2.transAxes,
             ha="right", va="top", fontsize=10, color=PALETTE["text"],
             bbox=dict(facecolor=PALETTE["grid"], alpha=0.7, edgecolor="none", pad=4))

    ax1.annotate("Long tail → 5–10σ\noutliers after scaling", xy=(raw_nz.max() * 0.7, 5),
                 xytext=(raw_nz.max() * 0.45, 50),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.5),
                 color=PALETTE["accent"], fontsize=10, ha="center")

    fig.suptitle("Failure 2 — Raw counts broke gradient flow. log1p fixed it.",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_log1p_transform.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — GNN architecture schematic
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    dark_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, color, label, sublabel="", fontsize=11):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x, y + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                color=PALETTE["text"], fontweight="bold")
        if sublabel:
            ax.text(x, y - 0.3, sublabel,
                    ha="center", va="center", fontsize=9,
                    color=PALETTE["text"], alpha=0.7)

    def arrow(x0, x1, y=3.5, color="#546e7a"):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    # Node features
    box(1.4, 3.5, 2.2, 4.6, "#546e7a", "Node features", "")
    for i, feat in enumerate(["lightning\n count", "fire\n count", "mean CO", "lat", "lon"]):
        fy = 5.5 - i * 1.0
        feat_rect = mpatches.FancyBboxPatch((0.45, fy - 0.35), 1.9, 0.7,
                                            boxstyle="round,pad=0.05", linewidth=1,
                                            edgecolor="#546e7a", facecolor="#546e7a44")
        ax.add_patch(feat_rect)
        ax.text(1.4, fy, feat, ha="center", va="center", fontsize=8.5,
                color=PALETTE["text"])
    ax.text(1.4, 0.8, "5 features", ha="center", fontsize=9, color="#90a4ae")

    arrow(2.5, 3.8)

    # GAT Layer 1
    box(5.0, 3.5, 2.2, 4.6, "#1565c0", "GATConv", "5 → 32 ch × 4 heads\n+ ELU")
    ax.text(5.0, 0.8, "128 channels", ha="center", fontsize=9, color="#90a4ae")

    # Attention weight callout
    ax.text(5.0, 6.6,
            r"$\alpha_{ij} = \mathrm{softmax}\!\left(\mathrm{LeakyReLU}\!\left(a^{\top}[Wh_i \| Wh_j]\right)\right)$",
            ha="center", va="center", fontsize=9.5,
            color="#64b5f6",
            bbox=dict(facecolor="#0d2137", edgecolor="#1565c0", alpha=0.85, pad=5))

    arrow(6.1, 7.4)

    # GAT Layer 2
    box(8.6, 3.5, 2.2, 4.6, "#0277bd", "GATConv", "128 → 32 ch × 1 head\n+ ELU")
    ax.text(8.6, 0.8, "32 channels", ha="center", fontsize=9, color="#90a4ae")

    arrow(9.7, 11.0)

    # Linear readout
    box(11.8, 3.5, 1.8, 2.0, PALETTE["pangat"], "Linear", "32 → 1")
    ax.text(11.8, 0.8, "PAN (ppbv)", ha="center", fontsize=9, color="#90a4ae")

    # k=8 neighbor annotation
    ax.annotate("k=8 nearest\nneighbors\n(haversine)",
                xy=(3.8, 4.7), xytext=(2.8, 6.1),
                arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.2),
                fontsize=9, color="#90a4ae", ha="center")

    ax.set_title("PanGAT Architecture — Graph Attention Network for PAN Prediction",
                 fontsize=14, fontweight="bold", pad=10, color=PALETTE["text"])

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Predicted vs actual PAN (from saved model)
# ---------------------------------------------------------------------------
def fig_pred_vs_actual() -> None:
    dark_style()

    features = pl.read_parquet(FEATURES_PATH)

    # Replicate the exact spatial mode preprocessing from train_gnn.py
    GRID = 1.0
    COUNT_COLS = ["lightning_3d_count", "fire_3d_count"]
    half = GRID / 2.0

    spatial = (
        features
        .with_columns([
            ((pl.col("lat").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lat"),
            ((pl.col("lon").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lon"),
        ])
        .with_columns([pl.col(c).log1p().alias(c) for c in COUNT_COLS])
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

    torch.manual_seed(SEED)
    n = spatial.height
    perm      = torch.randperm(n)
    n_train   = int(0.8 * n)
    train_idx = perm[:n_train].tolist()
    test_idx  = perm[n_train:].tolist()

    train_rows = spatial.with_row_index("_i").filter(
        pl.col("_i").is_in(train_idx)
    ).drop("_i")

    feat_scaler = StandardScaler()
    feat_scaler.fit(train_rows.select(FEATURE_COLS).to_numpy())
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(train_rows["mean_pan"].to_numpy().reshape(-1, 1))

    sx = feat_scaler.transform(spatial.select(FEATURE_COLS).to_numpy())
    sy = tgt_scaler.transform(spatial["mean_pan"].to_numpy().reshape(-1, 1)).flatten()

    spatial_scaled = spatial.with_columns(
        [pl.Series(name=col, values=sx[:, i]) for i, col in enumerate(FEATURE_COLS)]
        + [pl.Series(name="mean_pan", values=sy)]
    )

    data = build_static_graph(spatial_scaled)

    model = PanGAT(in_channels=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred_scaled = model(data.x, data.edge_index).numpy()

    def inv(arr):
        return tgt_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True

    yt = inv(data.y.numpy()[test_mask])
    yp = inv(pred_scaled[test_mask])
    r2 = float(r2_score(yt, yp))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Hexbin density plot
    hb = ax.hexbin(yt, yp, gridsize=45, cmap="YlOrRd",
                   mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Cell count", color=PALETTE["text"], fontsize=11)
    cb.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=PALETTE["text"])

    # 1:1 reference line
    lims = [min(yt.min(), yp.min()) - 0.01,
            max(yt.max(), yp.max()) + 0.01]
    ax.plot(lims, lims, "--", color="#78909c", lw=1.5, label="Perfect prediction", zorder=5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Observed PAN (ppbv)", fontsize=13)
    ax.set_ylabel("Predicted PAN (ppbv)", fontsize=13)
    ax.set_title("PanGAT — Predicted vs Observed PAN\n(test set, 20% held-out nodes)",
                 fontsize=13, pad=12)

    ax.text(0.05, 0.93, f"R² = {r2:.3f}\nRMSE = {rmse:.4f} ppbv",
            transform=ax.transAxes, fontsize=12, va="top",
            color=PALETTE["text"],
            bbox=dict(facecolor=PALETTE["grid"], edgecolor=PALETTE["pangat"],
                      alpha=0.9, pad=6))

    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.yaxis.grid(True, zorder=0, alpha=0.4)
    ax.xaxis.grid(True, zorder=0, alpha=0.4)

    plt.tight_layout()
    out = FIGURES_DIR / "fig5_pred_vs_actual.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — CONUS storytelling maps: lightning → fire → PAN
# ---------------------------------------------------------------------------
def fig_conus_maps() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LogNorm

    features = pl.read_parquet(FEATURES_PATH)

    # Monthly spatial aggregate at 0.5° resolution
    GRID = 0.5
    half = GRID / 2.0
    spatial = (
        features
        .with_columns([
            ((pl.col("lat").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lat"),
            ((pl.col("lon").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lon"),
        ])
        .group_by(["lat", "lon"])
        .agg([
            pl.col("lightning_3d_count").sum().alias("lightning"),
            pl.col("fire_3d_count").sum().alias("fires"),
            pl.col("mean_co").mean().alias("co"),
            pl.col("mean_pan").mean().alias("pan"),
        ])
        .filter(
            (pl.col("lat") >= 24) & (pl.col("lat") <= 50) &
            (pl.col("lon") >= -125) & (pl.col("lon") <= -66)
        )
    )

    lats = spatial["lat"].to_numpy()
    lons = spatial["lon"].to_numpy()
    lightning = spatial["lightning"].to_numpy()
    fires     = spatial["fires"].to_numpy()
    co        = spatial["co"].to_numpy()
    pan       = spatial["pan"].to_numpy()

    EXTENT = [-125, -66, 24, 50]
    proj   = ccrs.PlateCarree()

    from matplotlib.colors import Normalize

    # Percentile-clipped norms for CO and PAN — prevents extreme outliers
    # from washing out the colormap
    co_norm  = Normalize(vmin=np.percentile(co,  5), vmax=np.percentile(co,  95))
    pan_norm = Normalize(vmin=np.percentile(pan, 5), vmax=np.percentile(pan, 95))

    panels = [
        # (title, values, cmap, norm, cbar_label, mask_zeros)
        ("Lightning Events",    lightning, "inferno",  LogNorm(vmin=1, vmax=lightning.max()), "Total strikes (72h window)", True),
        ("Fire Activity",       fires,     "hot",      LogNorm(vmin=1, vmax=fires.max()),     "Fire count",                 True),
        ("CO Concentration",    co,        "plasma",   co_norm,                               "CO (ppb)",                   False),
        ("PAN Concentration",   pan,       "magma",    pan_norm,                              "PAN (ppbv)",                 False),
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    for i, (title, values, cmap, norm, cbar_label, mask_zeros) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        ax.set_facecolor("#0d1117")

        # Coastlines + borders
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       linewidth=0.7, edgecolor="#78909c")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       linewidth=0.6, edgecolor="#78909c")
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       linewidth=0.3, edgecolor="#546e7a")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor="#0a1929")
        ax.add_feature(cfeature.LAKES.with_scale("50m"),
                       facecolor="#0a1929", edgecolor="#37474f", linewidth=0.3)

        mask = (values > 0) if mask_zeros else np.ones(len(values), dtype=bool)
        sc = ax.scatter(
            lons[mask], lats[mask], c=values[mask],
            cmap=cmap, norm=norm, s=18, marker="s",
            transform=proj, zorder=5, linewidths=0)

        cb = plt.colorbar(sc, ax=ax, orientation="horizontal",
                          pad=0.04, fraction=0.046, shrink=0.85)
        cb.set_label(cbar_label, color=PALETTE["text"], fontsize=9)
        cb.ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=PALETTE["text"], pad=8)

    # Causal chain arrow annotation between subplots
    fig.text(0.5, 0.97,
             "The Causal Chain:  Lightning + Fires → CO plumes → PAN formation",
             ha="center", va="top", fontsize=14, fontweight="bold",
             color=PALETTE["text"])
    fig.text(0.5, 0.935,
             "September 2020 · CONUS · 0.5° grid · 72-hour causal lookback window",
             ha="center", va="top", fontsize=10.5, color="#90a4ae")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGURES_DIR / "fig6_conus_maps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Medium article figures...\n")
    fig_model_comparison()
    fig_sparsity()
    fig_log1p_transform()
    fig_architecture()
    fig_pred_vs_actual()
    fig_conus_maps()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
