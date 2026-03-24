"""
export_for_r.py — Export preprocessed data to CSV for R baseline runs
======================================================================
Applies the same preprocessing as run_baselines.py / train_gnn.py,
then writes CSV files R can read directly.

Outputs (data/processed/):
    gwr_train.csv       monthly 1° spatial, 80% split, log1p + scaled
    gwr_test.csv        monthly 1° spatial, 20% split, log1p + scaled
    gtwr_sample.csv     daily 1° data, 6000-row sample, log1p + scaled
    r_scaler_params.csv mean/std for inverse-transforming R predictions

Run:
    python research/export_for_r.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FEATURES_PATH = ROOT / "data" / "processed" / "graph_features.parquet"
OUT_DIR       = ROOT / "data" / "processed"

SEED         = 42
GRID         = 1.0
GTWR_N       = 2_000
COUNT_COLS   = ["lightning_3d_count", "fire_3d_count"]
FEATURE_COLS = ["lightning_3d_count", "fire_3d_count", "mean_co", "lat", "lon"]
TARGET_COL   = "mean_pan"


def rebin(df: pl.DataFrame) -> pl.DataFrame:
    half = GRID / 2.0
    return df.with_columns([
        ((pl.col("lat").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lat"),
        ((pl.col("lon").cast(pl.Float32) / GRID).floor() * GRID + half).alias("lon"),
    ])


def apply_log1p(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).log1p().alias(c) for c in COUNT_COLS])


def split_80_20(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def main() -> None:
    print("Loading features...")
    features = pl.read_parquet(FEATURES_PATH)

    # -----------------------------------------------------------------------
    # Monthly spatial (for GWR)
    # -----------------------------------------------------------------------
    print("Building monthly spatial data for GWR...")
    spatial = (
        rebin(features)
        .pipe(apply_log1p)
        .group_by(["lat", "lon"])
        .agg([
            pl.col("lightning_3d_count").sum(),
            pl.col("fire_3d_count").sum(),
            pl.col("mean_co").mean(),
            pl.col(TARGET_COL).mean(),
        ])
        .drop_nulls(TARGET_COL)
        .sort(["lat", "lon"])
    )
    print(f"  Grid cells: {spatial.height:,}")

    X = spatial.select(FEATURE_COLS).to_numpy()
    y = spatial[TARGET_COL].to_numpy()
    tr_idx, te_idx = split_80_20(len(y))

    fx = StandardScaler().fit(X[tr_idx])
    fy = StandardScaler().fit(y[tr_idx].reshape(-1, 1))

    def to_df(idx: np.ndarray, split: str) -> pl.DataFrame:
        Xs = fx.transform(X[idx])
        ys = fy.transform(y[idx].reshape(-1, 1)).flatten()
        lats = spatial["lat"].to_numpy()[idx]
        lons = spatial["lon"].to_numpy()[idx]
        return pl.DataFrame({
            "lat":                lats,
            "lon":                lons,
            "lightning_3d_count": Xs[:, 0],
            "fire_3d_count":      Xs[:, 1],
            "mean_co":            Xs[:, 2],
            "lat_scaled":         Xs[:, 3],
            "lon_scaled":         Xs[:, 4],
            "mean_pan_scaled":    ys,
            "mean_pan_raw":       y[idx],
            "split":              [split] * len(idx),
        })

    gwr_train = to_df(tr_idx, "train")
    gwr_test  = to_df(te_idx, "test")

    gwr_train.write_csv(OUT_DIR / "gwr_train.csv")
    gwr_test.write_csv(OUT_DIR / "gwr_test.csv")
    print(f"  Saved gwr_train.csv ({len(tr_idx):,} rows)")
    print(f"  Saved gwr_test.csv  ({len(te_idx):,} rows)")

    # -----------------------------------------------------------------------
    # Daily grid (for GTWR)
    # -----------------------------------------------------------------------
    print("\nBuilding daily grid data for GTWR...")
    daily = (
        rebin(features)
        .pipe(apply_log1p)
        .group_by(["date", "lat", "lon"])
        .agg([
            pl.col("lightning_3d_count").sum(),
            pl.col("fire_3d_count").sum(),
            pl.col("mean_co").mean(),
            pl.col(TARGET_COL).mean(),
        ])
        .drop_nulls(TARGET_COL)
        .sort(["date", "lat", "lon"])
    )
    print(f"  Total daily rows: {daily.height:,}")

    rng    = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(daily.height, size=min(GTWR_N, daily.height), replace=False))
    sample = daily[sample_idx.tolist()]

    X_d = sample.select(FEATURE_COLS).to_numpy()
    y_d = sample[TARGET_COL].to_numpy()
    tr_d, te_d = split_80_20(len(y_d))

    fx_d = StandardScaler().fit(X_d[tr_d])
    fy_d = StandardScaler().fit(y_d[tr_d].reshape(-1, 1))

    # Day-of-year as temporal coordinate (UTC — timezone bug fixed)
    dates = sample["date"].to_numpy()
    doy   = np.array([
        (d.astype("datetime64[D]") - np.datetime64("2020-01-01", "D")).astype(float)
        for d in dates
    ])

    Xs_d = fx_d.transform(X_d)
    ys_d = fy_d.transform(y_d.reshape(-1, 1)).flatten()

    lats_d = sample["lat"].to_numpy()
    lons_d = sample["lon"].to_numpy()

    gtwr_df = pl.DataFrame({
        "lat":                lats_d,
        "lon":                lons_d,
        "doy":                doy,
        "lightning_3d_count": Xs_d[:, 0],
        "fire_3d_count":      Xs_d[:, 1],
        "mean_co":            Xs_d[:, 2],
        "lat_scaled":         Xs_d[:, 3],
        "lon_scaled":         Xs_d[:, 4],
        "mean_pan_scaled":    ys_d,
        "mean_pan_raw":       y_d,
        "split":              ["train" if i in set(tr_d.tolist()) else "test"
                               for i in range(len(y_d))],
    })
    gtwr_df.write_csv(OUT_DIR / "gtwr_sample.csv")
    print(f"  Saved gtwr_sample.csv ({len(sample_idx):,} rows, "
          f"{int(len(tr_d))} train / {int(len(te_d))} test)")

    # -----------------------------------------------------------------------
    # Scaler params (for inverse-transforming R predictions back to ppbv)
    # -----------------------------------------------------------------------
    params = pl.DataFrame({
        "dataset":   ["gwr",              "gtwr"],
        "pan_mean":  [float(fy.mean_[0]), float(fy_d.mean_[0])],
        "pan_std":   [float(fy.scale_[0]),float(fy_d.scale_[0])],
    })
    params.write_csv(OUT_DIR / "r_scaler_params.csv")
    print(f"\nSaved r_scaler_params.csv")
    print("\nDone. Ready for R.")


if __name__ == "__main__":
    main()
