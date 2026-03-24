"""
run_baselines.py — OLS, GWR, and GTWR on the modernized pipeline
=================================================================
Reruns the thesis baseline models using the 2026 cleaned data:
  - UTC timestamps (timezone bug fixed)
  - Full dataset (no 6% compute cap)
  - Correct TES pressure indices and QC filter
  - 1° grid for thesis-comparable results

Models
------
OLS   : sklearn LinearRegression, full monthly dataset
GWR   : mgwr GWR, monthly 1° spatial aggregation (~1,000 cells)
GTWR  : mgwr GTWR, daily 1° data, UTC-fixed timestamps, n=GTWR_N sample

Results are written to data/processed/baseline_results.txt as each
model completes, so partial results are captured if something times out.

Run
---
    python research/run_baselines.py          # all three models
    python research/run_baselines.py --ols    # OLS only (fast sanity check)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FEATURES_PATH  = ROOT / "data" / "processed" / "graph_features.parquet"
RESULTS_PATH   = ROOT / "data" / "processed" / "baseline_results.txt"

SEED      = 42
GRID      = 1.0       # 1° to match thesis
GTWR_N    = 6_000     # sample size for GTWR (thesis used 3,144)
COUNT_COLS = ["lightning_3d_count", "fire_3d_count"]
FEATURE_COLS = ["lightning_3d_count", "fire_3d_count", "mean_co", "lat", "lon"]
TARGET_COL   = "mean_pan"

THESIS = {
    "OLS (thesis)":    {"r2": 0.061, "rmse": 0.252, "mae": 0.073},
    "GWR (thesis)":    {"r2": 0.361, "rmse": 0.078, "mae": 0.030},
    "GTWR (thesis)":   {"r2": 0.227, "rmse": 0.184, "mae": 0.071},
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(line + "\n")


def log_metrics(label: str, r2: float, rmse: float, mae: float,
                elapsed: float | None = None) -> None:
    t = f"  elapsed: {elapsed:.1f}s" if elapsed else ""
    log(f"  {label:<28} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}{t}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------
def rebin(df: pl.DataFrame, grid: float = GRID) -> pl.DataFrame:
    half = grid / 2.0
    return df.with_columns([
        ((pl.col("lat").cast(pl.Float32) / grid).floor() * grid + half).alias("lat"),
        ((pl.col("lon").cast(pl.Float32) / grid).floor() * grid + half).alias("lon"),
    ])


def monthly_spatial(df: pl.DataFrame) -> pl.DataFrame:
    """One row per grid cell: sum lightning/fire, mean CO/PAN. log1p on counts."""
    return (
        rebin(df)
        .with_columns([pl.col(c).log1p().alias(c) for c in COUNT_COLS])
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


def daily_grid(df: pl.DataFrame) -> pl.DataFrame:
    """Daily rows at 1° grid, with log1p on counts. Keeps date column."""
    return (
        rebin(df)
        .with_columns([pl.col(c).log1p().alias(c) for c in COUNT_COLS])
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


def split_80_20(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def scale(X_train: np.ndarray, X_test: np.ndarray,
          y_train: np.ndarray, y_test: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    fx = StandardScaler().fit(X_train)
    fy = StandardScaler().fit(y_train.reshape(-1, 1))
    Xs_tr = fx.transform(X_train)
    Xs_te = fx.transform(X_test)
    ys_tr = fy.transform(y_train.reshape(-1, 1)).flatten()
    ys_te = fy.transform(y_test.reshape(-1, 1)).flatten()
    return Xs_tr, Xs_te, ys_tr, ys_te, fy


def inv(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def latlon_to_meters(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Equirectangular projection: degrees to approximate metres at CONUS centre."""
    lat0 = np.radians(37.0)   # centre of CONUS
    R    = 6_371_000.0
    x    = R * np.radians(lons) * np.cos(lat0)
    y    = R * np.radians(lats)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------
def run_ols(features: pl.DataFrame) -> None:
    log("\n" + "=" * 60)
    log("OLS — monthly 1° grid, full dataset")
    log("=" * 60)

    spatial = monthly_spatial(features)
    log(f"  Grid cells: {spatial.height:,}")

    X = spatial.select(FEATURE_COLS).to_numpy()
    y = spatial[TARGET_COL].to_numpy()

    tr_idx, te_idx = split_80_20(len(y))
    Xs_tr, Xs_te, ys_tr, ys_te, fy = scale(X[tr_idx], X[te_idx],
                                             y[tr_idx], y[te_idx])

    t0  = time.time()
    clf = LinearRegression().fit(Xs_tr, ys_tr)

    for idx_s, label, ys in [(tr_idx, "Train", ys_tr), (te_idx, "Test ", ys_te)]:
        Xs = Xs_tr if label == "Train" else Xs_te
        yp_s = clf.predict(Xs)
        yp   = inv(yp_s, fy)
        yt   = inv(ys,   fy)
        r2, rmse, mae = metrics(yt, yp)
        log_metrics(f"OLS 2026 ({label})", r2, rmse, mae,
                    elapsed=time.time() - t0 if label == "Test " else None)

    log(f"  Thesis OLS baseline:          R²=0.061  RMSE=0.252  MAE=0.073")


# ---------------------------------------------------------------------------
# GWR
# ---------------------------------------------------------------------------
def run_gwr(features: pl.DataFrame) -> None:
    log("\n" + "=" * 60)
    log("GWR — monthly 1° grid, mgwr adaptive bandwidth")
    log("=" * 60)

    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
    except ImportError:
        log("  mgwr not installed — skipping GWR")
        return

    spatial = monthly_spatial(features)
    log(f"  Grid cells: {spatial.height:,}")

    coords_deg = np.column_stack([
        spatial["lat"].to_numpy(),
        spatial["lon"].to_numpy(),
    ])
    coords = latlon_to_meters(coords_deg[:, 0], coords_deg[:, 1])

    X_raw = spatial.select(FEATURE_COLS).to_numpy()
    y_raw = spatial[TARGET_COL].to_numpy()

    tr_idx, te_idx = split_80_20(len(y_raw))
    Xs_tr, Xs_te, ys_tr, ys_te, fy = scale(
        X_raw[tr_idx], X_raw[te_idx], y_raw[tr_idx], y_raw[te_idx])

    # mgwr expects intercept as first column
    Xs_tr_i = np.hstack([np.ones((len(Xs_tr), 1)), Xs_tr])
    Xs_te_i = np.hstack([np.ones((len(Xs_te), 1)), Xs_te])

    log("  Selecting GWR bandwidth (adaptive bisquare, golden section)...")
    t0 = time.time()
    selector = Sel_BW(coords[tr_idx], ys_tr.reshape(-1, 1), Xs_tr_i,
                      fixed=False, kernel="bisquare")
    bw = selector.search(search_method="golden_section", criterion="AICc")
    log(f"  Optimal bandwidth: {bw:.1f} neighbours  ({time.time()-t0:.1f}s)")

    log("  Fitting GWR on train set...")
    gwr = GWR(coords[tr_idx], ys_tr.reshape(-1, 1), Xs_tr_i, bw,
              fixed=False, kernel="bisquare").fit()

    # Train metrics
    yp_tr = inv(gwr.predy.flatten(), fy)
    yt_tr = inv(ys_tr, fy)
    r2, rmse, mae = metrics(yt_tr, yp_tr)
    log_metrics("GWR 2026 (Train)", r2, rmse, mae)

    log("  Predicting on test set (local kernel at test coords)...")
    t1 = time.time()
    pred_te = gwr.model.predict(coords[te_idx], Xs_te_i)
    yp_te = inv(pred_te.predy.flatten(), fy)
    yt_te = inv(ys_te, fy)
    r2, rmse, mae = metrics(yt_te, yp_te)
    log_metrics("GWR 2026 (Test)", r2, rmse, mae, elapsed=time.time() - t1)

    log(f"  Thesis GWR baseline:          R²=0.361  RMSE=0.078  MAE=0.030")


# ---------------------------------------------------------------------------
# GTWR
# ---------------------------------------------------------------------------
def run_gtwr(features: pl.DataFrame) -> None:
    log("\n" + "=" * 60)
    log(f"GTWR — daily 1° grid, UTC-fixed timestamps, n={GTWR_N:,} sample")
    log("=" * 60)
    log("  Key change vs thesis: timezone bug fixed (UTC, not 12AM Pacific)")
    log(f"  Thesis ran on 3,144 obs; this run uses {GTWR_N:,}")

    try:
        from mgwr.gwr import GTWR
        from mgwr.sel_bw import Sel_BW
    except ImportError:
        log("  mgwr GTWR not available — skipping")
        return

    daily = daily_grid(features)
    log(f"  Total daily rows: {daily.height:,}")

    # Reproducible subsample
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(daily.height, size=min(GTWR_N, daily.height), replace=False)
    idx.sort()
    sample = daily[idx.tolist()]
    log(f"  Sample size: {sample.height:,}")

    coords_deg = np.column_stack([
        sample["lat"].to_numpy(),
        sample["lon"].to_numpy(),
    ])
    coords = latlon_to_meters(coords_deg[:, 0], coords_deg[:, 1])

    # Temporal coordinate: day-of-year (UTC, so distances are meaningful)
    dates    = sample["date"].to_numpy()
    doy      = np.array([(d.astype("datetime64[D]") - np.datetime64("2020-01-01", "D")).astype(float)
                          for d in dates])
    # Scale time to same order of magnitude as spatial (metres / 111,000 m per degree)
    t_coords = (doy / doy.max()) * 1_000_000   # normalise to ~1M metre range

    X_raw = sample.select(FEATURE_COLS).to_numpy()
    y_raw = sample[TARGET_COL].to_numpy()

    tr_idx, te_idx = split_80_20(len(y_raw))
    Xs_tr, Xs_te, ys_tr, ys_te, fy = scale(
        X_raw[tr_idx], X_raw[te_idx], y_raw[tr_idx], y_raw[te_idx])

    Xs_tr_i = np.hstack([np.ones((len(Xs_tr), 1)), Xs_tr])
    Xs_te_i = np.hstack([np.ones((len(Xs_te), 1)), Xs_te])

    t_tr = t_coords[tr_idx]
    t_te = t_coords[te_idx]

    log("  Selecting GTWR bandwidth (adaptive bisquare, golden section)...")
    log("  This is the slow step. Estimating 20-90 min depending on hardware.")
    t0 = time.time()

    selector = Sel_BW(coords[tr_idx], ys_tr.reshape(-1, 1), Xs_tr_i,
                      fixed=False, kernel="bisquare",
                      # GTWR-specific: pass temporal coordinates
                      spherical=False,
                      t_coords=t_tr.reshape(-1, 1))
    bw = selector.search(search_method="golden_section", criterion="AICc")
    tau = selector.optimum_tau  # spatial/temporal bandwidth ratio
    elapsed_bw = time.time() - t0
    log(f"  Optimal spatial bw: {bw:.1f}  tau: {tau:.4f}  ({elapsed_bw:.1f}s)")

    log("  Fitting GTWR on train set...")
    t1 = time.time()
    gtwr = GTWR(coords[tr_idx], ys_tr.reshape(-1, 1), Xs_tr_i,
                bw, tau, fixed=False, kernel="bisquare",
                t_coords=t_tr.reshape(-1, 1)).fit()

    yp_tr = inv(gtwr.predy.flatten(), fy)
    yt_tr = inv(ys_tr, fy)
    r2, rmse, mae = metrics(yt_tr, yp_tr)
    log_metrics("GTWR 2026 (Train)", r2, rmse, mae, elapsed=time.time() - t1)

    log("  Predicting on test set...")
    t2 = time.time()
    pred_te = gtwr.model.predict(coords[te_idx], Xs_te_i,
                                 t_coords=t_te.reshape(-1, 1))
    yp_te = inv(pred_te.predy.flatten(), fy)
    yt_te = inv(ys_te, fy)
    r2, rmse, mae = metrics(yt_te, yp_te)
    log_metrics("GTWR 2026 (Test)", r2, rmse, mae, elapsed=time.time() - t2)

    log(f"  Thesis GTWR baseline:         R²=0.227  RMSE=0.184  MAE=0.071")
    log(f"  Total GTWR elapsed:           {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ols",  action="store_true", help="OLS only")
    parser.add_argument("--gwr",  action="store_true", help="GWR only")
    parser.add_argument("--gtwr", action="store_true", help="GTWR only")
    args = parser.parse_args()

    run_all = not any([args.ols, args.gwr, args.gtwr])

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Run started: {datetime.now()}\n")
        f.write(f"{'='*60}\n")

    log(f"Loading features from {FEATURES_PATH}")
    features = pl.read_parquet(FEATURES_PATH)
    log(f"  {features.height:,} rows, {features['date'].n_unique()} dates, "
        f"{features.select(['lat','lon']).unique().height:,} unique cells")

    if run_all or args.ols:
        run_ols(features)

    if run_all or args.gwr:
        run_gwr(features)

    if run_all or args.gtwr:
        run_gtwr(features)

    log("\nAll done. Results in data/processed/baseline_results.txt")


if __name__ == "__main__":
    main()
