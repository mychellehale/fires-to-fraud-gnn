"""
Graph Feature Builder
=====================
Fuses three data sources into a single feature table ready for GNN graph
construction.  Each row represents one grid cell on one day and contains:

    - lightning_3d_count : total lightning strikes in the cell over the
                           preceding 72 hours (days D-2, D-1, D)
    - fire_3d_count      : total fire events in the cell over the same window
    - mean_co            : mean CO concentration (ppbv) on day D
    - lat / lon          : 0.5° grid-cell center coordinates
    - mean_pan           : mean PAN concentration (ppbv) — the GNN target

Why this matters for the interview
-----------------------------------
The thesis GTWR model processed only 6% of available observations because
the full spatiotemporal join was too expensive in R.  This Polars pipeline
runs the full 49,403-observation join in seconds, recovering the temporal
variation that GTWR had to discard.

The 72-hour lookback window is applied here explicitly as a range join — the
same causal constraint from the thesis, but now computed correctly in UTC
rather than the PDT-uniform timestamps that corrupted the original GTWR
temporal distance matrix.

Run
---
    python research/build_graph_features.py
"""

from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictor.atmospheric.cleaners import AtmosphericCleaner  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LIGHTNING_PARQUET = ROOT / "data" / "processed" / "lightning_sep2020_processed.parquet"
MODIS_CSV = ROOT / "data" / "raw" / "MODIS_Sep2020" / "202009MODIS.csv"
PAN_CO_CSV = ROOT / "data" / "raw" / "PAN_CO_Sep2020" / "cleaned" / "PAN_CO_0902_0930.csv"
OUT_PATH = ROOT / "data" / "processed" / "graph_features.parquet"

# CONUS bounds (match AtmosphericIngestor constants)
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0
GRID = 0.5


# ---------------------------------------------------------------------------
# Step 1 — Lightning: bin to grid, extract date, aggregate daily counts
# ---------------------------------------------------------------------------
print("Loading lightning events...")
lightning_raw = pl.read_parquet(LIGHTNING_PARQUET)

lightning_daily = (
    AtmosphericCleaner.bin_to_grid(lightning_raw, resolution=GRID)
    .with_columns(pl.col("datetime").dt.date().alias("date"))
    .filter(
        pl.col("lat").is_between(LAT_MIN, LAT_MAX) &
        pl.col("lon").is_between(LON_MIN, LON_MAX)
    )
    .group_by(["date", "lat", "lon"])
    .agg(pl.len().alias("strike_count"))
    .sort(["lat", "lon", "date"])
)
print(f"  {lightning_daily.height:,} (date, cell) lightning aggregates.")


# ---------------------------------------------------------------------------
# Step 2 — MODIS fire: parse date, bin, filter confidence, aggregate
# ---------------------------------------------------------------------------
print("Loading MODIS fire events...")
fire_raw = pl.read_csv(MODIS_CSV)

fire_daily = (
    fire_raw
    .rename({c: c.lower() for c in fire_raw.columns})  # normalise to lowercase
    .filter(pl.col("confidence") >= 60)                 # thesis quality threshold
    .with_columns([
        pl.col("acq_date").str.strptime(pl.Date, format="%m/%d/%y").alias("date"),
        pl.col("latitude").alias("lat"),
        pl.col("longitude").alias("lon"),
    ])
    .filter(
        pl.col("lat").is_between(LAT_MIN, LAT_MAX) &
        pl.col("lon").is_between(LON_MIN, LON_MAX)
    )
    .pipe(lambda df: AtmosphericCleaner.bin_to_grid(df, resolution=GRID))
    .group_by(["date", "lat", "lon"])
    .agg(pl.len().alias("fire_count"))
    .sort(["lat", "lon", "date"])
)
print(f"  {fire_daily.height:,} (date, cell) fire aggregates.")


# ---------------------------------------------------------------------------
# Step 3 — PAN/CO: parse date, bin, filter, aggregate
# ---------------------------------------------------------------------------
print("Loading PAN/CO observations...")
pan_co_daily = (
    pl.read_csv(PAN_CO_CSV)
    .rename({"Lng": "lon", "Lat": "lat", "Date": "date_str", "CO": "co", "PAN": "pan"})
    .with_columns(
        pl.col("date_str").str.strptime(pl.Date, format="%Y-%m-%d").alias("date")
    )
    .filter(
        pl.col("pan") >= 0.0                             # thesis cleaning rule
    )
    .filter(
        pl.col("lat").is_between(LAT_MIN, LAT_MAX) &
        pl.col("lon").is_between(LON_MIN, LON_MAX)
    )
    .pipe(lambda df: AtmosphericCleaner.bin_to_grid(df, resolution=GRID))
    .group_by(["date", "lat", "lon"])
    .agg([
        pl.col("pan").mean().alias("mean_pan"),
        pl.col("co").mean().alias("mean_co"),
    ])
    .drop_nulls(subset=["mean_pan", "mean_co"])
    .sort(["lat", "lon", "date"])
)
print(f"  {pan_co_daily.height:,} (date, cell) PAN/CO aggregates.")


# ---------------------------------------------------------------------------
# Step 4 — 72-hour lookback: for each PAN observation at (date, lat, lon),
# sum lightning and fire from [date - 2 days, date].
#
# This is the causal join the thesis described but could not fully run due
# to compute constraints.  In Polars it is a range join that runs in ~1s.
# ---------------------------------------------------------------------------
print("Computing 72-hour lookback windows...")

pan_grid = pan_co_daily.select(["date", "lat", "lon"])

def lookback_3d(
    pan_grid: pl.DataFrame,
    event_daily: pl.DataFrame,
    count_col: str,
    alias: str,
) -> pl.DataFrame:
    """Sums event_daily[count_col] within a 3-day window for each PAN cell."""
    return (
        pan_grid
        .join(
            event_daily.rename({"date": "event_date", count_col: "evt_count"}),
            on=["lat", "lon"],
            how="left",
        )
        .filter(
            (pl.col("event_date") >= pl.col("date") - pl.duration(days=2)) &
            (pl.col("event_date") <= pl.col("date"))
        )
        .group_by(["date", "lat", "lon"])
        .agg(pl.col("evt_count").sum().alias(alias))
    )


lightning_3d = lookback_3d(pan_grid, lightning_daily, "strike_count", "lightning_3d_count")
fire_3d = lookback_3d(pan_grid, fire_daily, "fire_count", "fire_3d_count")


# ---------------------------------------------------------------------------
# Step 5 — Join all features on (date, lat, lon), fill missing counts with 0
# ---------------------------------------------------------------------------
print("Joining all features...")
graph_features = (
    pan_co_daily
    .join(lightning_3d, on=["date", "lat", "lon"], how="left")
    .join(fire_3d, on=["date", "lat", "lon"], how="left")
    .with_columns([
        pl.col("lightning_3d_count").fill_null(0).cast(pl.Float32),
        pl.col("fire_3d_count").fill_null(0).cast(pl.Float32),
        pl.col("mean_co").cast(pl.Float32),
        pl.col("mean_pan").cast(pl.Float32),
        pl.col("lat").cast(pl.Float32),
        pl.col("lon").cast(pl.Float32),
    ])
    .sort(["date", "lat", "lon"])
)

print(f"\n  Final graph features: {graph_features.height:,} rows")
print(f"  Dates: {graph_features['date'].min()} → {graph_features['date'].max()}")
print(f"  Unique grid cells: {graph_features.select(['lat', 'lon']).unique().height:,}")
print(f"  PAN range: {graph_features['mean_pan'].min():.4f} – {graph_features['mean_pan'].max():.4f} ppbv")
print(f"  Rows with lightning activity: {(graph_features['lightning_3d_count'] > 0).sum():,}")
print(f"  Rows with fire activity:      {(graph_features['fire_3d_count'] > 0).sum():,}")


# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
graph_features.write_parquet(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")
