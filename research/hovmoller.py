"""
Hovmöller Diagram: Lightning Strike Density — September 2020 (CONUS)
=====================================================================

A Hovmöller diagram plots a geophysical field on two axes: latitude (y)
vs. time (x), with color representing intensity.  Here the field is daily
lightning strike density, binned to a 0.5° latitude grid.

Scientific rationale
--------------------
The thesis hypothesis is that lightning-induced NOx chemistry produces
detectable PAN plumes on a ~3-day timescale.  A Hovmöller lets us visually
inspect whether a burst of lightning activity at a given latitude precedes
elevated PAN by roughly 3 days — the causal lag encoded as the 72-hour
lookback window in the GNN pipeline.

If the hypothesis holds, the Hovmöller should show a repeating pattern of
lightning bursts that "lead" the PAN signal in time, confirming that the
lookback window is physically motivated rather than arbitrary.

Prerequisites
-------------
Run the lightning processing notebook first so that the processed parquet
exists at data/processed/lightning_sep2020_processed.parquet.

Steps
-----
1. Load processed lightning events from parquet (Polars → fast I/O).
2. Filter to CONUS latitude band (24°–50°N) to match the GNN spatial scope.
3. Bin lat/lon to a 0.5° grid — same resolution as the CO tracer data,
   which reduces point-level noise and enables a clean spatial join later.
4. Cap strike counts at the 99th percentile to prevent single super-cell
   convective events from collapsing the color scale.
5. Aggregate to daily strike counts per latitude bin.
6. Pivot to a 2D matrix: rows = latitude bins, columns = calendar days.
7. Plot with a log-normalized color scale because strike counts span orders
   of magnitude within a single day (one orbit may see 200k events; another 5).
8. Save to figures/ for reproducibility and version control.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polars as pl

# -- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictor.atmospheric.cleaners import AtmosphericCleaner  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "lightning_sep2020_processed.parquet"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# CONUS latitude bounds (matches AtmosphericIngestor)
LAT_MIN, LAT_MAX = 24.0, 50.0


# -- Step 1: Load -------------------------------------------------------------
# Parquet is columnar and Polars reads it without deserializing the whole file,
# so even 3M rows loads in under a second.
print("Loading processed lightning events...")
lightning = pl.read_parquet(PROCESSED)
print(f"  {lightning.height:,} events loaded.")


# -- Step 2: Filter to CONUS --------------------------------------------------
# The LIS instrument covers the full tropics; we restrict to CONUS to stay
# consistent with the thesis spatial scope and the GNN training region.
lightning = lightning.filter(pl.col("lat").is_between(LAT_MIN, LAT_MAX))
print(f"  {lightning.height:,} events after CONUS filter.")


# -- Step 3: Bin to 0.5° grid -------------------------------------------------
# Binning snaps each event to the center of its grid cell.  The 0.5° resolution
# matches the MOPITT CO tracer data, which is the target variable in the GNN.
# Without binning, tiny GPS offsets between instruments create false spatial
# mismatches during the graph construction join.
lightning = AtmosphericCleaner.bin_to_grid(lightning, resolution=0.5)


# -- Step 4: Extract date for grouping ----------------------------------------
# Truncate datetime to the day boundary so daily aggregation is unambiguous
# regardless of intra-day UTC timing of individual orbits.
lightning = lightning.with_columns(
    pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date")
)


# -- Step 5: Aggregate to daily strike count per latitude bin -----------------
daily_lat = (
    lightning
    .group_by(["date", "lat"])
    .agg(pl.len().alias("strike_count"))
    .sort(["date", "lat"])
)

# Step 4 (cap): clip at 99th percentile to prevent one massive convective cell
# from dominating the color scale and washing out the broader spatial pattern.
daily_lat = AtmosphericCleaner.cap_at_percentile(daily_lat, "strike_count", percentile=0.99)


# -- Step 6: Pivot to 2D matrix -----------------------------------------------
# Rows = latitude bin centers, columns = calendar days (sorted).
# Cells with no observed events become null → fill with 0.
pivot = (
    daily_lat
    .pivot(values="strike_count", index="lat", on="date", aggregate_function="sum")
    .sort("lat")
)

date_cols = sorted(c for c in pivot.columns if c != "lat")
lat_vals = pivot["lat"].to_numpy()
matrix = pivot.select(date_cols).to_numpy().astype(float)
matrix = np.nan_to_num(matrix, nan=0.0)

print(f"  Hovmöller matrix: {matrix.shape[0]} lat bins × {matrix.shape[1]} days")


# -- Step 7: Plot -------------------------------------------------------------
# Log normalization: a linear scale would make the 99th-percentile cap value
# dominate and render everything else the same shade.  Log scale lets the
# spatial structure within lower-activity days remain visible.
fig, ax = plt.subplots(figsize=(15, 7))

x = np.arange(len(date_cols))
y = lat_vals

nonzero_min = matrix[matrix > 0].min() if (matrix > 0).any() else 1.0
norm = mcolors.LogNorm(vmin=nonzero_min, vmax=matrix.max())

mesh = ax.pcolormesh(x, y, matrix, cmap="YlOrRd", norm=norm, shading="auto")

# Colorbar
cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Daily Strike Count (log scale)", fontsize=11)

# Axis labels
ax.set_xlabel("Date (September 2020)", fontsize=12)
ax.set_ylabel("Latitude (°N)", fontsize=12)
ax.set_title(
    "Hovmöller Diagram — CONUS Lightning Strike Density, September 2020\n"
    "0.5° lat bins  ·  99th-percentile capped  ·  log color scale",
    fontsize=13,
    pad=14,
)

# x-tick formatting: show every 3rd day to avoid crowding
step = 3
tick_idx = list(range(0, len(date_cols), step))
ax.set_xticks(tick_idx)
ax.set_xticklabels([date_cols[i][5:] for i in tick_idx], rotation=45, ha="right")

# Reference line: mean active-fire latitude for September 2020 (CONUS ~38°N)
# Drawn as a dashed line so reviewers can visually anchor the lightning band
# to the known fire activity corridor.
ax.axhline(y=38.0, color="steelblue", linestyle="--", linewidth=1.0,
           label="Approx. mean fire latitude (~38°N)")
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()


# -- Step 8: Save -------------------------------------------------------------
out = FIGURES / "hovmoller_lightning_sep2020.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
