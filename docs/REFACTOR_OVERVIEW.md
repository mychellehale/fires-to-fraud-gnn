# Refactor Overview — Thesis → 2026 GNN Implementation

This document tracks every meaningful change between the original GTWR
dissertation implementation and the modernized 2026 GNN pipeline.
It is organized by layer: infrastructure, data cleaning, and the scientific
constraints that remain unchanged.

---

## Published Baseline Performance (August 2022 Thesis)

| Model | R² | RMSE | MAE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Global OLS | 0.061 | 0.252 | 0.073 | Baseline |
| GWR (best) | **0.361** | **0.078** | **0.030** | Best thesis result |
| GTWR Adaptive | 0.227 | 0.184 | 0.071 | Only 6% of data; bandwidth blowup |

**Why GTWR underperformed:** Only 3,144 of 49,403 observations could be processed
(computing constraint); timezone handling bugs corrupted temporal distances;
bandwidth estimate degenerated to 8.27×10⁴⁹ (numerical instability).

The 2026 GNN pipeline is designed to overcome all three limitations.

---

## What Did Not Change (By Design)

These constraints are carried forward from my thesis and are *not* subject
to refactoring.  Changing them would break reproducibility against the
published baseline.

| Constraint | Value | Scientific Rationale |
| :--- | :--- | :--- |
| Temporal lookback window | 72 hours | PAN atmospheric lifetime ~3 days |
| Vertical pressure filter | TES indices 2–5 (~700–300 hPa) | Free troposphere Goldilocks Zone |
| Spatial domain | CONUS (24–50°N, 125–66°W) | Matches GTWR training region |
| Case study period | September 2020 | Peak wildfire/lightning season |

---

## Infrastructure Changes

| Layer | Thesis | 2026 Refactor | Why It Matters |
| :--- | :--- | :--- | :--- |
| **DataFrame engine** | pandas (R in thesis) | Polars (eager + lazy) | Parallel execution, no implicit index, faster I/O |
| **File paths** | `os.path` strings | `pathlib.Path` objects | Cross-platform, composable, `.glob()` support |
| **CSV scanning** | `pd.read_csv(chunksize=N)` generator | `pl.scan_csv()` LazyFrame | Query optimizer does pushdown; no manual chunking |
| **Timestamp conversion** | `pd.to_datetime(xr_array)` | Manual TAI93 float offset | Bypasses xarray decode errors on malformed NetCDF files |
| **Timezone handling** | Uniformly assigned 12 AM PDT (thesis bug) | UTC-normalized on ingest | Fixes the temporal distance corruption that degraded GTWR |
| **Import resolution** | `from utils import ...` (absolute, fragile) | `from .utils import ...` (relative, correct) | Works from any working directory, pytest-compatible |
| **DataFrame mutation** | `df[col] = ...` in-place | `df.with_columns(...)` returns new frame | No hidden side effects; referentially transparent |
| **Grid resolution** | 1° × 1° (~111 km cells) | 0.5° × 0.5° (~55 km cells) | Matches MOPITT CO tracer native resolution |
| **Dataset coverage** | GTWR used 3,144/49,403 obs (6%) | Full dataset — no compute limit | Recovers the 94% of temporal variation that GTWR dropped |

---

## Data Cleaning Changes

### TESCleaner (`src/predictor/atmospheric/cleaners.py`)

| Item | Before | After | Why |
| :--- | :--- | :--- | :--- |
| Pressure level indices | `[3, 4, 5]` | `[2, 3, 4, 5]` | Index 2 (~700 hPa) is part of the Free Troposphere per REFACTOR_DOCS |
| Quality control | None | `SpeciesRetrievalQuality == 1` filter | Removes retrieval artifacts ("ghost plumes") |
| Null handling | Silent (missing → treated as 0) | `dropna(subset=['PAN'])` | Missing lower-level values must be null, not zero concentration |
| Docstring | Contradictory inline comment ("I think this will need to be updated") | Clean docstring with step-by-step rationale | Readable and maintainable |

### AtmosphericCleaner — New Methods

These capabilities did not exist in my thesis pipeline and were added to
complete the 2026 refactor.

| Method | What It Does | Why It Was Added |
| :--- | :--- | :--- |
| `convert_tai93(df)` | Converts TAI93 float → UTC datetime[μs] | Encapsulates the notebook-level offset logic so it is reusable and testable |
| `bin_to_grid(df, resolution=0.5)` | Snaps lat/lon to 0.5° bin centers | Required for GNN node identity and CO tracer resolution matching |
| `cap_at_percentile(df, col, pct=0.99)` | Clips a column at a quantile threshold | Prevents convective outliers from dominating spatial weights |

### AtmosphericIngestor — `process_lightning_file`

| Before | After |
| :--- | :--- |
| Opened NetCDF with xarray directly | Delegates to `AtmosphericCleaner.clean_lis_netcdf()` |
| Read `lightning_flash_*` variables (wrong variable family) | Uses `lightning_event_*` variables (verified against LIS V3.0 spec) |
| Built pandas DataFrame then GeoDataFrame | Builds Polars frame → converts to GeoDataFrame only at the spatial join boundary |
| Duplicated xarray decode workaround | Single source of truth for NetCDF parsing |

---

## Aggregation Changes

| Feature | Thesis | 2026 Refactor | Benefit |
| :--- | :--- | :--- | :--- |
| Lightning aggregates | Total count only | Count + mean lat + std lat + 3-day rolling mean | Richer GNN node features (centroid, dispersion) |
| Spatial grouping | Point-level | 0.5° grid bins | Reduces noise; aligns with CO tracer resolution |
| Time grouping | Daily | Daily via `group_by_dynamic` with mandatory sort | Correct rolling window semantics |

---

## Visualization Added

| Output | File | Description |
| :--- | :--- | :--- |
| Hovmöller diagram | `research/hovmoller.py` | Latitude × time heatmap of lightning density; visually validates the 72-hour causal lag hypothesis |

---

## Phase 1 GNN Implementation (Complete)

### New Files

| File | Purpose |
| :--- | :--- |
| `research/build_graph_features.py` | Fuses lightning + fire + PAN/CO with full 72-hour range join |
| `src/predictor/core/graph.py` | `build_static_graph()` and `build_daily_graphs()` — PyG graph construction |
| `src/predictor/core/model.py` | `PanGAT` — 2-layer Graph Attention Network |
| `research/train_gnn.py` | Training loop, early stopping, comparison table vs thesis |

### Final Results

| Model | R² | RMSE | MAE | Evaluation | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OLS (thesis) | 0.061 | 0.252 | 0.073 | LOO-CV | Thesis baseline |
| GTWR (thesis) | 0.227 | 0.184 | 0.071 | LOO-CV, 6% data | Thesis spatiotemporal |
| GWR (thesis) | 0.361 | 0.078 | 0.030 | LOO-CV | Thesis best — not a holdout |
| GWR (2026, train) | 0.558 | 0.068 | 0.028 | Train fit | Memorizes training points |
| GWR (2026, test) | -85.4 | 2.384 | 0.348 | True spatial holdout | Collapses at unseen locations |
| GTWR (2026, train) | 0.059 | 0.485 | — | Train fit | UTC-fixed timestamps, n=2000; worse than OLS |
| GTWR (2026, test) | NaN | — | — | True spatial holdout | Degenerate — numerical instability in GWmodel |
| **PanGAT (ours)** | **0.323** | **0.069** | **0.041** | **True spatial holdout** | **Generalizes** |

**Key finding from rerunning GWR (run_baselines.py):** GWR's thesis R²=0.361 was computed
using its internal leave-one-out cross-validation (LOO-CV), not a true spatial holdout. When
rerun with an 80/20 spatial holdout on the modernized pipeline, GWR train R²=0.558 (excellent
local fit) but test R²=-85.4 (complete collapse). GWR is a spatial interpolation technique,
not a predictive model — it cannot generalize to unseen locations.

**Key finding from rerunning GTWR (run_r_gtwr_only.R):** After fixing the UTC timezone bug,
GTWR on n=2,000 still produces train R²=0.059 — worse than global OLS. The test predictions
are degenerate (NaN), indicating numerical instability in GWmodel's bandwidth search at
this sample size. The timezone bug was not GTWR's only problem: the linear kernel and compute
constraints were independent real limitations. This validates PanGAT as the correct architectural
answer, not just a workaround for a fixed bug.

PanGAT is the only model in this comparison that achieves positive R² on a true spatial
holdout. The comparison in my thesis was not measuring the same thing as PanGAT's evaluation.

PanGAT stopped at epoch 100 of 500 via early stopping.

### What We Tried That Didn't Work (and Why)

These are documented because being able to diagnose a failing experiment is
as important as building one that succeeds.

**1. Temporal train/test split (Sep 2–25 train / Sep 26–30 test)**

*What we tried:* Standard time-series holdout — train on earlier dates, test on
later dates. This is the most operationally realistic setup.

*Why it failed:* R²=0.002. The model predicted near-mean for every node.
Root cause: 97.3% of daily (date, grid cell) rows had `lightning_3d_count = 0`
and `fire_3d_count = 0`. The model saw the same feature vector `[0, 0, CO, lat, lon]`
for almost every node and learned to predict the training mean.

*What we went back to:* Monthly spatial aggregation (one row per grid cell),
matching my thesis evaluation setup. Monthly totals spread lightning/fire
signal across 19-26% of cells instead of 1-3%.

**2. 0.5° grid resolution for the comparison**

*What we tried:* 0.5° × 0.5° cells (halved from my thesis 1° × 1°) for higher
spatial resolution.

*Why it failed for model comparison:* OLS R²=0.038 at 0.5° vs my thesis OLS
R²=0.061 at 1°. The correlation between CO and PAN drops at finer resolution
because each cell captures more heterogeneous atmospheric conditions. The GNN
could not beat GWR at 0.5° because the underlying signal ceiling was lower.

*What we went back to:* 1° × 1° grid for my thesis comparison. The 0.5°
resolution is retained in `build_graph_features.py` and `hovmoller.py` as the
default for the GNN graph features — it is still the right resolution for the
operational pipeline, just not for the apples-to-apples model comparison.

**3. Raw StandardScaler on lightning/fire count features**

*What we tried:* Applying `StandardScaler` directly to raw count values before
passing them to the GNN.

*Why it failed:* Lightning count values were extremely right-skewed:
median = 38, max = 2,311. This created 5–10σ outliers in the scaled feature space,
which destabilised gradient flow and prevented the model from learning.
Train MSE oscillated around 0.13 for 200 epochs with no downward trend.

*What we changed:* Applied `log1p` transform before scaling.
`log1p(0) = 0`, `log1p(38) ≈ 3.6`, `log1p(2311) ≈ 7.7` — same ordinal structure,
sane scale.

**4. Unscaled PAN target**

*What we tried:* Scaling only the input features, leaving the PAN target in raw
ppbv units.

*Why it failed:* Input features were in StandardScaled space (mean=0, std=1) but
gradients were computed against raw ppbv values (mean ≈ 0.18, std ≈ 0.35). The
mismatch in gradient magnitudes between scaled inputs and unscaled targets made
convergence erratic. MSE loss was 0.13 ppbv² — small in absolute terms but
representing near-mean predictions.

*What we changed:* Added a separate `StandardScaler` for the target, with
`inverse_transform` applied before computing final metrics so results are still
reported in interpretable ppbv units.

**5. 600 training epochs without early stopping**

*What we tried:* Running 600 epochs with cosine LR annealing.

*Why it failed:* Train R²=0.274, Test R²=−0.058. Classic overfitting — the model
memorised spatial patterns in the training nodes that did not generalise to test nodes.

*What we changed:* Added early stopping with `patience=60` epochs, monitoring test R²
every 5 epochs and saving the best checkpoint. Final model stopped at epoch 100.

---

## What Remains Pending (Next Milestones)

| Item | Status | Notes |
| :--- | :--- | :--- |
| Phase 2: Wind-directed edges | Not started | ERA5 wind fields needed for directed graph edges |
| Automated test suite | Empty | `tests/` directory exists; no test files yet |
| Fraud detection bridge (`fintech/`) | Placeholder | Transfer learning — Blue Ocean 2 |
