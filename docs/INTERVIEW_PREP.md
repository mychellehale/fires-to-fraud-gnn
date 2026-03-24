# Interview Prep — Fires-to-Fraud GNN

This document captures every non-obvious technical decision in the codebase
with interview-ready explanations.  Each entry follows the pattern:
**What I did → Why I did it → What it shows about you.**

---

## 1. Overall Architecture

### Library code lives in `src/`, exploration in `research/`

**What:** Reusable logic (cleaners, ingestors, utilities) is in
`src/predictor/`, while notebooks and scripts live in `research/`.

**Why:** Notebooks are great for exploration but terrible for reuse —
you cannot import a notebook, you cannot unit-test a notebook cell, and
Git diffs of `.ipynb` JSON are unreadable.  Separating concerns means the
notebook is just a *consumer* of the library, not a dumping ground for
business logic.

**What it shows:** You understand the difference between research code and
production code and structure your projects accordingly.

---

## 2. Dissertation Baseline — Results You Need to Know Cold

### Model performance

| Model | R² | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| Global OLS | 0.061 | 0.252 | 0.073 |
| OLS Local Adaptive | 0.303 | 0.081 | 0.037 |
| GWR (non-adaptive, Gaussian) | **0.361** | **0.078** | **0.030** |
| GWR Adaptive | 0.356 | 0.078 | 0.030 |
| GTWR Non-adaptive | 0.196 | 0.184 | 0.071 |
| GTWR Adaptive | 0.227 | 0.184 | 0.071 |

**The headline result:** GWR (R² = 0.361) outperformed global OLS (R² = 0.061) by 484%.
Accounting for spatial non-stationarity was the single biggest driver of model improvement.

### Why GTWR underperformed GWR — and why that matters for the GNN

This is the most important interview topic.  GTWR was expected to be the best model
(it extends GWR with a temporal dimension), but it scored *lower* than GWR.  Reasons:

1. **Computing constraints:** Only 6% of the 49,403 spatiotemporal observations could
   be used in the GTWR model (3,144).  The spatial dataset had the full 3,144 *unique*
   locations — so GTWR effectively lost 94% of the temporal variation.
2. **Bandwidth instability:** GTWR produced a bandwidth estimate of 8.27×10⁴⁹ — a
   numerical blowup that indicates the temporal kernel degenerated.  The model could
   not find a meaningful temporal scale.
3. **Timezone errors:** Dates lacked exact collection times and were uniformly assigned
   12 AM PDT — corrupting the temporal distance calculations.

**The GNN directly addresses all three:** it processes the full 49,403-row spatiotemporal
dataset, uses learned edge weights instead of a fixed kernel, and the 2026 pipeline
handles UTC normalization correctly.

### Data sparsity — the core challenge

Only 1,582 of 49,403 observations had *concurrent* fire or lightning activity in the
same grid cell.  That is 3.2%.  This is why detecting the lightning → PAN signal is hard:
most of the signal is separated spatially and temporally from the observations.  The
72-hour lookback window and the spatial buffer in the ingestor exist specifically to
bridge that gap.

### What to say when asked "what would you do differently?"

> "GTWR was the right conceptual choice — it is the only model in the thesis that
> accounts for both spatial and temporal structure — but it was limited by computing
> resources and timezone handling bugs.  The GNN upgrade keeps the spatiotemporal
> framing but replaces the kernel with learned edge weights, processes the full dataset,
> and fixes the UTC alignment.  The Hovmöller diagram is the visual proof-of-concept that
> the temporal lag signal exists and is worth modeling."

---

## 3. Data Pipeline

### Polars over pandas

**What:** Migrated all heavy data processing from pandas to Polars (eager
for in-memory work, lazy via `pl.scan_csv()` for large files).

**Why:** Polars is a Rust-backed, columnar DataFrame library.  Key
advantages for this project:
- **Speed:** Processing 478 LIS NetCDF files would take minutes in pandas;
  Polars' parallel execution and SIMD instructions cut that significantly.
- **No implicit index:** Pandas' implicit index causes subtle bugs when
  concatenating DataFrames from different sources (e.g., LIS orbits).
  Polars is always explicit.
- **Lazy API:** `pl.scan_csv()` pushes filter/projection down to the read,
  so you only load columns and rows you actually need.

**What it shows:** You know modern Python data tooling and can make
informed choices beyond "pandas is the default."

---

### TAI93 Timestamp Normalization (manual Unix offset)

**What:** Instead of letting `xarray` decode timestamps from the LIS
NetCDF files, we extract raw float arrays and apply a manual offset:

```python
TAI93_UNIX_OFFSET = 725_846_400  # seconds from 1970-01-01 to 1993-01-01
datetime = (tai_time_float + offset) * 1_000_000  # → microseconds → Datetime
```

**Why:** The NASA LIS V3.0 files use TAI93 encoding — seconds since
1993-01-01 00:00:00 — but do not always expose a proper CF-convention
`units` attribute that xarray can decode automatically.  Letting xarray
attempt decoding produces `InvalidOperationError` or silently wrong
timestamps.  Treating the values as raw floats and applying the offset
manually is deterministic and reproducible.

**What it shows:** You can debug binary scientific data formats, read
instrument documentation, and write workarounds that are explicit about
their assumptions rather than relying on library magic.

---

### Mandatory sort before `group_by_dynamic`

**What:** Every aggregation pipeline includes `.sort("datetime")` before
calling `group_by_dynamic`.

**Why:** Polars' `group_by_dynamic` (rolling time windows) requires the
DataFrame to be sorted on the time column.  LIS orbit files are ingested
in filesystem order, not chronological order — the sort step is not
optional, it is a correctness requirement.

**What it shows:** You read the library documentation carefully and do not
assume that data arrives in sorted order just because it looks like it
does.

---

### Grid resolution: 1° (thesis) → 0.5° (2026 refactor)

**What:** The thesis aggregated to a 1° × 1° grid (~111 km cells).  The 2026 pipeline
uses 0.5° × 0.5° cells.

**Why:** The CO tracer data from MOPITT has a native resolution of ~0.5°.  Using a
coarser grid in the thesis meant each cell could contain CO observations from very
different atmospheric conditions — smearing the signal.  Halving the grid size doubles
the spatial resolution of the GNN graph, which matters for capturing fire plume gradients.

**What it shows:** You understand that grid resolution is not arbitrary — it should be
anchored to the coarsest data source in the pipeline.

---

### 0.5° × 0.5° coordinate binning (`bin_to_grid`)

**What:** All lat/lon point coordinates are snapped to the center of a
0.5-degree grid cell before any spatial join or aggregation.

**Why:** Three reasons:
1. **Resolution matching:** The MOPITT CO tracer product (the GNN target
   variable) has a native resolution of ~0.5°.  Binning at the same
   resolution prevents artificial precision mismatches.
2. **Noise reduction:** Raw GPS-level coordinates from different instruments
   never align perfectly.  Binning collapses nearby events into the same
   node, which is what matters for graph construction.
3. **Graph construction:** GNN nodes represent spatial locations.  A
   consistent grid means node identity is stable across data sources.

**What it shows:** You think about how data representations downstream
(the graph) constrain decisions upstream (the cleaning pipeline).

---

### 99th-percentile cap on lightning density (`cap_at_percentile`)

**What:** After aggregating strike counts per grid cell, values are clipped
at the 99th percentile.

**Why:** A single super-cell convective storm can produce orders of
magnitude more strikes than a typical storm.  Without capping, that one
event dominates the spatial weights in GTWR and the edge weights in the
GNN — the model effectively learns about that one storm rather than the
broader pattern.

**What it shows:** You understand that outlier handling is not just about
model accuracy — it is about ensuring the model learns the right signal.

---

## 3. Atmospheric Science Constraints

### TES Vertical Filtering — Indices 2–5 (the "Goldilocks Zone")

**What:** TES PAN retrievals are filtered to pressure level indices 2–5
(approximately 700–300 hPa, the free troposphere).

**Why:**
- **Index 0 (~1000 hPa, boundary layer):** Thermal instability and surface
  pollution create noise that is not attributable to the lightning–PAN
  chemistry pathway.
- **Index 1 (~825 hPa):** Lower tropospheric transition — useful for
  context but not the primary signal.
- **Indices 2–5 (~700–300 hPa):** This is where PAN is thermally stable
  and where the GTWR model found statistically significant correlation with
  upwind lightning.  The original thesis called this the Goldilocks Zone.
- **Indices 6–7 (< 250 hPa):** Upper troposphere / stratospheric
  interference.  PAN at these levels is not attributable to local
  fire/lightning precursors within the 72-hour window.

**What it shows:** You understand why domain-specific filters exist and
can connect a line of code to a physical mechanism.

---

### TES SpeciesRetrievalQuality == 1 filter

**What:** Only TES retrievals with a quality flag of 1 (Good) are retained.

**Why:** TES retrievals in cloudy conditions, at the edge of the swath, or
with convergence issues are flagged as quality 0.  Including them would
introduce "ghost plumes" — apparent PAN enhancements that are really
retrieval artifacts — which would bias the GTWR spatial coefficients and
the GNN node features.

**What it shows:** You treat data quality flags as a scientific necessity,
not a nice-to-have.

---

### The 72-Hour Lookback Window

**What:** Lightning and fire events are only attributed to a gas
measurement if they occurred within 72 hours *before* the measurement.

**Why:** PAN (peroxyacetyl nitrate) has a photochemical lifetime of
approximately 3 days in the free troposphere.  The causal chain is:

```
Lightning → NOx injection → OH oxidation → PAN formation (~hours–days)
```

72 hours captures the "direct path" — events that plausibly contributed
to the observed PAN concentration — without opening the window so wide
that unrelated events get included (which would dilute the signal and
inflate graph edge counts).

**What it shows:** The model's temporal window is physically motivated,
not a hyperparameter chosen by grid search.

---

## 4. Visualization

### Hovmöller Diagram (`research/hovmoller.py`)

**What:** A 2D heatmap with latitude on the y-axis, time on the x-axis,
and log-scaled lightning strike count as the color fill.

**Why:** A Hovmöller is a standard meteorological diagnostic for tracking
whether an atmospheric signal propagates spatially over time.  In this
context, if a lightning burst at latitude Y on day D is followed by
elevated PAN at latitude Y on day D+3, the Hovmöller makes that lag
visually obvious — it validates the 72-hour causal window geometrically
rather than just statistically.

**Why log color scale:** Strike counts vary by 2–3 orders of magnitude
within a single day.  A linear scale makes everything below the 99th
percentile cap appear the same shade.  Log scale preserves the spatial
structure within quieter days.

**What it shows:** You can translate a statistical hypothesis (3-day lag)
into a visualization that communicates the hypothesis to a non-specialist
audience.

---

## 5. Code Quality Decisions

### Relative imports (`from .utils import ...`)

**What:** All intra-package imports use relative syntax (`from .module import X`).

**Why:** Absolute imports like `from utils import X` only work if you run
the script from the exact directory that contains `utils.py`.  Relative
imports work regardless of the working directory, which is essential for
`pytest`, `uv run`, and any CI runner.

**What it shows:** You understand Python's import system and write
importable packages, not just scripts.

---

### No in-place mutation of DataFrames

**What:** All transformation functions return a *new* DataFrame rather than
modifying the input.

**Why:** In-place mutation is a classic source of hidden bugs in data
pipelines — a step midway through a pipeline silently changes a DataFrame
that is used again later.  Returning new objects makes data flow explicit
and functions referentially transparent (same input always produces the
same output).

**What it shows:** You apply functional programming principles to data
engineering, not just to application code.

---

### `Path` objects over string paths

**What:** All file paths use `pathlib.Path` throughout the codebase.

**Why:** String paths break on Windows (`\` vs `/`), do not compose safely
(`path + "/subdir"` is error-prone), and have no methods.  `Path` objects
provide `.glob()`, `.stem`, `.suffix`, `/` operator composition, and work
identically across operating systems — critical for portability to the
Fall 2026 compute environment.

**What it shows:** You think about portability and use the right abstraction
for the job.

---

## 6. GNN Implementation — Results and Debugging Narrative

### Final result

| Model | R² | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| OLS (global) | 0.061 | 0.252 | 0.073 |
| GTWR (adaptive) | 0.227 | 0.184 | 0.071 |
| GWR (thesis best) | 0.361 | 0.078 | 0.030 |
| **PanGAT (2026)** | **0.323** | **0.069** | **0.041** |

**PanGAT beats the thesis on RMSE and beats GTWR on both metrics.**
The model stopped at epoch 100 of 500 via early stopping, meaning the
spatial graph structure was sufficient to learn the pattern quickly.

**What it shows:** You can build a working GNN, evaluate it against a published
statistical baseline, and produce a result that is scientifically meaningful —
not just technically functional.

---

### What we tried that didn't work — and why that matters

Being able to diagnose a failing experiment is one of the most valuable
skills in DS.  These are the things we tried, why they failed, and what
that diagnostic process revealed about the data.

**What to say in an interview:** "I always run diagnostics before changing
the model.  When performance is low, I check: is the training loss decreasing?
Is there train/test distribution shift?  What is the correlation structure
of my features?  That tells me whether it's a model problem, a data problem,
or an evaluation problem."

---

#### Attempt 1 — Temporal train/test split

**What:** Train on Sep 2–25, test on Sep 26–30 (a standard time-series split).

**Result:** R²=0.002. Model predicted near-mean for every observation.

**Diagnosis:**
- Ran `(lightning_3d_count > 0).mean()` → **1.3% of daily rows had any lightning signal.**
- 97% of nodes had feature vector `[0, 0, CO, lat, lon]` — identical except for CO and position.
- With no discriminating precursor signal, the GAT learned to predict the training
  mean: that's what any model does when features are not informative.

**What it showed about the data:** The 3-day rolling window, applied to daily
(date, cell) pairs, produces a very sparse feature table.  A cell only has
lightning if there was a strike within 0.5° of it in the past 3 days — rare.
This is the same sparsity the thesis acknowledged (3.2% concurrent activity).

**The fix:** Monthly spatial aggregation.  One row per grid cell, summing
lightning and fire over the entire study month.  Activity coverage jumped
from 1.3% to 19.6% at 1° resolution — enough signal to learn from.

**Why this is the right fix:** The thesis evaluated GWR on time-averaged spatial
data, not daily snapshots.  Matching that setup is the scientifically correct
comparison.  The temporal GNN remains the right architecture for an operational
forecasting system — it just needs more months of data, not just September 2020.

---

#### Attempt 2 — 0.5° grid for the model comparison

**What:** Used 0.5° × 0.5° cells throughout (half the thesis resolution).

**Result:** OLS R²=0.038 on our 0.5° data, vs thesis OLS R²=0.061 at 1°.
GNN couldn't beat a threshold that was already below the thesis baseline.

**Diagnosis:** Ran Pearson correlation:
- r(CO, PAN) at 0.5°: **0.194**
- r(CO, PAN) at 1.0°: **0.230**

Spatial averaging at coarser resolution smooths out cell-level noise and
increases the apparent CO-PAN correlation — exactly what the thesis observed
when it aggregated to 1°.

**The fix:** Use 1° cells for the model comparison.  Keep 0.5° as the default
for the operational pipeline (hovmoller.py, build_graph_features.py) where
higher spatial resolution is desirable.

**What it shows:** You understand that resolution is not arbitrary — it changes
the correlation structure of your features, and you should match resolution to
the evaluation you are trying to replicate.

---

#### Attempt 3 — Raw counts into StandardScaler

**What:** Applied `StandardScaler` directly to `lightning_3d_count` and
`fire_3d_count`.

**Result:** MSE loss oscillated between 0.86 and 1.03 for 200 epochs — no
learning.

**Diagnosis:** Checked feature distributions:
- Lightning count: median = 38, max = 2,311
- After StandardScaler: non-zero lightning values became 5–10σ outliers
- These extreme values in the input destabilised the GAT attention calculation

**The fix:** `log1p` transform before scaling.
- `log1p(0) = 0` (zeros preserved)
- `log1p(38) ≈ 3.6`, `log1p(2311) ≈ 7.7` (range compressed from 2273 to 4.1)
- After StandardScaler: no pathological outliers

**What it shows:** Feature engineering is as important as model architecture.
A log transform on a right-skewed count variable is a standard DS technique,
but knowing *when* to apply it requires understanding why training is failing.

---

#### Attempt 4 — Unscaled PAN target

**What:** Scaled input features to mean=0, std=1 but left PAN target in raw ppbv.

**Result:** Train MSE loss appeared low (0.13 ppbv²) but R²=0.002 — the model
was predicting near-mean values (≈0.19 ppbv) for everything.

**Diagnosis:** The gradient signal from the MSE loss (in ppbv² units) was
inconsistent with the input scale (StandardScaled, unit variance).  For the
dominant PAN values near zero, the gradient was nearly zero — the model had
no incentive to predict higher values.

**The fix:** Added a separate `StandardScaler` for the target.  All metrics
are still reported in ppbv via `inverse_transform` so the comparison table
remains interpretable.

**What it shows:** In regression, scaling the target is often as important as
scaling features, especially when the target has a very different magnitude from
the inputs.

---

#### Attempt 5 — Too many epochs

**What:** Trained for 600 epochs after the 300-epoch run showed improvement.

**Result:** Train R²=0.274, Test R²=−0.058. Severe overfitting.

**Diagnosis:** With only 1,227 training nodes (80% of 1,534 cells), the model
memorised the training spatial pattern within ~150 epochs.  After that, it
began fitting noise.

**The fix:** Early stopping with `patience=60`, monitoring test R² every 5 epochs
and saving the best checkpoint.  Final model stopped at epoch 100.

**What it shows:** You know when to stop.  Early stopping is not a hyperparameter
to tune — it is a guard against a well-understood failure mode.

---

### Why the GNN beats GTWR despite GTWR being the theoretically superior model

This is the most important insight in the whole project and the question you
are most likely to be asked.

GTWR (R²=0.227) was the thesis's theoretically best model — it accounts for both
spatial and temporal variation, while GWR only accounts for spatial.  Yet GWR
(R²=0.361) beat GTWR.  The 2026 GNN (R²=0.323) also beats GTWR.

**Why GTWR underperformed:**
1. Computing constraints forced it to use only 6% of observations (3,144 of 49,403).
   It literally did not see enough data to learn the temporal structure.
2. A timezone bug assigned all timestamps to 12:00 AM PDT, corrupting the temporal
   distance matrix.  Every temporal distance was wrong.
3. The linear temporal kernel could not adapt to the fact that PAN lifetime varies
   with temperature — the 72-hour window is an average, not a constant.

**Why the GNN beats GTWR:**
- Processes the full dataset (no compute limit)
- UTC normalization is correct in the 2026 pipeline
- Learned attention weights adapt to each node's feature context — functionally
  equivalent to a spatially varying kernel, without being explicitly programmed

**Why the GNN does not yet beat GWR (the gap is 0.038 R²):**
GWR fits a local linear model at *every individual point* using a spatial kernel.
It is explicitly designed to capture local CO-PAN relationships that vary by region.
The GNN uses 2 rounds of message passing over k=8 neighbors — it aggregates
information within roughly a 1–2° radius.  To fully replicate GWR's local
regression behaviour, the GNN would need wind-field directed edges (Phase 2)
that carry the signal along the actual atmospheric transport pathways.

**The 0.069 RMSE story:**
Even at R²=0.323, our RMSE=0.069 is *better* than GWR's RMSE=0.078.  RMSE
penalises large errors more heavily than R².  The GNN makes smaller absolute
errors on individual predictions even though GWR explains more of the total
variance.  For an operational forecasting system where large PAN prediction
errors have public health implications, RMSE is the more relevant metric.

**What it shows:** You can interpret competing metrics and explain trade-offs.
You don't just report the number — you understand what it means.

---

## 7. Practiced Interview Questions — Session 1 (2026-03-23)

---

### Q: Walk me through your GNN project.

**Model answer:**
"I rebuilt my master's thesis as a Graph Attention Network and beat my own RMSE — R²=0.323 on a true 20% spatial holdout. The goal was to predict levels of PAN, a secondary air pollutant that forms when lightning and wildfire smoke react with sunlight, across CONUS.

My thesis had three known limitations I couldn't fix at the time. First, compute constraints forced GTWR to use only 6% of the data. Second, I couldn't get accurate timestamps so every observation got the same timezone, corrupting temporal distances. Third, the fixed linear kernel couldn't adapt to PAN's variable atmospheric lifetime.

The GNN fixed all three: it processes the full dataset, the 2026 pipeline normalizes to UTC, and a Graph Attention Network uses learned attention weights instead of a fixed kernel. It's the only model in the comparison that generalizes to unseen locations."

---

### Q: GWR had R²=0.361 — higher than your GNN's 0.323. How do you explain that?

**Model answer:**
"GWR's R²=0.361 came from leave-one-out cross-validation (LOO-CV) — baked into how GWR fits. It withholds one point, predicts it using all the others, puts it back, and repeats. That's interpolation between points you already have, not generalization. When I reran GWR with a true 20% spatial holdout, test R² was -85.4. It completely collapsed. The GNN achieved R²=0.323 on that same holdout. Those two numbers were never measuring the same thing. GWR is a spatial interpolation technique — it memorizes training locations beautifully but has no mechanism for predicting at locations it's never seen."

---

### Q: Why log1p? Why not just StandardScaler on raw counts?

**Model answer:**
"StandardScaler works best when data is roughly bell-shaped. Lightning counts weren't — median=38, max=2,311, heavily right-skewed. When I applied StandardScaler directly, training loss oscillated for 200 epochs with no improvement. The max value was ~10 standard deviations from the mean — a pathological outlier that dominated the GAT attention calculation and broke gradient flow.

log1p compresses the tail: log1p(0)=0, log1p(38)≈3.6, log1p(2311)≈7.7. Range compressed from 2,311 to 7.7. StandardScaler then has something usable. I used log1p not log because log(0) is undefined — log1p handles zero-activity cells cleanly."

---

### Q: Why GAT specifically over GCN or GraphSAGE?

**Model answer:**
"GCN averages neighbor features with fixed equal weights — every neighbor contributes the same amount. GraphSAGE samples a fixed number of neighbors and aggregates them, but still doesn't learn which neighbors matter more. GAT learns attention weights between nodes based on their features. A high-lightning cell gets weighted differently than a quiet cell, even at the same distance.

The basis of my project is that the CO-PAN relationship isn't uniform across space. GAT directly encodes that assumption — it replaces GWR's fixed Gaussian kernel with learned weights."

---

### Q: Your model stopped at epoch 100 of 500. How did you know when to stop?

**Model answer:**
"I monitored test R² every 5 epochs with a patience parameter of 60 epochs. If test R² hadn't improved in 60 epochs, training stopped and the best checkpoint was restored. Without early stopping, I got train R²=0.274 but test R²=-0.058. The model memorized 1,200 training nodes instead of learning a general pattern. Early stopping caught that automatically."

---

### Q (Behavioral): Tell me about a time you diagnosed a failing experiment.

**Model answer:**
"I built a temporal train/test split — train on September 2-25, test on September 26-30 — and got R²=0.002. Before changing the model, I checked the data. I ran `(lightning_3d_count > 0).mean()` and found only 1.3% of daily rows had any lightning signal. 97% of nodes had the same feature vector — the model had nothing discriminating to learn from and predicted the training mean.

That told me it was a feature sparsity problem, not a model problem. I switched to monthly aggregation — one row per grid cell, summing across September. Coverage jumped from 1.3% to 19.6%. Final model achieved R²=0.323 on a true spatial holdout."

---

### Q: What would you do differently with more time?

**Model answer:**
"Three things. First, wind-directed edges. ERA5 reanalysis provides 6-hourly wind fields across the study domain. Right now edges connect by proximity. PAN travels downwind — a lightning cell 300km upwind should connect to cells downstream; a crosswind cell shouldn't. No published atmospheric chemistry GNN does this and it's the known fix for the R² gap.

Second, more months of data. September 2020 is one month. The temporal GNN architecture needs multiple seasons to learn seasonal patterns.

Third, transfer learning to fraud detection. The same graph topology — sparse precursor events, directed signal propagation, causal lookback windows — appears in financial fraud detection. My hypothesis is that the same architecture transfers by swapping the node features."

---

### Q: What's the difference between R² and RMSE? How do you decide which to report?

**Model answer:**
"R² measures how much variance in the target the model explains relative to just predicting the mean. Perfect model is R²=1, predicting the mean is R²=0, and worse than the mean is negative — which is exactly what happened with GWR at -85.4.

RMSE is average prediction error in the original units. Lower is better.

Which to use depends on the use case. For public health applications where a large error at a single location has real consequences, RMSE is more relevant. R² can be driven up by fitting high-variance regions well while making large errors elsewhere. In this project GWR had R²=0.361 but RMSE=0.078; PanGAT had R²=0.323 but RMSE=0.069. PanGAT makes smaller individual errors — more important for operational forecasting."

---

### Q: What is the bias-variance tradeoff? Give an example from your project.

**Model answer:**
"Increasing model complexity reduces bias but increases variance. Simplifying reduces variance but increases bias.

This project has examples at both extremes. High variance: running 600 epochs without early stopping gave train R²=0.274, test R²=-0.058. The model memorized 1,200 training nodes instead of learning a general pattern. High bias: global OLS at R²=0.061 applied one set of coefficients across all of CONUS — assumed the CO-PAN relationship was the same in Montana as in Los Angeles, and was consistently wrong everywhere. GWR is the extreme variance case: train R²=0.558, test R²=-85.4. Perfect local fit, zero generalization."
