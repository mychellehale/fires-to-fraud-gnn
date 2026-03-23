# Future Research — Blue Ocean Opportunities

This document maps the white space beyond the published dissertation:
what the current pipeline makes possible that was not possible before,
and where the research could go next.

> Grounded in the published dissertation (Hale, CSULB August 2022) and the
> limitations it explicitly acknowledges.

---

## Phase 1 Status — Complete ✓

The GNN pipeline (Phase 1) has been built and evaluated:

| Model | R² | RMSE | Status |
| :--- | :--- | :--- | :--- |
| Thesis GWR (best) | 0.361 | 0.078 | Published baseline |
| Thesis GTWR | 0.227 | 0.184 | Theoretically better, practically worse |
| **PanGAT Phase 1** | **0.323** | **0.069** | **Better RMSE ✓ Beats GTWR ✓** |

The R² gap to GWR (0.038) is the target for Phase 2.

---

## What the Dissertation Itself Says to Do Next

The thesis (Chapter 8: Future Works) explicitly named three directions:

1. **Geographically Weighted Random Forest Classification (GWRFC):** Classify PAN
   observations as fire-sourced or not — a spatial classification problem.
2. **Geographically Weighted Neural Networks (GWRNN):** Capture non-linear precursor
   relationships that GTWR's linear kernel cannot represent.
3. **Geographically Weighted Autoregressive (GWAR) models:** Account for spatial
   autocorrelation (Moran's I) in the residuals — something OLS/GWR both ignored.
4. **Include wind speed, wind direction, and temperature** as co-variates to improve
   temporal distance calculations (the thesis explicitly called out the lack of these
   as a limitation).

The GNN pipeline is the architectural answer to all four: it is a spatiotemporal neural
network (addresses GWRNN), its graph edges encode spatial adjacency and can be directed
by wind transport (addresses GWAR autocorrelation), and its node classification head
can distinguish fire-sourced from non-fire-sourced PAN (addresses GWRFC).

---

## From GTWR to GNN — The Core Upgrade

The best published result was non-adaptive **GWR** (R² = 0.361, RMSE = 0.078), not
GTWR — which is counterintuitive since GTWR is the more powerful model.
GTWR (R² = 0.227) underperformed because: (a) compute constraints limited it to 6%
of observations, (b) a timezone bug corrupted temporal distances, and (c) the linear
temporal kernel cannot represent non-stationary lag structure.

The GNN upgrade removes all three root causes and removes the linearity assumption.

| GTWR Limitation | GNN Opportunity |
| :--- | :--- |
| Linear predictor | Can learn non-linear PAN formation chemistry |
| Spatial weights decay smoothly with distance | Graph edges can represent physical transport pathways (wind trajectories) |
| One model per spatial kernel | Single model generalizes across the CONUS domain |
| Cannot learn from multiple precursors jointly | Graph nodes can encode multi-source features (lightning + fire + CO simultaneously) |
| Static coefficients | Attention weights can vary with atmospheric conditions |

---

## Blue Ocean 1 — Causal Graph Construction

**The opportunity:** The 72-hour lookback window is currently applied as a
hard filter — an event either falls inside the window or it doesn't.
A GNN can replace this binary rule with **learned edge weights** that
encode how much a lightning event at node A contributed to PAN at node B,
conditioned on atmospheric transport (wind fields).

**What this requires:**
- ERA5 or HRRR wind field data to define physically motivated graph edges
  (not just k-nearest-neighbor or distance-based)
- Directed edges (lightning → gas measurement node) rather than undirected
- An edge feature encoding the transport time along the wind trajectory

**Why it's blue ocean:** No published atmospheric chemistry GNN uses
physically motivated directed edges derived from wind transport.  Most
use undirected spatial proximity.

---

## Blue Ocean 2 — Transfer Learning: Fires → Fraud

**The opportunity:** The `fintech/` module is a placeholder for the
"fires-to-fraud" bridge.  The structural similarity between the two
problems is what makes this project unusual:

| Atmospheric Problem | Financial Fraud Problem |
| :--- | :--- |
| Lightning event (point in space/time) → PAN formation | Transaction event → fraud signal |
| Precursor signal propagates with a lag | Fraud rings operate on coordinated lags |
| Geographic clustering of fire activity | Geographic clustering of fraud (card-present clusters) |
| 72-hour causal window | Fraud detection lookback windows |
| CONUS spatial graph | Merchant/cardholder transaction graph |

The hypothesis is that **the same GNN architecture** — spatiotemporal
graph with causal lookback windows, multi-source node features, and
directed edges — transfers to fraud detection with domain-specific
node/edge features substituted in.

**What this requires:**
- A fraud transaction dataset with geographic and temporal resolution
  (IEEE-CIS is already listed in `pyproject.toml`)
- A node feature schema that maps: lightning → transaction amount/type,
  fire → merchant category, PAN → fraud label
- An experiment showing that pretraining on the atmospheric graph and
  fine-tuning on the fraud graph outperforms training on fraud data alone

**Why it's blue ocean:** Transfer learning across domains as structurally
different as atmospheric chemistry and financial fraud has not been
demonstrated.  The claim is that *graph topology* — not domain content —
is the transferable artifact.

---

## Blue Ocean 3 — Temporal Attention as a Climate Signal Detector

**The opportunity:** Replace the fixed 72-hour window with a
**learned temporal attention mechanism** that dynamically determines how
far back to look based on current atmospheric conditions (temperature,
humidity, pressure gradient).

In warm, dry conditions PAN degrades faster — the effective lookback
window should shrink.  In cold upper tropospheric air the window can
extend to 7+ days.  A model that learns this relationship would be both
more accurate and scientifically interesting.

**What this requires:**
- ERA5 temperature/humidity co-variates at TES pressure levels
- A temporal attention layer (e.g., transformer encoder over the 72-hour
  event sequence) rather than a hard window filter
- Ablation study comparing fixed-window vs. learned-window performance

---

## Blue Ocean 4 — Expanding to Other PAN Precursors

**The opportunity:** The current pipeline uses two precursor sources
(lightning → NOx, fires → VOCs).  The atmospheric literature identifies
additional pathways:

| Precursor | Source | Data Availability |
| :--- | :--- | :--- |
| Biogenic VOC emissions | MEGAN model output | Publicly available |
| Anthropogenic NOx | EPA emissions inventory | Publicly available |
| Aircraft NOx | AEIC inventory | Publicly available |
| Stratospheric intrusions | ERA5 potential vorticity | Publicly available |

Adding these as additional node feature channels to the GNN would allow
the model to learn which precursor pathway dominates under different
conditions — something GTWR cannot do because it treats precursors
independently.

---

## Blue Ocean 5 — Operational Forecasting

**The opportunity:** The current system is retrospective (September 2020).
With NWP (numerical weather prediction) wind forecasts and near-real-time
LIS data, the same pipeline could produce **72-hour PAN forecasts** for
air quality applications.

This would transform the project from a research result into an
operational tool — a framing that is more compelling for industry
interviews.

---

## Summary — The "So What" for Interviews

When asked "where does this go next?", the answer has three levels:

1. **Technical next step:** Graph construction with wind-directed edges
   (Blue Ocean 1) — this is the `core/` module that is currently a stub.

2. **Research novelty:** Transfer learning from atmospheric graphs to
   fraud graphs (Blue Ocean 2) — this is the "fires-to-fraud" thesis of
   the whole repository name.

3. **Real-world impact:** Operational PAN forecasting as an air quality
   product (Blue Ocean 5) — this shows you think about deployment, not
   just modeling.
