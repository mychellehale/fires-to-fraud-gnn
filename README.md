# Fires-to-Fraud GNN
**Rebuilding a Published GTWR Model as a Graph Attention Network**

This repository is a 2026 reimplementation and extension of my 2022 master's thesis:

> Hale, M. (2022). *Predicting PAN from CO, Fires, and Lightning: A Geographically and Temporally Weighted Regression Approach.* M.S. in Applied Statistics, California State University Long Beach.

---

## What This Solves

My thesis identified three limitations in the GTWR model that I could not fix at the time:

1. **Compute constraints** — GTWR could only run on 6% of the 49,403 observations due to memory limits
2. **Timezone bug** — all timestamps were uniformly assigned 12:00 AM PDT, corrupting temporal distance calculations
3. **Fixed linear kernel** — could not adapt to PAN's spatially varying atmospheric lifetime

This pipeline addresses all three: it processes the full dataset, normalizes timestamps to UTC on ingest, and replaces the fixed kernel with learned Graph Attention Network weights.

---

## Results

| Model | R² | RMSE | Evaluation |
| :--- | :--- | :--- | :--- |
| OLS (thesis) | 0.061 | 0.252 | LOO-CV |
| GTWR (thesis) | 0.227 | 0.184 | LOO-CV, 6% of data |
| GWR (thesis) | 0.361 | 0.078 | LOO-CV — not a spatial holdout |
| GWR (2026, test) | -85.4 | 2.384 | True 20% spatial holdout |
| **PanGAT (ours)** | **0.323** | **0.069** | **True 20% spatial holdout** |

PanGAT is the only model in this table that generalizes to unseen locations. GWR's thesis R²=0.361 came from internal leave-one-out cross-validation — when rerun with a true holdout it collapses. See `docs/REFACTOR_OVERVIEW.md` for the full analysis.

---

## What's in the Repo

```
src/predictor/
├── atmospheric/        # Data ingestion: NASA LIS lightning, MODIS fire, CrIS PAN/CO
│   ├── ingestion.py    # AtmosphericIngestor — reads NetCDF, applies 72hr causal window
│   └── cleaners.py     # AtmosphericCleaner — UTC normalization, grid binning, QC filters
└── core/
    ├── graph.py        # Graph construction — k-NN spatial edges, node feature assembly
    └── model.py        # PanGAT — 2-layer Graph Attention Network (PyTorch Geometric)

research/
├── build_graph_features.py   # Fuses lightning + fire + PAN/CO into graph_features.parquet
├── train_gnn.py              # Training loop, early stopping, results table
├── run_baselines.py          # OLS and GWR Python baselines (mgwr)
├── export_for_r.py           # Exports preprocessed CSVs for R GWmodel
├── run_r_baselines.R         # GWR via R GWmodel
├── run_r_gtwr_only.R         # GTWR via R GWmodel (UTC-fixed, n=2000)
├── hovmoller.py              # Hovmöller diagram — validates 72hr lightning-PAN lag
└── make_figures.py           # Publication figures for the write-up

docs/
├── REFACTOR_OVERVIEW.md      # Every change from thesis to 2026 pipeline, with rationale
└── INTERVIEW_PREP.md         # Technical decision explanations and Q&A
```

---

## Key Technical Decisions

- **Polars** instead of pandas — parallel execution, no implicit index, lazy scan API
- **Manual TAI93 → UTC conversion** — bypasses xarray decode failures on malformed NASA NetCDF files
- **log1p before StandardScaler** — lightning counts are right-skewed (median=38, max=2311); log1p compresses the tail without dropping zeros
- **Graph Attention Network** — learned attention weights replace GWR's fixed Gaussian kernel; weights are a function of node features, not just distance
- **Early stopping (patience=60)** — guards against overfitting on the ~1,200 training nodes

---

## Setup

```bash
uv sync
python research/build_graph_features.py
python research/train_gnn.py
```

---

**Author:** Mychelle Hale
**Status:** Phase 1 complete. Phase 2 (wind-directed edges via ERA5) in planning.
