# Fires-to-Fraud GNN
**Spatio-Temporal Modeling: From Atmospheric Transport to Financial Fraud**

## Section 1: Atmospheric Fires & Causal Ingestion
This phase focuses on the "Fuel" for the model. We ingest messy, high-dimensional data from NASA sensors to track how wildfires and lightning events create causal signatures in the atmosphere.
* **Wildfire Tracking:** Processing MODIS Active Fire data to identify heat signatures.
* **Lightning Energy:** Integrating LIS (Lightning Imaging Sensor) data from NetCDF4 files.
* **Causal Windowing:** A custom 72-hour temporal window and 11km spatial radius to link events to Gas (CO/PAN) measurements.

## Section 2: GNN Architecture & Graph Construction
This phase transforms geographic points into a relational graph structure.
* **Spatial Edges:** Using K-Nearest Neighbors (KNN) to connect atmospheric sensors based on distance.
* **Graph Neural Networks:** Utilizing PyTorch Geometric to propagate "event energy" through the spatial network.
* **Anomaly Detection:** Training the model to recognize "normal" transport vs. fraudulent/anomalous signatures.

## 📁 Architecture
```text
src/predictor/
├── atmospheric/    # Section 1: Ingestion (LIS, MODIS, PAN-CO)
├── core/           # Section 2: GNN & Graph Construction
└── utils.py        # Shared infrastructure (Progress tracking)
```

## 🛠 Setup
```Bash
uv sync
uv run mypy src/predictor/atmospheric/ingestion.py
```
---
**Author:** Mychelle Hale  
**Status:** 🏗️ Research in Progress (January 2026)  
<!-- **Affiliation:** Data Science Prep / Thesis Development -->