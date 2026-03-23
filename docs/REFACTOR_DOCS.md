# REFACTOR_DOCS.md

## 1. Dissertation Overview
* **Title:** Predicting PAN from CO, Fires, and Lightning: A Geographically and Temporally Weighted Regression Approach
* **Case Study:** September 2020
* **Objective:** Modernizing the original GTWR methodology into a Graph Neural Network (GNN) framework.

## 2. Scientific Constraints (Thesis Baseline)

### Temporal Window: The 72-Hour Rule
The model utilizes a 3-day lookback window for all precursor data (Lightning and Fires).
* **Rationale:** PAN has a typical lifetime of ~3 days in the free troposphere.
* **Mechanism:** Captures the "Direct Path" from NOx injection to stabilized PAN plumes.

### Vertical Pressure Filtering (TES Indices 0–7)
Analysis is restricted to the Free Troposphere to ensure consistency with GTWR coefficients.

| Index | Approx. Pressure | Status | Rationale |
| :--- | :--- | :--- | :--- |
| 0 | ~1000 hPa | Excluded | Boundary Layer noise; thermal instability. |
| 1 | ~825 hPa | Context | Lower tropospheric transition. |
| **2–5** | **~700–300 hPa** | **PRIMARY** | **Free Troposphere (Goldilocks Zone).** |
| 6–7 | < 250 hPa | Excluded | Upper Troposphere / Stratospheric interference. |

## 3. Technical Refactor (2026 Implementation)
* **Engine:** Polars (Eager) for high-speed aggregation.
* **Portability:** Pathlib integration for the Fall 2026 relocation.
* **Time Normalization:** Manual TAI93 float offsets to bypass Xarray decoding errors.
* **Sorting:** Mandatory pre-sort for `group_by_dynamic` operations.

While the temporal window remains 3 days for consistency with the thesis baseline, the **infrastructure** has been modernized for the 2026 GNN implementation.

| Feature | Thesis Implementation | 2026 Refactor | Benefit |
| :--- | :--- | :--- | :--- |
| **Engine** | Pandas | **Polars** | Faster processing of 478 LIS files |
| **IO** | `os.path` strings | **Pathlib Objects** | Robustness for 2026 relocation |
| **Aggregates** | Total Count | **Centroid + Dispersion** | Improved GNN node features |

## 4. Advanced Data Cleaning & Integrity
To improve model performance beyond standard regression, the following cleaning protocols were developed during the thesis and migrated to the 2026 Polars pipeline:

### A. Outlier Mitigation & QC Flags

Satellite Quality Control: Automated filtering using instrument-specific quality flags (e.g., TES SpeciesRetrievalQuality). Only retrievals with a flag of 1 (Good) are retained to prevent "ghost plumes" from impacting the GTWR coefficients.

Extreme Value Capping: Implementation of a 99th-percentile cap for lightning flash density to prevent single massive convective cells from skewing the spatial weights.

### B. Vertical Profile Standardization (TES 0-7)

The "Zero-Value" Problem: Handled missing values at lower levels (Indices 0–1) by ensuring they weren't treated as "zero concentration," but rather as "null," preventing artificial dips in the predicted PAN levels.

Index Consistency: Enforced the Indices 2–5 filter to isolate the Free Tropospheric signal from surface-level pollution.

### C. Spatio-Temporal Alignment

Coordinate Binning: Instead of raw point data, events are now binned into a 0.5 ×0.5∘ grid. This reduces noise and aligns with the resolution of the CO tracer data.

Temporal Lag Handling: Explicitly synchronized the 3-day lookback window across multiple timezones (UTC vs. Local Solar Time) to ensure a lightning strike at 11:59 PM is correctly attributed to the right daily aggregate.

### 5. Technical "Escape Hatches" (2026 Updates)
TAI93 Normalization: Manual Unix offset (725,846,400) applied to raw floats to avoid datetime cast conflicts.

Strict Sorting: Mandatory .sort("datetime") pre-processing implemented to enable optimized group_by_dynamic rolling window aggregations.