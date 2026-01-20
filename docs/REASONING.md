# Technical Reasoning

## 1. Thesis Baseline: The 3-Day Window
The decision to utilize a **3-day lookback** is rooted in atmospheric chemistry. PAN (Peroxyacetyl nitrate) has a typical lifetime of ~3 days in the mid-troposphere. This window captures the "Direct Path" from lightning-induced $NO_x$ to detectable PAN concentrations.

## 2. Structural Improvements
While the temporal window remains 3 days for consistency with the thesis baseline, the **infrastructure** has been modernized for the 2026 GNN implementation.

| Feature | Thesis Implementation | 2026 Refactor | Benefit |
| :--- | :--- | :--- | :--- |
| **Engine** | Pandas | **Polars** | Faster processing of 478 LIS files |
| **IO** | `os.path` strings | **Pathlib Objects** | Robustness for 2026 relocation |
| **Aggregates** | Total Count | **Centroid + Dispersion** | Improved GNN node features |

## 3. Data Integrity Fix
The NASA LIS `V3.0` NetCDF files do not explicitly label an "event" dimension in a way `xarray` always recognizes. The refactored cleaner now accesses variable arrays directly via the `.values` attribute to ensure 100% data recovery from all orbits.