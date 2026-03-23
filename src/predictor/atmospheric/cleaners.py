import pandas as pd
import polars as pl
from abc import ABC, abstractmethod
import xarray as xr
from pathlib import Path

class BaseSourceCleaner(ABC):
    """
    Abstract base class to define the interface for atmospheric data cleaners.
    
    Ensures that all data sources (Satellite, Fire, etc.) implement a standard 
    cleaning pipeline for GNN node feature consistency.
    """
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and filters the input DataFrame.

        Args:
            df (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: A standardized DataFrame ready for GNN ingestion.
        """
        pass

class TESCleaner(BaseSourceCleaner):
    """
    Cleaner for Tropospheric Emission Spectrometer (TES) satellite data.

    Focuses on PAN (Peroxyacetyl nitrate) concentrations and vertical
    pressure level filtering aligned with the GTWR thesis baseline.
    """
    # Free Troposphere "Goldilocks Zone" per REFACTOR_DOCS: ~700–300 hPa
    FREE_TROP_INDICES = [2, 3, 4, 5]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters TES data to the Free Troposphere and applies quality control.

        Steps:
          1. Retain only retrievals with SpeciesRetrievalQuality == 1 (Good),
             preventing ghost plumes from biasing GTWR coefficients.
          2. Filter to level indices 2–5 (~700–300 hPa), the Free Tropospheric
             signal isolated from boundary-layer and stratospheric interference.
          3. Drop rows with null PAN values so missing lower-level measurements
             are never silently treated as zero concentration.

        Args:
            df (pd.DataFrame): Raw TES readings.

        Returns:
            pd.DataFrame: Quality-controlled PAN concentrations with lat/lon.
        """
        # QC: keep only Good retrievals if quality column is present
        if 'SpeciesRetrievalQuality' in df.columns:
            df = df[df['SpeciesRetrievalQuality'] == 1]

        # Vertical filter: Free Troposphere only (indices 2–5)
        df = df[df['level_index'].isin(self.FREE_TROP_INDICES)]

        # Null guard: missing values must not be treated as zero concentration
        df = df.dropna(subset=['PAN'])

        return df[['timestamp', 'lat', 'lon', 'PAN', 'CO', 'level_index']]
    
class FireEventCleaner(BaseSourceCleaner):
    """
    Cleaner for biomass burning and wildfire event datasets.
    """
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes fire event columns for temporal alignment.

        Args:
            df (pd.DataFrame): Raw fire event logs.

        Returns:
            pd.DataFrame: Fire intensity and location data.
        """
        # Fire data logic: e.g., converting 'start_date' to 'timestamp'
        df = df.rename(columns={'start_date': 'timestamp'})
        return df[['timestamp', 'lat', 'lon', 'intensity']]

class AtmosphericCleaner:
    """
    High-performance utilities for processing raw atmospheric instrument files.

    Handles complex binary formats like NetCDF4, focusing on lightning
    event detection and coordinate normalization.
    """

    # Seconds between 1970-01-01 (Unix epoch) and 1993-01-01 (TAI93 epoch)
    TAI93_UNIX_OFFSET: int = 725_846_400

    @staticmethod
    def convert_tai93(df: pl.DataFrame, tai_col: str = "tai_time") -> pl.DataFrame:
        """
        Converts TAI93 float timestamps to UTC datetime[μs].

        Applies the manual Unix offset (725,846,400 s) to raw floats to bypass
        Xarray datetime cast conflicts. Requires .sort("datetime") before any
        group_by_dynamic call downstream.

        Args:
            df (pl.DataFrame): DataFrame containing a TAI93 float column.
            tai_col (str): Name of the TAI93 seconds column.

        Returns:
            pl.DataFrame: Original DataFrame with an added 'datetime' column.
        """
        return df.with_columns(
            ((pl.col(tai_col) + AtmosphericCleaner.TAI93_UNIX_OFFSET) * 1_000_000)
            .cast(pl.Int64)
            .cast(pl.Datetime("us"))
            .alias("datetime")
        )

    @staticmethod
    def bin_to_grid(df: pl.DataFrame, resolution: float = 0.5) -> pl.DataFrame:
        """
        Bins lat/lon coordinates to a regular spatial grid.

        A resolution of 0.5° aligns with the CO tracer data and reduces
        point-level noise for GNN node feature construction.

        Args:
            df (pl.DataFrame): DataFrame with 'lat' and 'lon' columns.
            resolution (float): Grid cell size in degrees (default 0.5°).

        Returns:
            pl.DataFrame: DataFrame with lat/lon snapped to bin centers.
        """
        half = resolution / 2.0
        return df.with_columns([
            ((pl.col("lat") / resolution).floor() * resolution + half).alias("lat"),
            ((pl.col("lon") / resolution).floor() * resolution + half).alias("lon"),
        ])

    @staticmethod
    def cap_at_percentile(
        df: pl.DataFrame, col: str, percentile: float = 0.99
    ) -> pl.DataFrame:
        """
        Caps a column at the given percentile to mitigate extreme outliers.

        Prevents single massive convective cells from skewing spatial weights
        (e.g., lightning flash density). Per REFACTOR_DOCS, the default cap
        is the 99th percentile.

        Args:
            df (pl.DataFrame): Input DataFrame.
            col (str): Column name to cap.
            percentile (float): Upper quantile threshold (default 0.99).

        Returns:
            pl.DataFrame: DataFrame with the column clipped at the threshold.
        """
        cap_val: float = df[col].quantile(percentile)  # type: ignore[assignment]
        return df.with_columns(pl.col(col).clip(upper_bound=cap_val))

    @staticmethod
    def clean_lis_netcdf(file_path: Path) -> pl.DataFrame:
        """
        Extracts lightning event data from NASA LIS V3.0 NetCDF files.

        This method uses raw array extraction to bypass Xarray dimension-mapping 
        errors and treats timestamps as raw floats to ensure type safety 
        during UTC conversion.

        Args:
            file_path (Path): Path to the .nc orbit file.

        Returns:
            pl.DataFrame: A Polars DataFrame containing 'lat', 'lon', and 
                         'tai_time' (seconds since 1993-01-01). 
                         Returns an empty frame if no events are detected.
        """
        with xr.open_dataset(file_path, decode_timedelta=False, decode_times=False) as ds:
            if 'lightning_event_lat' not in ds.variables:
                return pl.DataFrame()
            
            # Use decode_times=False in open_dataset and force float here
            lats = ds['lightning_event_lat'].values.flatten()
            lons = ds['lightning_event_lon'].values.flatten()
            times = ds['lightning_event_TAI93_time'].values.flatten().astype(float)

            if lats.size == 0:
                return pl.DataFrame()

            return pl.DataFrame({
                "lat": lats,
                "lon": lons,
                "tai_time": times
            })