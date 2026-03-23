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
    pressure level filtering.

    I think this will need to be updated to have the correct pressure and atmospheric levels. 
    """
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters TES data to mid-to-lower tropospheric pressure levels.

        Args:
            df (pd.DataFrame): Raw TES readings.

        Returns:
            pd.DataFrame: Filtered PAN concentrations with lat/lon coordinates.
        """
        # TES specific logic: e.g., pressure-level filtering
        """
        Filters data to the Free Troposphere (Indices 3, 4, 5).
        These levels (~511 to ~287 hPa) are where the GTWR model
        found the highest correlation with precursor lightning.
        """
        # Thesis Alignment: Level Indices 3, 4, 5
        valid_indices = [3, 4, 5]
        df = df[df['level_index'].isin(valid_indices)]
        
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