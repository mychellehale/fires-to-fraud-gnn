import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar, cast

import pandas as pd
import geopandas as gpd
import xarray as xr
from utils import track_progress


# 1.  THE INGESTOR CLASS
class AtmosphericIngestor:
    # CONUS Bounds
    LAT_MIN, LAT_MAX = 24.0, 50.0
    LON_MIN, LON_MAX = -125.0, -66.0
    SPATIAL_BUFFER = 0.1  # Approx 11km

    def __init__(self, gas_dir: str, fire_path: str, lightning_dir: str) -> None:
        self.gas_dir = Path(gas_dir)
        self.fire_path = Path(fire_path)
        self.lightning_dir = Path(lightning_dir)
        
        # MODIS Fire is usually small enough to load once
        self.fire_gdf: gpd.GeoDataFrame = self._load_modis_fire()

    def _load_modis_fire(self) -> gpd.GeoDataFrame:
        df: pd.DataFrame = pd.read_csv(self.fire_path)
        # Fix Mypy: list[str] to Index[str]
        df.columns = cast(pd.Index, df.columns.str.lower())
        
        df['fire_time'] = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), 
            format='%Y-%m-%d %H%M'
        ).dt.tz_localize('UTC')
        
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )

    def process_lightning_file(self, nc_path: Path) -> gpd.GeoDataFrame:
        """Type-annotated Xarray processing."""
        # Annotate as xr.Dataset
        ds: xr.Dataset = xr.open_dataset(nc_path)
        
        # Access variables as xr.DataArray
        times: xr.DataArray = ds['lightning_flash_time']
        lats: xr.DataArray = ds['lightning_flash_lat']
        lons: xr.DataArray = ds['lightning_flash_lon']

        df = pd.DataFrame({
            'strike_time': times.values,
            'lat': lats.values,
            'lon': lons.values
        })
        ds.close() # Clean up file handle
        
        df['strike_time'] = pd.to_datetime(df['strike_time']).dt.tz_localize('UTC')
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    @track_progress(desc="Linking Gas, Fire, and Lightning")
    def stream_pipeline(self, chunk_size: int = 2000) -> Generator[gpd.GeoDataFrame, None, None]:
        # Pathlib glob returns a generator; sorted() turns it into a List[Path] for Mypy
        gas_files = sorted(self.gas_dir.glob("PAN_CO_*.txt"))
        
        for gas_file in gas_files:
            # Match the lightning file for the same day (assuming PAN_CO_MMDD.txt)
            day_str = gas_file.stem.split('_')[-1]
            lightning_files = list(self.lightning_dir.glob(f"*2020{day_str}*.nc"))
            
            if not lightning_files:
                continue
                
            # Now using the day_lightning data!
            day_lightning = self.process_lightning_file(lightning_files[0])
            
            reader = pd.read_csv(gas_file, names=['date', 'lng', 'lat', 'co', 'pan'], chunksize=chunk_size)
            
            for chunk in reader:
                chunk.columns = cast(pd.Index, chunk.columns.str.lower())
                
                # Spatial filtering to CONUS
                chunk = chunk[chunk['lat'].between(self.LAT_MIN, self.LAT_MAX) & 
                              chunk['lng'].between(self.LON_MIN, self.LON_MAX)]
                if chunk.empty: continue
                
                chunk['gas_time'] = pd.to_datetime(chunk['date']).dt.tz_localize('UTC')
                gdf_gas = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk.lng, chunk.lat), crs="EPSG:4326")

                # Create spatial buffer for join
                gas_buffer = gdf_gas.copy()
                gas_buffer['geometry'] = gas_buffer.geometry.buffer(self.SPATIAL_BUFFER)

                # Causal Join 1: Lightning
                l_join = gpd.sjoin(gas_buffer, day_lightning, how="inner", predicate="intersects")
                l_matches = l_join[
                    (l_join['gas_time'] > l_join['strike_time']) & 
                    (l_join['gas_time'] <= l_join['strike_time'] + pd.Timedelta(hours=72))
                ]

                # Causal Join 2: Fire
                f_join = gpd.sjoin(gas_buffer, self.fire_gdf, how="inner", predicate="intersects")
                f_matches = f_join[
                    (f_join['gas_time'] > f_join['fire_time']) & 
                    (f_join['gas_time'] <= f_join['fire_time'] + pd.Timedelta(hours=72))
                ]

                # Yield combined results
                final = pd.concat([l_matches, f_matches]).drop_duplicates(subset=['gas_time', 'lat', 'lng'])
                if not final.empty:
                    yield cast(gpd.GeoDataFrame, final)