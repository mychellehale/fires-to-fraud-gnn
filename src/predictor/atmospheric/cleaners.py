import pandas as pd
from abc import ABC, abstractmethod

class BaseSourceCleaner(ABC):
    """Abstract base class to ensure all sources follow the same rules."""
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class TESCleaner(BaseSourceCleaner):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # TES specific logic: e.g., pressure-level filtering
        df = df[df['pressure'] > 500] 
        return df[['timestamp', 'lat', 'lon', 'PAN']]

class FireEventCleaner(BaseSourceCleaner):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fire data logic: e.g., converting 'start_date' to 'timestamp'
        df = df.rename(columns={'start_date': 'timestamp'})
        return df[['timestamp', 'lat', 'lon', 'intensity']]