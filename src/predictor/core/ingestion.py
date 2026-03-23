import polars as pl
from pathlib import Path


def stream_data_chunks(file_path: Path | str) -> pl.LazyFrame:
    """
    Returns a lazy scan of a large CSV file for deferred, memory-efficient processing.

    Polars lazy evaluation replaces manual pandas chunking: the query optimizer
    pushes filters and projections down to the scan, so only the needed data
    is read. Call .collect() on the result when you are ready to materialize.

    Args:
        file_path (Path | str): Path to the CSV file.

    Returns:
        pl.LazyFrame: Unevaluated scan ready for chained transformations.
    """
    return pl.scan_csv(file_path)
