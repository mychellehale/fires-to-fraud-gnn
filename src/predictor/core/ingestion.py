from typing import Generator
import pandas as pd

def stream_data_chunks(
        file_path: str, 
        chunk_size: int = 1000
) -> Generator[pd.DataFrame, None, None]:
    """
    Reads a large dataset (like IEEE-CIS or NASA TES) in chunks.
    Yields each chunk to the processing pipeline.
    
    :param file_path: Description
    :type file_path: str
    :param chunk_size: Description
    :type chunk_size: int
    :return: Description
    :rtype: Generator[DataFrame, None, None]
    """
    # Using 'chunksize' in pandas returns an iterator
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # We perform basic UTC normalization inside the generator
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
        yield chunk