import polars as pl


def normalize_timestamps(
    df: pl.DataFrame,
    col: str,
    timezone: str = "UTC",
) -> pl.DataFrame:
    """
    Parses a string column to timezone-aware datetime.

    Returns a new DataFrame; the original is never mutated.

    Args:
        df (pl.DataFrame): Input data.
        col (str): Name of the string column containing timestamps.
        timezone (str): IANA timezone string to attach (default 'UTC').

    Returns:
        pl.DataFrame: DataFrame with the column cast to Datetime and
                      localized to the requested timezone.
    """
    return df.with_columns(
        pl.col(col).str.to_datetime().dt.replace_time_zone(timezone).alias(col)
    )
