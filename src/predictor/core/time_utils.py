import pandas as pd

def normalize_timestamps(
        df: pd.DataFrame, 
        col: str, 
        timezone: str = 'UTC'
        )-> pd.DataFrame:
    """
    Docstring for normalize_timestamps
    
    :param df: Description
    :type df: pd.DataFrame
    :param col: Description
    :type col: str
    :param timezone: Description
    :type timezone: str
    :return: A DataFrame with standardized time
    :rtype: DataFrame
    """

    df[col] = pd.to_datetime(df[col], utc = True)

    # We return the whole DataFrame, hence the pd.DataFrame hint
    return df