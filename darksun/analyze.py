"""
IROS output data management and computation.
"""

from pandas import DataFrame

from .data import Log

__all__ = [""]


def IROS_log() -> Log:
    """
    
    """
    raise NotImplementedError


def IROS_sources_log() -> Log:
    """
    
    """
    raise NotImplementedError






def dict2df(data: dict) -> DataFrame:
    """
    Converts `Log` database to a Pandas dataframe.

    Args:
        data (dict): Input data.
    
    Returns:
        df (DataFrame): Output dataframe.
    """
    df = DataFrame({
        (cam, param): values
        for cam, cam_data in data.items() 
        for param, values in cam_data.items()
    })
    return df


# end