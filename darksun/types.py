"""
Custom data types and containers for the WFM analysis pipeline.
"""

from typing import NamedTuple

__all__ = [
    "LogEntry",
]

class LogEntry(NamedTuple):
    """
    Data entry container for log structure.

    Attributes:
        entry (str):
            Parameter name.
        frmt (str):
            Data type in string format following astropy's
            FITS data format (e.g., J for `int` data).
        unit (str):
            Parameter's physical units (e.g., 'mm', 'deg', etc.).
    """
    entry: str
    frmt: str
    unit: str


# end