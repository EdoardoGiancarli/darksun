"""
Custom data types and containers for the LEM-X camera modules analysis pipeline.
"""

from typing import NamedTuple

__all__ = [
    "LogEntry", "Tag", "Candidate",
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


class Tag(NamedTuple):
    """
    Tag container for marking a point in a plot.

    Attributes:
        ID (str):
            Tag name.
        y (int | float):
            Position along the y-axis.
        x (int | float):
            Position along the x-axis.
    """
    ID: str
    y: int | float
    x: int | float


class Candidate(NamedTuple):
    """
    Source candidate parameters container.

    Attributes:
        shift_x (float):
            Coded-mask camera local frame sky-coord along the x-axis [mm].
        shift_y (float):
            Coded-mask camera local frame sky-coord along the y-axis [mm].
        fluence (float):
            Observed candidate fluence [ph].
        snr (float):
            Candidate significance [adim].
    """
    shift_x: float
    shift_y: float
    fluence: float
    snr: float


# end