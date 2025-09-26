"""
Module for data handling.
"""

from typing import Any, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from copy import deepcopy

import numpy as np
from pandas import DataFrame
from astropy.io.fits.fitsrec import FITS_rec
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from bloodmoon.types import CoordEquatorial
from bloodmoon.coords import pos2equatorial
from bloodmoon.io import SimulationDataLoader
from bloodmoon.io import _exists_valid
from bloodmoon.mask import CodedMaskCamera

from .types import LogEntry
from .filtering import filter_data
from .filtering import filter_catalogue

__all__ = [
    "Log", "create_log", "DataLoader", "get_data",
    "CatalogueLoader", "get_catalogue", "fit_WCS"
]


class Log:
    """
    Structure for data logging.\n

    This class creates a flexible database structure aimed to store
    IROS output parameters. It provides methods for initializing the
    log, structured as a dictionary, and to update the entries.
    
    Attributes:
        log (dict[str, list] | None):
            Log structure (initialized to `None`).
        params (Sequence[LogEntry] | None):
            Log parameter entries (initialized to `None`).
        name (str | None, optional (default=`None`)):
            Log name.
    """
    def __init__(self, name: str | None = None) -> None:
        self.name = f"{name.upper()}" if name else ''
        self._log = None
        self._params = None
    
    @property
    def log(self) -> dict[str, list] | None:
        """Log structure."""
        return self._log
    
    @property
    def params(self) -> LogEntry | Sequence[LogEntry] | None:
        """Log parameter entries, with format and units."""
        return self._params
    
    def initialize(self, params: LogEntry | Sequence[LogEntry]) -> None:
        """
        Initializes the Log structure with the specified parameter entries.
        Inside the Log, the parameters are accessible as keys.

        Args:
            params (LogEntry | Sequence[LogEntry]):
                Sequence with the parameter entries.
        """
        def make_log(params: LogEntry | Sequence[LogEntry]) -> dict:
            """
            Creates the log structure.

            Args:
                params (LogEntry | Sequence[LogEntry]):
                    Sequence with the parameter entries.
            
            Returns:
                log (dict[str, list]):
                    Log structure with the parameter entries.
            """
            params = (params,) if isinstance(params, LogEntry) else params
            return {p.entry: [] for p in params}
    
        self._log = make_log(params)
        self._params = params

    def insert(self, entries: LogEntry | Sequence[LogEntry]) -> None:
        """
        Inserts the specified new entries in the Log.

        Args:
            entries (LogEntry | Sequence[LogEntry]): New entries for the Log.
        """
        entries = (entries,) if isinstance(entries, LogEntry) else entries
        for entry in entries:
            self._log[entry.entry] = []
        self._params += entries
    
    def update(self, values: Sequence[tuple[str, Any]]) -> None:
        """
        Updates the entries inside the Log by appending values.

        Args:
            values (Sequence[tuple[str, Any]]):
                Sequence containing the name and the value of the
                parameter to add to the database inside the Log.
        """
        for (entry, value) in values:
            self._log[entry].append(value)
    
    def add_entry_values(self, entry: str, values: Sequence[Any]) -> None:
        """
        Add the specified sequence of values to the Log entry.

        Args:
            entry (str): Entry name of the Log.
            values (Sequence[Any]): Values for the entry.
        """
        self._log[entry] += values
    
    def replace_entry_values(self, entry: str, values: Sequence[Any]) -> None:
        """
        Replace the Log entry values with the specified sequence.

        Args:
            entry (str): Entry name of the Log.
            values (Sequence[Any]): Values for the entry.
        """
        self._log[entry] = values
    
    def to_dataframe(self) -> DataFrame:
        """
        Converts the dict Log to a Pandas DataFrame.

        Returns:
            output (DataFrame):
                Deepcopy of the Log converted to DataFrame.
        """
        return DataFrame(deepcopy(self._log))


def create_log(
    params: LogEntry | Sequence[LogEntry],
    name: str | None = None,
) -> Log:
    """
    Initializes a Log instance with the given parameters to manage
    the logging of the IROS procedure for the WFM cameras.

    Args:
        params (LogEntry | Sequence[LogEntry]):
            Sequence with the specified parameter entries.
        name (str | None, optional (default=`None`)):
            Log name.

    Returns:
        output (Log):
            Log instance containing the initialized log structure.
    """
    log = Log(name)
    log.initialize(params)
    return log


@dataclass(frozen=True)
class DataLoader(SimulationDataLoader):
    """
    Container for WFM coded mask simulation data.

    The class provides access to photon events and instrument configuration
    from a FITS file containing WFM simulation data for a single camera.

    This class inherits from bloodmoon's `SimulationDataLoader`, and allows
    for data filtering in the photons energy and incoming direction.

    Attributes:
        filepath (Path):
            Path to the FITS file.
        E_min (int | float | None):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None):
            Input photons RA/Dec in [deg] to filter out.

    Properties:
        DLdata (FITS_rec):
            Photon event data from FITS extension 1, eventually filtered.
    """
    E_min: int | float | None
    E_max: int | float | None
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None
    
    @cached_property
    def DLdata(self) -> FITS_rec:
        if not any((self.E_min, self.E_max, self.coords)):
            return self.data
        
        rec = deepcopy(self.data)
        return filter_data(
            data=rec,
            E_min=self.E_min,
            E_max=self.E_max,
            coords=self.coords,
        )


def get_data(
    filepath: str | Path,
    *,
    E_min: int | float | None = None,
    E_max: int | float | None = None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None = None,
) -> DataLoader:
    """
    Checks validity of filepath and intializes DataLoader.

    Args:
        filepath (Path):
            Path to the FITS file.
        E_min (int | float | None, optional (default=`None`)):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None, optional (default=`None`)):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None, optional (default=`None`)):
            Input photons RA/Dec in [deg] to filter out.

    Returns:
        output (DataLoader):
            DataLoader instance with filterable photons list data.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if _exists_valid(filepath):
        sdl = DataLoader(
            filepath=filepath,
            E_min=E_min,
            E_max=E_max,
            coords=coords,
        )
        return sdl


@dataclass(frozen=True)
class CatalogueLoader(SimulationDataLoader):
    """
    Container for WFM coded mask sources catalog.

    The class provides access to the catalog and instrument configuration
    from a FITS file containing WFM simulation data for a single camera.

    This class inherits from bloodmoon's `SimulationDataLoader`, and allows
    for catalog filtering in the brightness and flux channels.

    Attributes:
        filepath (Path):
            Path to the FITS file.
        n (int | tuple[int, int]):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        F_min (int | float | None):
            Minimum flux range in [ph/cm2/s] for the data filtering.
        F_max (int | float | None):
            Maximum flux range in [ph/cm2/s] for the data filtering.

    Properties:
        DLdata (FITS_rec):
            Catalog data from FITS extension 1, eventually filtered.
    """
    n: int | tuple[int, int] | None
    F_min: int | float | None
    F_max: int | float | None
    
    @cached_property
    def DLdata(self) -> FITS_rec:
        if not any((self.n, self.F_min, self.F_max)):
            return self.data
        
        rec = deepcopy(self.data)
        return filter_catalogue(
            catalogue=rec,
            n=self.n,
            F_min=self.F_min,
            F_max=self.F_max,
        )


def get_catalogue(
    filepath: str | Path,
    *,
    n: int | tuple[int, int] | None = None,
    F_min: int | float | None = None,
    F_max: int | float | None = None,
) -> CatalogueLoader:
    """
    Checks validity of filepath and intializes CatalogueLoader.

    Args:
        filepath (Path):
            Path to the FITS file.
        n (int | tuple[int, int] | None, optional (default=`None`)):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        F_min (int | float | None, optional (default=`None`)):
            Minimum flux range in [ph/cm2/s] for the data filtering.
        F_max (int | float | None, optional (default=`None`)):
            Maximum flux range in [ph/cm2/s] for the data filtering.

    Returns:
        output (CatalogueLoader):
            CatalogueLoader instance with filterable sources catalogue.
    
    Raises:
        ValueError: If `n` or the flux range are both specified for catalogues filtering.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    
    if _exists_valid(filepath):
        if n and any((F_min, F_max)):
            raise ValueError("Specify either 'n' OR the flux range to filter the catalogue.")
        
        sdl = CatalogueLoader(
            filepath=filepath,
            n=n,
            F_min=F_min,
            F_max=F_max,
        )
        return sdl


def fit_WCS(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    pixels: list[tuple[int, int]] | None = None,
    grid_step: int = 200,
) -> WCS:
    """
    Fit the WCS for a camera of the WCS fitting given RA/DEC
    and sky pixels.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (DataLoader):
            DataLoader instance for the given camera.
        pixels (list[tuple[int, int]], optional (default=`None`)):
            List of pixels position (row, col) for the WCS fit.
        grid_step (int, optional (default=`200`)):
            Sky grid points along each axis for computing the WCS fit.
    
    Returns:
        output (WCS):
            WCS instance with info on the coords fit.
    """
    n, m = camera.shape_sky
    pxs = pixels if pixels else [
        (grid_step * y, grid_step * x) for y in range(1, n // grid_step) for x in range(1, m // grid_step)
    ]

    coords = [pos2equatorial(sdl, camera, *pos) for pos in pxs]
    # WARNING: the next is not a typo, WCS wants the px indexes as (x, y)
    coord_pxs = tuple(np.array([px[idx] for px in pxs]) for idx in (1, 0))
    coord_radec = SkyCoord(
        ra=np.array([c.ra for c in coords]),
        dec=np.array([c.dec for c in coords]),
        frame="icrs",
        unit="deg",
    )
    wcs = fit_wcs_from_points(
        xy=coord_pxs,
        world_coords=coord_radec,
        projection="TAN",
        sip_degree=1,
        proj_point=SkyCoord(*sdl.pointings["z"], frame="icrs", unit="deg"),
    )
    return wcs


# end