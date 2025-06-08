"""
Module for data handling.
"""

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd

from bloodmoon.types import CoordEquatorial
from bloodmoon.coords import pos2equatorial
from bloodmoon.io import SimulationDataLoader
from bloodmoon.io import _exists_valid
from bloodmoon.mask import CodedMaskCamera

from .types import LogEntry
from .filtering import filter_data

__all__ = []


class Log:
    """
    Structure for data logging.\n

    This class creates a flexible database structure aimed to store
    IROS output parameters.
    It provides methods for initializing the log, structured as a
    dictionary with a specific subdivision, and to update the entries.
    The dict structure is organized using a template that contains the
    data, the data format, and the data units (e.g., 'mm', 'deg', etc.).
    This structure is repeated for the two cameras of the WFM instrument.
    
    Attributes:
        camA_ID (str): WFM camera A ID.
        camB_ID (str): WFM camera B ID.
    """
    def __init__(
        self,
        camA_ID: str,
        camB_ID: str,
    ) -> None:
        self.cams = (camA_ID, camB_ID)
        self._log = None
    
    def _template(self, frmt: str, unit: str) -> dict:
        """Data entry template."""
        return {"data": [], "format": frmt, "unit": unit}

    def _make_log(self, params: Sequence[LogEntry]) -> dict:
        """Creates the log structure."""        
        structure = {
            log.entry: self._template(log.frmt, log.unit)
            for log in params
        }
        return {camID: deepcopy(structure) for camID in self.cams}
    
    def initialize(self, params: Sequence[LogEntry]) -> dict:
        """
        Initializes the Log structure with the specified parameter entries.
        The Log is generated as a dict containing two macro-dict, one for
        each camera of the Wide Field Monitor. Inside these macro-dict, the
        parameters are accessible as keys.
        Each parameter entry contains:
            - a 'data' field where the data is stored;
            - a 'format' field where the data type is specified;
            - a 'unit' field with the data units in SI format.

        Args:
            params (Sequence[LogEntry]):
                Sequence with all the parameter entries for the two WFM cameras.
        
        Returns:
            log (dict):
                Log structure with the parameter entries for the two WFM cameras.
        """
        self._log = self._make_log(params)
        return self._log
    
    def update(
        self,
        cameraID: str,
        values: Sequence[tuple[str, int | float]],
    ) -> None:
        """
        Updates the entries of the specified WFM camera inside the Log.

        Args:
            cameraID (str):
                WFM camera to update.
            values (Sequence[tuple[str, int | float]]):
                Sequence containing the name and the value of the
                parameter to add to the database inside the Log.
        """
        for (entry, value) in values:
            self._log[cameraID][entry]["data"].append(value)


def create_log(*, camA_ID: str, camB_ID: str) -> Log:
    """
    Initializes a Log instance to manage the logging of
    IROS parameters for the specified WFM cameras.

    Args:
        camA_ID (str): WFM camera A ID.
        camB_ID (str): WFM camera B ID.

    Returns:
        output (Log):
            Log instance containing the initialized log structure.
    """
    return Log(camA_ID, camB_ID)


def IROS_log() -> Log:
    """
    
    """
    raise NotImplementedError


def IROS_sources_log() -> Log:
    """
    
    """
    raise NotImplementedError
























class FlexibleSDL(SimulationDataLoader):
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
        SDLdata (np.recarray):
            Photon event data from FITS extension 1.
    """
    def __init__(
        self,
        filepath: Path,
        E_min: int | float | None,
        E_max: int | float | None,
        coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
    ) -> None:
        super().__init__(filepath=filepath)

        self.E_min = E_min
        self.E_max = E_max
        self.coords = coords
    
    @cached_property
    def SDLdata(self) -> np.recarray:
        """Photon event data, eventually filtered."""
        if not any((self.E_min, self.E_max, self.coords)):
            return self.data
        
        rec = deepcopy(self.data)
        return filter_data(
            data=rec,
            E_min=self.E_min,
            E_max=self.E_max,
            coords=self.coords,
        )


def simulation(
    filepath: str | Path,
    *,
    E_min: int | float | None = None,
    E_max: int | float | None = None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None = None,
) -> FlexibleSDL:
    """
    Checks validity of filepath and intializes FlexibleSDL.

    Args:
        filepath (Path):
            Path to the FITS file.
        E_min (int | float | None, optional (default=None)):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None, optional (default=None)):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None, optional (default=None)):
            Input photons RA/Dec in [deg] to filter out.

    Returns:
        output (FlexibleSDL):
            FlexibleSDL instance with filterable photons list data.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if _exists_valid(filepath):
        sdl = FlexibleSDL(
            filepath=filepath,
            E_min=E_min,
            E_max=E_max,
            coords=coords,
        )
        return sdl

















def fit_WCS(
    camera: CodedMaskCamera,
    sdl: SimulationDataLoader,
    pixels: list[tuple[int, int]] = None,
    grid_step: int = 200,
) -> WCS:
    """
    Fit the WCS for a camera of the WCS fitting given RA/DEC
    and sky pixels.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (SimulationDataLoader):
            SimulationDataLoader instance for the given camera.
        pixels (list[tuple[int, int]], optional (default=None)):
            List of pixels position (row, col) for the WCS fit.
        grid_step (int, optional (default=200)):
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


def WFMcomposition(
    skyA_path: str | Path,
    skyB_path: str | Path,
) -> tuple[NDArray, NDArray, WCS]:
    """
    Performs the composition of the WFM cameras skies and significances,
    including the reprojection of the World Coordinates System for RA/Dec.

    Specifically, it:
        a. Opens the skies FITS file
        b. Finds the optimal WCS fit and sky shape for the composition
        c. Reprojects and sums the two skies making the composition
        d. Reprojects the two SNRs and takes the max

    Args:
        skyA_path (str, Path):
            File path for the camera A sky.
        skyB_path (str, Path):
            File path for the camera B sky.
    
    Returns:
        sky (NDArray):
            WFM cameras sky composition.
        snr (NDArray):
            WFM composed sky significance computed by taking
            the max of the two cameras individual sky SNR.
        wcs (WCS):
            Output reprojected WCS fit.

    Notes:
        - If the WCS fit keys are not present in the camera skies headers,
          a TypeError will be raised from `find_optimal_celestial_wcs()`:
        >>> TypeError: "WCS does not have celestial components."
    """
    with fits.open(skyA_path) as hduA, fits.open(skyB_path) as hduB:
        skies = (hduA[1], hduB[1])
        snrs = (hduA[2], hduB[2])
    
        print("# Composing WFM skies...")
        wcs_out, shape_out = find_optimal_celestial_wcs(input_data=skies)
        sky_comp, _ = reproject_and_coadd(
            input_data=skies,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="sum",
        )
        snr_comp, _ = reproject_and_coadd(
            input_data=snrs,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="max",
        )
    
    print("# WFM composition completed!")
    return sky_comp, snr_comp, wcs_out


# end