"""
Data filters for photons energy range, sources flux and sources positions.
"""

from typing import Sequence

import numpy as np
from astropy.io.fits.fitsrec import FITS_rec

from bloodmoon.types import CoordEquatorial

__all__ = [
    "select_source_photons", "filter_data", "flux_filter",
    "source_filter", "filter_catalogue",
]


def select_source_photons(
    coords: CoordEquatorial | Sequence[CoordEquatorial],
    data: FITS_rec,
    verbose: bool = True,
) -> FITS_rec:
    """
    Selects the photon events in the input `data` relative to
    the selected sources RA/Dec coords.

    Args:
        coords (CoordEquatorial | Sequence[CoordEquatorial]):
            Input photons RA/Dec in [deg] to select from `data`.
        data (FITS_rec):
            Input simulated data container.
        verbose (bool, optional (default=`True`)):
            If `True`, prints out the selected events number with
            respect to the total number of photons in `data`.
    
    Returns:
        output (FITS_rec): Output filtered data container.
    """
    mask = np.ones(len(data), dtype=bool)
    coords_ = (coords,) if isinstance(coords, CoordEquatorial) else coords
    for c in coords_:
        mask &= (
            (np.isclose(data['RA'], c.ra) & np.isclose(data['DEC'], c.dec))
        )
    selected = data[mask]
    if verbose:
        print(f'Selected {len(selected)}/{len(data)} photons.')
    return selected


def filter_data(
    data: FITS_rec,
    *,
    E_min: int | float | None,
    E_max: int | float | None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
) -> FITS_rec:
    """
    Filters the input `data` record based on the photons energy
    and/or incoming direction.
    
    Args:
        data (FITS_rec):
            Input simulated data container.
        E_min (int | float | None):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None):
            Input photons RA/Dec in [deg] to filter out.
    
    Returns:
        output (FITS_rec): Output filtered data container.
    """
    mask = np.ones(len(data), dtype=bool)

    if E_min is not None:
        mask &= (data["ENERGY"] > E_min)
    if E_max is not None:
        mask &= (data["ENERGY"] < E_max)
    
    if coords is not None:
        for c in ((coords,) if isinstance(coords, CoordEquatorial) else coords):
            mask &= ~(np.isclose(data["RA"], c.ra) & np.isclose(data["DEC"], c.dec))
    
    return data[mask]


def flux_filter(
    data: FITS_rec,
    F_min: int | float | None,
    F_max: int | float | None,
) -> FITS_rec:
    """
    Filters the input `data` record based on the sources flux.
    
    Args:
        data (FITS_rec):
            Input simulated data container.
        F_min (int | float | None):
            Minimum flux range in [ph/cm2/s] for the data filtering.
        F_max (int | float | None):
            Maximum flux range in [ph/cm2/s] for the data filtering.
    
    Returns:
        output (FITS_rec): Output filtered data container.
    """
    mask = np.ones(len(data), dtype=bool)
    if F_min is not None:
        mask &= (data["FLUX"] > F_min)
    if F_max is not None:
        mask &= (data["FLUX"] < F_max)
    return data[mask]


def source_filter(
    data: FITS_rec,
    n: int | tuple[int, int],
) -> FITS_rec:
    """
    Select the `n` brightest sources from the input catalogue `data`,
    or a given interval of sources.

    Args:
        data (FITS_rec):
            Input simulated data container.
        n (int | tuple[int, int]):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.

    Returns:
        output (FITS_rec): Output filtered data container.
    
    ## Notes:
        - `n` follows the std Python indexing rules.
    """
    sorted_rec = np.sort(data, order="NPHOTONS")[::-1]
    runs = len(sorted_rec) // len(np.unique(sorted_rec["ID"]))
    return sorted_rec[:runs * n] if isinstance(n, int) else sorted_rec[runs * n[0] : runs * n[1]]


def filter_catalogue(
    catalogue: FITS_rec,
    *,
    n: int | tuple[int, int] | None,
    F_min: int | float | None = None,
    F_max: int | float | None = None,
) -> FITS_rec:
    """
    Filters the input `catalogue` record based on the sources fluence OR flux.
    If `n` is given, it selects the `n` brightest sources from the input
    record, or a given interval of sources. If a flux range is given, it
    filters the input record for a given flux range.
    
    Args:
        catalogue (FITS_rec):
            Input simulated data container.
        n (int | tuple[int, int] | None):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        F_min (int | float | None, optional (default=`None`)):
            Minimum flux range in [ph/cm2/s] for the data filtering.
        F_max (int | float | None, optional (default=`None`)):
            Maximum flux range in [ph/cm2/s] for the data filtering.
    
    Returns:
        output (FITS_rec): Output filtered data container.
    
    Raises:
        ValueError: If `n` or the flux range are both specified for catalogues filtering.
    """
    if n and any((F_min, F_max)):
        raise ValueError("Specify either 'n' OR the flux range to filter the catalogue.")
    
    if n is not None:
        return source_filter(catalogue, n)
    elif any((F_min, F_max)):
        return flux_filter(catalogue, F_min, F_max)
    
    return catalogue


# end