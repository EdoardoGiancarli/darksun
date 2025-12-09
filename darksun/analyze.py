"""
IROS output data management and computation.
"""

from functools import lru_cache

import numpy as np
from pandas import DataFrame
from astropy.io.fits import Column, BinTableHDU
from astropy.io.fits.fitsrec import FITS_rec

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import equatorial2shift
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2angle
from bloodmoon.optim import _detector_footprint_cached
from bloodmoon.optim import _mask_pattern_projection

from .types import LogEntry
from .data import DataLoader
from .data import CatalogueLoader
from .data import Log

__all__ = [
    "run_IROS", "eq_coords_errors", "camera_area_unit", "get_effective_area",
    "compute_parameters", "data_screening", "catalogue_comparison",
]


def run_IROS(
    *args,
    **kwargs,
) -> None:
    """
    !!! As of now, the two `run_IROS` wrapper for the IROS pipeline have been
        inserted in the two respective folder for the singleCAM and doubleCAM
        based analyses.
        * `IROS/doubleCAM_iros.py`,
        * `singleCAM_IROS/singleCAM_iros.py`,


    Runs the IROS (Iterative Removal of Sources) loop and stores the output.

    This wrapper iteratively removes the detected sources candidates from the sky until
    either the maximum number of iterations is reached or the SNR threshold is met.
    At each iteration, two logs for the specified LEM-X module coded-mask cameras
    are updated with the following candidates estimated parameters:

        - camera local frame sky-coordinates shifts along the (x, y)
          axes wrt the coded-mask camera optical axis, in [mm]*
        - fluence, in [ph]**
        - significance at the selection
    
    * The candidates shifts errors at upscaling `(x, y)=(1, 1)` are assumed to be
      `5 arcmin` along x and `60 arcmin` along y.
    ** The candidates fluence is assumed to follow a Poissonian statistics, so the
       fluence error is the square root of the fluence.
    
    Args:
        

    Returns:
        
    """
    raise NotImplementedError


def eq_coords_errors(
    camera: CodedMaskCamera,
    shiftx: float,
    dshiftx: float,
    shifty: float,
    dshifty: float,
) -> tuple[float, float]:
    """Computes RA/DEC source errors."""
    # TODO: compute errs from transform
    raise NotImplementedError


@lru_cache(maxsize=1)
def camera_area_unit(camera: CodedMaskCamera) -> float:
    """
    Computes the detector unit area value in [cm^2], corrected for
    the camera layers and detectors efficiency photons absorption.

    Args:
        camera (CodedMaskCamera): Instance with camera geometry info.
    
    Returns:
        output (float): Camera pixel area value in [cm^2].
    """
    # LEM-X camera layers and detector efficiency correction factors
    # here we consider the absorption action of (photons absorption @ 8keV):
    #   - 12.5um Kapton layer (coded-mask camera MLI)
    #   - 25um Be layer (micro-meteoroid filter)
    #   - 300nm Al layers (total, between MLI and micro-meteoroid filter)
    #   - detector dead layers
    #   - SDD QE (i.e., detector absrp efficiency)
    # https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/Effective_area_and_Sens_2D.ipynb
    corrs: dict[str, float] = {
        'Kapton_layer': 0.98945,
        'Be_layer': 0.99527,
        'Al_layer': 0.99614,
        'dead_layers': 0.974,
        'QE': 0.999,
    }
    qe_factor: float = np.prod(np.array(list(corrs.values())))
    # pixel area in [cm^2]
    pixel_area: float = (
        1e-2 * camera.specs.mask_deltax * camera.specs.mask_deltay / np.prod(camera.upscale_f)
    )
    return qe_factor * pixel_area


def get_effective_area(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
) -> float:
    """
    Computes the source effective area on detector in [cm^2].

    Args:
        camera (CodedMaskCamera):
            Instance with camera geometry info.
        shift_x (float):
            Source coord along fine direction in camera local-frame system.
        shift_y (float):
            Source coord along coarse direction in camera local-frame system.
        vignetting (bool, optional (default=`True`)):
            If `True`, the source model will simulate mask vignetting effects.
    
    Returns:
        output (float):
            Source effective area value on detector in [cm^2].
    """
    # mask pattern projection WTO detector sp. res.
    sg = _mask_pattern_projection(
        camera, shift_x, shift_y, vignetting, False,
    )
    # extract detector
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    return detector.sum() * camera_area_unit(camera)


def compute_parameters(
    log: Log,
    camera: CodedMaskCamera,
    sdl: DataLoader,
    vignetting: bool = True,
) -> Log:
    """
    Computes parameters for IROS reconstructed candidates.
    The input LEM-X camera Log is updated with the following parameters:

        - candidates image pixel indexes
        - LEM-X camera local frame (x, y) angular coordinates and errors, in [deg]
        - candidate equatorial coordinates (RA, Dec) and errors, in [deg]
        - candidate photons rate and error, in [ph/s]
        - candidate photons flux and error, in [ph/cm2/s]

    Args:
        log (Log):
            Log instance with IROS data output from `run_IROS()`.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (DataLoader):
            Data container instance for chosen LEM-X coded-mask camera.
        vignetting (bool, optional (default=`True`)):
            If `True`, the source model will simulate mask vignetting effects.

    Returns:
        output (Log):
            Log instance with computed parameters for each candidate.
    """
    # retrieve observation exposure [s] and IROS data
    EXPOSURE = sdl.header["EXPOSURE"]
    
    shifts_x, dshifts_x = log.log['shift_x'], log.log['dshift_x']
    shifts_y, dshifts_y = log.log['shift_y'], log.log['dshift_y']
    fluences, dfluences = log.log['fluence'], log.log['dfluence']

    # insert new entries
    params = (
        LogEntry('y', 'J', 'px'), LogEntry('x', 'J', 'px'),
        LogEntry('angle_x', 'D', 'deg'), LogEntry('dangle_x', 'D', 'deg'),
        LogEntry('angle_y', 'D', 'deg'), LogEntry('dangle_y', 'D', 'deg'),
        LogEntry('ra', 'D', 'deg'), # LogEntry('dra', 'D', 'deg'),
        LogEntry('dec', 'D', 'deg'), # LogEntry('ddec', 'D', 'deg'),
        LogEntry('rate', 'D', 'ph/s'), LogEntry('drate', 'D', 'ph/s'),
        LogEntry('flux', 'D', 'ph/cm2/s'), LogEntry('dflux', 'D', 'ph/cm2/s'),
    )
    log.insert(params)

    # compute parameters
    y, x = zip(
        *tuple(shift2pos(camera, sx, sy) for sx, sy in zip(shifts_x, shifts_y))
    )
    log.add_entry_values('y', list(y))
    log.add_entry_values('x', list(x))

    thetas_x, thetas_y = map(
        lambda shifts: tuple(shift2angle(camera, s) for s in shifts),
        (shifts_x, shifts_y),
    )
    dthetas_x, dthetas_y = map(
        lambda dshifts: tuple(abs(shift2angle(camera, ds)) for ds in dshifts),
        (dshifts_x, dshifts_y),
    )
    log.add_entry_values('angle_x', list(thetas_x))
    log.add_entry_values('angle_y', list(thetas_y))
    log.add_entry_values('dangle_x', list(dthetas_x))
    log.add_entry_values('dangle_y', list(dthetas_y))

    ras, decs = zip(
        *tuple(shift2equatorial(sdl, camera, sx, sy) for sx, sy in zip(shifts_x, shifts_y))
    )
    #dras, ddecs = zip(
    #    *tuple(
    #        eq_coords_errors() for _ in zip()
    #    )
    #)
    log.add_entry_values('ra', list(ras))
    log.add_entry_values('dec', list(decs))
    #log.add_entry_values('dra', list(dras))
    #log.add_entry_values('ddec', list(ddecs))

    rates = [f / EXPOSURE for f in fluences]
    drates = [df / EXPOSURE for df in dfluences]
    log.add_entry_values('rate', rates)
    log.add_entry_values('drate', drates)

    effective_area = lambda sx, sy: get_effective_area(camera, sx, sy, vignetting)
    fluxes = [
        f / (effective_area(sx, sy) * EXPOSURE) for f, sx, sy in zip(fluences, shifts_x, shifts_y)
    ]
    dfluxes = [
        df / (effective_area(sx, sy) * EXPOSURE) for df, sx, sy in zip(dfluences, shifts_x, shifts_y)
    ]
    log.add_entry_values('flux', fluxes)
    log.add_entry_values('dflux', dfluxes)

    return log


def data_screening(
    data: DataFrame,
    groupby: str,
    column: str,
) -> DataFrame:
    """
    Groups the DataFrame rows by the `groupby` key and selects the
    entry with the maximum value in the specified column.

    The resulting DataFrame is sorted by its index. If there are
    multiple rows with the same maximum value within a group,
    `df.idxmax()` returns the index of the first occurrence.

    Args:
        data (DataFrame):
            The input pandas DataFrame.
        groupby (str):
            The name of the column to group the DataFrame by.
        column (str):
            The name of the column for which to find the maximum
            value within each group.

    Returns:
        output (DataFrame):
            Filtered DataFrame containing only the rows with the maximum
            value in the specified `column` for each group, sorted by index.

    Examples:
        >>> # assuming `df` as a pandas DataFrame
        >>> df
            category   value   other_data
        0          A      10            x
        1          A      25            y
        2          B      15            z
        3          B      30            w
        4          A       5            p
        ...
        >>> data_screening(df, groupby='category', column='value')
            category   value   other_data
        1          A      25            y
        3          B      30            w
    """
    return (
        data.loc[
            data.groupby(groupby)[column].idxmax()
        ].sort_index()
    )


def catalogue_comparison(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
    screening: bool = False,
) -> Log:
    """
    Compares the reconstructed IROS data with the catalogue,
    associating the candidates with the known sources.

    The association is performed by comparing the local frame sky-coords
    `shifts` of the catalogue sources with the shifts and relative errorboxes
    of the decoded candidates at `3` sigma level.
    If no catalogue sources are found, the candidates are labeled as new
    sources, with the respective LEM-X coded-mask camera ID.

    Args:
        log (Log):
            IROS data output.
        catalogue (CatalogueLoader):
            Catalogue data for the LEM-X coded-mask camera.
        sdl (DataLoader):
            Data container instance for chosen LEM-X coded-mask camera.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        screening (bool, optional (default=`False`)):
            If `True`, the repeating sources in the database
            are screened by significance comparison.

    Returns:
        output (Log):
            Log with the updated entries for the input IROS candidates,
            featuring sources IDs and catalogues calibrated fluxes.
    """
    def extend_catalogue(rec: FITS_rec) -> FITS_rec:
        """Adds sources local frame angular coords to catalogue."""
        extended = [
            Column(name=name, format=rec.columns[name].format, array=rec[name])
            for name in rec.names
        ]
        ssx, ssy = zip(
            *tuple(
                equatorial2shift(sdl, camera, ra, dec) for ra, dec in zip(rec['RA'], rec['DEC'])
            )
        )
        shifts = [
            Column(name='SHIFT_X', format='D', array=np.array(ssx)),
            Column(name='SHIFT_Y', format='D', array=np.array(ssy)),
        ]
        return BinTableHDU.from_columns(extended + shifts).data
    
    # set up
    KEYMAP = {
        'cxb_tag': 'gctr_diffuse',
        'NEW_ID': 1,
    }
    DATABASE = extend_catalogue(catalogue.DLdata)

    # catalogue comparison and sources association
    def candidate_association(
        shiftx: float,
        dshiftx: float,
        shifty: float,
        dshifty: float,
        sigma: int | float = 3,
    ) -> tuple[str, float]:
        """Candidate association from catalogue."""

        def closest_source(batch: FITS_rec) -> int:
            """Returns candidate's closer catalogue source index."""
            arg = np.argmin(
                np.square(batch['SHIFT_X'] - shiftx) + np.square(batch['SHIFT_Y'] - shifty)
            )
            return arg
        
        def brightest_source(batch: FITS_rec) -> int:
            """Returns brightest catalogue source index within errorbox."""
            arg = np.argmax(batch['FLUX'])
            return arg
    
        box = (
            (DATABASE['SHIFT_X'] > shiftx - sigma * dshiftx) &
            (DATABASE['SHIFT_X'] < shiftx + sigma * dshiftx) &
            (DATABASE['SHIFT_Y'] > shifty - sigma * dshifty) &
            (DATABASE['SHIFT_Y'] < shifty + sigma * dshifty) &
            (DATABASE['ID'] != KEYMAP['cxb_tag'])
        )
        associated_batch = DATABASE[box]

        if not any(associated_batch):
            sourceID = f'lemx-{log.name.lower()}S{KEYMAP['NEW_ID']}'
            flux = -1.0
            sourceID_brightest = sourceID
            KEYMAP['NEW_ID'] += 1
        elif len(associated_batch) == 1:
            sourceID = associated_batch['ID'][0]
            flux = associated_batch['FLUX'][0]
            sourceID_brightest = sourceID
        else:
            arg = closest_source(associated_batch)
            sourceID = associated_batch['ID'][arg]
            flux = associated_batch['FLUX'][arg]
            arg_brightest = brightest_source(associated_batch)
            sourceID_brightest = associated_batch['ID'][arg_brightest]

        return sourceID, flux, sourceID_brightest
    
    # update Log
    params = (
        LogEntry('ID', '20A', ''),
        LogEntry('catalogue_flux', 'D', 'ph/cm2/s'),
        LogEntry('ID_brightest', '20A', ''),
    )
    log.insert(params)

    print("# Comparing with Catalogue...")
    # initial sources association
    for sx, dsx, sy, dsy in zip(
        log.log['shift_x'], log.log['dshift_x'],
        log.log['shift_y'], log.log['dshift_y'],
    ):
        sourceID, flux, sourceID_brightest = candidate_association(sx, dsx, sy, dsy)
        log.update(
            values=(('ID', sourceID), ('catalogue_flux', flux), ('ID_brightest', sourceID_brightest))
        )
    
    # sources screening based on significance
    if screening:
        df = data_screening(log.to_dataframe(), 'ID', 'snr')
        for col, series in df.items():
            log.replace_entry_values(col, list(series))

    print("# Successful comparison!")
    return log


# end