"""
IROS output data management and computation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve
from pandas import DataFrame
from astropy.io.fits.fitsrec import FITS_rec
from tqdm import tqdm

from bloodmoon.mask import _detector_footprint
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import equatorial2shift
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2angle
from bloodmoon.images import _shift
from bloodmoon.images import _rbilinear
from bloodmoon.optim import _wfm_psfy_kernel_cached
from bloodmoon.optim import apply_vignetting

from .types import LogEntry
from .data import DataLoader
from .data import CatalogueLoader
from .data import Log
from .data import create_log
from .optim import iros

__all__ = [
    "run_IROS", "compute_parameters",
    "data_screening", "catalogue_comparison",
]


def run_IROS(
    *,
    camera: CodedMaskCamera,
    sdl_camA: DataLoader,
    sdl_camB: DataLoader,
    max_iterations: int = 25,
    snr_threshold: int | float = 10,
    vignetting: bool = True,
    psfy: bool = True,
    id_camA: str | None = None,
    id_camB: str | None = None,
) -> tuple[tuple[Log, Log], tuple[NDArray, NDArray]]:
    """
    Runs the IROS (Iterative Removal of Sources) loop and stores the output.

    This wrapper iteratively removes the detected sources candidates from the sky until
    either the maximum number of iterations is reached or the SNR threshold is met.
    At each iteration, two logs for the coded-mask cameras of the Wide Field Monitor
    are updated with the following candidates estimated parameters:

        - camera local frame sky-coordinates shifts along the (x, y)
          axes wrt the coded-mask camera optical axis, in [mm]
        - fluence, in [ph]
        - significance at the selection
    
    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl_camA (DataLoader):
            DataLoader instance for camera A.
        sdl_camB (DataLoader):
            DataLoader instance for camera B.
        max_iterations (int, optional (default=`25`)):
            Maximum number of iterations for the IROS loop.
        snr_threshold (int | float, optional (default=`5`)):
            Minimum SNR value required to continue the iterative source removal process.
        vignetting (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate detector
            position reconstruction effects.
        id_camA (str | None, optional (default=`None`)):
            WFM camera A name (for the Log).
        id_camB (str | None, optional (default=`None`)):
            WFM camera B name (for the Log).

    Returns:
        output (tuple[tuple[Log, Log], tuple[NDArray, NDArray]]):
            - logs (tuple[Log, Log]):
                Camera `A` and `B` logs with metadata and results from IROS.
            - residuals (tuple[NDArray, NDArray]):
                Sky residuals for the WFM after IROS.
    """
    # generate IROS output log
    params = (
        LogEntry('shift_x', 'D', 'mm'), LogEntry('shift_y', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('snr', 'D', ''),
    )
    log_camA = create_log(params, id_camA)
    log_camB = create_log(params, id_camB)

    # init and run IROS loop
    print("# Initializing Loop...")
    loop = iros(
        camera=camera,
        sdl_cam1a=sdl_camA,
        sdl_cam1b=sdl_camB,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
    )
    print("# Looping around the FOV...")
    for candidates, residuals in tqdm(loop):
        parA, parB = candidates
        log_camA.update(
            tuple((p.entry, val) for p, val in zip(params, parA))
        )
        log_camB.update(
            tuple((p.entry, val) for p, val in zip(params, parB))
        )
    
    return (log_camA, log_camB), residuals


def compute_parameters(
    log: Log,
    camera: CodedMaskCamera,
    sdl: DataLoader,
    vignetting: bool = True,
    psfy: bool = True,
) -> Log:
    """
    Computes parameters for IROS reconstructed candidates.
    The input WFM camera Log is updated with the following parameters:

        - candidates output parameters errors (local frame (x, y) sky-shifts
          coords [mm] and fluence [ph])
        - candidates image pixel indexes
        - WFM camera local frame (x, y) angular coordinates and errors, in [deg]
        - candidate equatorial coordinates (RA, Dec) and errors, in [deg]
        - candidate photons rate and error, in [ph/s]
        - candidate photons flux and error, in [ph/cm2/s]
    
    The coords errors are assumed to be 1 arcmin along x and 60 arcmin along y.

    Args:
        log (Log):
            Log instance with IROS data output from `run_IROS()`.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (DataLoader):
            Data container instance for chosen WFM coded-mask camera.
        vignetting (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, optional (default=`True`)):
            If `True`, the model used for optimization will simulate detector
            position reconstruction effects.

    Returns:
        output (Log):
            Log instance with computed parameters for each candidate.
    """
    # retrieve observation data (px area [cm^2], camera exposure [s])
    UPX, UPY = camera.upscale_f
    PX_AREA = (
        1e-2 * camera.specs["mask_deltax"] * camera.specs["mask_deltay"] / np.prod((UPX, UPY))
    )
    EXPOSURE = sdl.header["EXPOSURE"]
    
    shifts_x = log.log['shift_x']
    shifts_y = log.log['shift_y']
    fluences = log.log['fluence']

    # coded-mask sensitivity along the (x, y) axis
    # - TODO: insert correct camera sensitivity estimation (this is a proxy,
    #         dthetax ~ 5 arcmin, dthetay ~ 60 arcmin at (upx, upy) = (1, 1))
    DTHETA_X = 5.0 / UPX        # [arcmin]
    DTHETA_Y = 60.0 / UPY       # [arcmin]

    # insert new entries
    params = (
        LogEntry('dshift_x', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
        LogEntry('dfluence', 'D', 'ph'),
        LogEntry('y', 'J', 'px'), LogEntry('x', 'J', 'px'),
        LogEntry('angle_x', 'D', 'deg'), LogEntry('dangle_x', 'D', 'deg'),
        LogEntry('angle_y', 'D', 'deg'), LogEntry('dangle_y', 'D', 'deg'),
        LogEntry('ra', 'D', 'deg'), LogEntry('dra', 'D', 'deg'),
        LogEntry('dec', 'D', 'deg'), LogEntry('ddec', 'D', 'deg'),
        LogEntry('rate', 'D', 'ph/s'), LogEntry('drate', 'D', 'ph/s'),
        LogEntry('flux', 'D', 'ph/cm2/s'), LogEntry('dflux', 'D', 'ph/cm2/s'),
    )
    log.insert(params)

    # helper functions
    def arcmin2deg(theta: float) -> float:
        """Angle unit conversion: [arcmin] to [deg]"""
        return theta / 60
    
    def shift_error(shift: float, dtheta: float) -> float:
        """Computes shift error."""
        l = camera.specs['mask_detector_distance']  # [mm]
        t = np.deg2rad(shift2angle(camera, shift))  # [rad]
        dt = np.deg2rad(arcmin2deg(dtheta))         # [rad]
        return l / np.square(np.cos(t)) * dt        # angle2shift(camera, arcmin2deg(dtheta))
    
    #def eq_coords_errors(
    #    shiftx: float, dshiftx: float,
    #    shifty: float, dshifty: float,
    #) -> tuple[float, float]:
    #    """Computes RA/DEC source errors."""
    #    up_ra, up_dec = shift2equatorial(
    #        sdl, camera, shiftx, shifty + dshifty
    #    )
    #    down_ra, down_dec = shift2equatorial(
    #        sdl, camera, shiftx, shifty - dshifty
    #    )
    #    left_ra, left_dec = shift2equatorial(
    #        sdl, camera, shiftx - dshiftx, shifty
    #    )
    #    right_ra, right_dec = shift2equatorial(
    #        sdl, camera, shiftx + dshiftx, shifty
    #    )
    #    return (
    #        abs(l_ra - r_ra) / 4,
    #        abs(up_dec - down_dec) / 4,
    #    )
    
    def effective_area(shiftx: float, shifty: float) -> float:
        """Computes detector area seen by the source."""

        def process_mask(sx: float, sy: float) -> NDArray:
            """Process mask pattern."""
            mask_maybe_vignetted = apply_vignetting(
                camera, camera.mask, sx, sy,
            ) if vignetting else camera.mask
            
            mask_maybe_vignetted_maybe_psfy = convolve(
                mask_maybe_vignetted, _wfm_psfy_kernel_cached(camera), mode="same",
            ) if psfy else mask_maybe_vignetted
            return mask_maybe_vignetted_maybe_psfy
    
        n, m = camera.shape_sky
        proj = np.zeros(camera.shape_detector)
        components = _rbilinear(
            shiftx, shifty, camera.bins_sky.x, camera.bins_sky.y
        )
        i_min, i_max, j_min, j_max = _detector_footprint(camera)

        for (c_i, c_j), weight in components.items():
            r, c = (n // 2 - c_i), (m // 2 - c_j)
            mask_p = process_mask(camera.bins_sky.x[c_j], camera.bins_sky.y[c_i])
            sg = _shift(mask_p, (r, c))
            proj += sg[i_min:i_max, j_min:j_max] * weight
        proj *= camera.bulk

        return proj.sum() * PX_AREA

    # compute parameters
    dshifts_x, dshifts_y = (
        tuple(shift_error(s, DTHETA_X) for s in shifts_x),
        tuple(shift_error(s, DTHETA_Y) for s in shifts_y),
    )
    log.add_entry_values('dshift_x', dshifts_x)
    log.add_entry_values('dshift_y', dshifts_y)

    dfluences = [np.sqrt(f) for f in fluences]
    log.add_entry_values('dfluence', dfluences)

    y, x = zip(
        *tuple(shift2pos(camera, sx, sy) for sx, sy in zip(shifts_x, shifts_y))
    )
    log.add_entry_values('y', list(y))
    log.add_entry_values('x', list(x))

    thetas_x, thetas_y = map(
        lambda shifts: tuple(shift2angle(camera, s) for s in shifts),
        (shifts_x, shifts_y),
    )
    dthetas_x, dthetas_y = zip(
        *tuple((arcmin2deg(DTHETA_X), arcmin2deg(DTHETA_Y)) for _ in range(len(shifts_x)))
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
    #        eq_coords_errors(sx, dsx, sy, dsy) for sx, dsx, sy, dsy in zip(
    #            shifts_x, dshifts_x, shifts_y, dshifts_y,
    #        )
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
    screening: bool = True,
) -> Log:
    """
    Compares the reconstructed IROS data with the catalogue,
    associating the candidates with the known sources.

    Args:
        log (Log):
            IROS data output from `compute_parameters()`.
        catalogue (CatalogueLoader):
            Catalogue data for the WFM coded-mask camera.
        sdl (DataLoader):
            Data container instance for chosen WFM coded-mask camera.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        screening (bool, optional (default=`True`)):
            If `True`, the repeating sources in the database
            are screened by significance comparison.

    Returns:
        output (Log):
            Log with the updated entries for the input IROS candidates,
            featuring sources IDs and catalogues calibrated fluxes.
    """
    def extend_catalogue(rec: FITS_rec) -> FITS_rec:
        """Adds sources local frame angular coords to catalogue."""
        from astropy.io.fits import Column, BinTableHDU
        shifts_x, shifts_y = zip(
            *tuple(
                equatorial2shift(sdl, camera, ra, dec) for ra, dec in zip(rec['RA'], rec['DEC'])
            )
        )
        ts_x, ts_y = map(
            lambda shifts: tuple(shift2angle(camera, s) for s in shifts),
            (shifts_x, shifts_y),
        )
        thetas_x = Column(name='ANGLE_X', format='D', array=np.array(ts_x))
        thetas_y = Column(name='ANGLE_Y', format='D', array=np.array(ts_y))
        extended = list(rec.columns) + [thetas_x, thetas_y]
        return BinTableHDU.from_columns(extended).data
    
    # set up
    KEYMAP = {
        'cxb_tag': 'gctr_diffuse',
        'NEW_ID': 1,
    }
    database = extend_catalogue(catalogue.DLdata)

    # update Log
    params = (
        LogEntry('ID', '20A', ''), LogEntry('catalogue_flux', 'D', 'ph/cm2/s'),
    )
    log.insert(params)

    def candidate_association(
        thetax: float,
        dthetax: float,
        thetay: float,
        dthetay: float,
        sigma: int | float = 3,
    ) -> tuple[str, float]:
        """Candidate association from catalogue."""

        def closest_source(batch: FITS_rec) -> int:
            """Returns candidate's closer catalogue source index."""
            arg = np.argmin(
                np.square(
                    np.tan(np.deg2rad(batch['ANGLE_X'])) - np.tan(np.deg2rad(thetax))
                )
                +
                np.square(
                    np.tan(np.deg2rad(batch['ANGLE_Y'])) - np.tan(np.deg2rad(thetay))
                )
            )
            return arg
    
        box = (
            (database['ANGLE_X'] > thetax - sigma * dthetax) &
            (database['ANGLE_X'] < thetax + sigma * dthetax) &
            (database['ANGLE_Y'] > thetay - sigma * dthetay) &
            (database['ANGLE_Y'] < thetay + sigma * dthetay) &
            (database['ID'] != KEYMAP['cxb_tag'])
        )
        associated_batch = database[box]

        if not any(associated_batch):
            sourceID = f'lemx-{log.name.lower()}S{KEYMAP['NEW_ID']}'
            flux = -1.0
            KEYMAP['NEW_ID'] += 1
        elif len(associated_batch) == 1:
            sourceID = associated_batch['ID'][0]
            flux = associated_batch['FLUX'][0]
        else:
            arg = closest_source(associated_batch)
            sourceID = associated_batch['ID'][arg]
            flux = associated_batch['FLUX'][arg]

        return sourceID, flux

    print("# Comparing with Catalogue...")
    # initial sources association
    for tx, dtx, ty, dty in zip(
        log.log["angle_x"], log.log["dangle_x"],
        log.log["angle_y"], log.log["dangle_y"],
    ):
        sourceID, flux = candidate_association(tx, dtx, ty, dty)
        log.update(values=(('ID', sourceID), ('catalogue_flux', flux)))
    
    # sources screening based on significance
    if screening:
        df = data_screening(log.to_dataframe(), 'ID', 'snr')
        for col, series in df.items():
            log.replace_entry_values(col, list(series))

    print("# Successful comparison!")
    return log


# end