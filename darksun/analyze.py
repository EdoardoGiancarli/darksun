"""
IROS output data management and computation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve
from tqdm import tqdm

from bloodmoon.mask import _detector_footprint
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2angle
from bloodmoon.images import _shift
from bloodmoon.images import _rbilinear
from bloodmoon.optim import _wfm_psfy_kernel_cached
from bloodmoon.optim import apply_vignetting
from bloodmoon.optim import iros

from .types import LogEntry
from .data import DataLoader
from .data import Log
from .data import create_log

__all__ = [
    "run_IROS", "compute_parameters", "catalogue_comparison"
]


def run_IROS(
    camera: CodedMaskCamera,
    *,
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

        - shifts along the (x, y) axes with respective errors* in [mm]
        - fluence and respective error, in [ph]
        - extracted significance

    *The shifts errors are assumed to be the half-bin size of the binning grid.
    
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
            - logs (tuple[Log, Log]): WFM databases with metadata and results from IROS.
            - residuals (tuple[NDArray, NDArray]): Sky residuals for the WFM after IROS.
    """
    # shifts errors along x and y in [mm]
    dsx = 0.5 * abs(camera.bins_sky.x[0] - camera.bins_sky.x[1])
    dsy = 0.5 * abs(camera.bins_sky.y[0] - camera.bins_sky.y[1])

    def callback(output: tuple[float]) -> tuple[float]:
        """Manage IROS candidate output parameters."""
        sx, sy, f, signf = output
        df = np.sqrt(f)
        return sx, dsx, sy, dsy, f, df, signf
        
    # generate IROS output log
    params = (
        LogEntry('shiftx', 'D', 'mm'), LogEntry('dshiftx', 'D', 'mm'),
        LogEntry('shifty', 'D', 'mm'), LogEntry('dshifty', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('snr', 'D', ''),
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
            values=tuple((p.entry, val) for p, val in zip(params, callback(parA)))
        )
        log_camB.update(
            values=tuple((p.entry, val) for p, val in zip(params, callback(parB)))
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
    The input Log is updated with the following parameters:

        - candidates image pixel indexes
        - WFM camera local frame (x, y) angular coordinates and errors, in [deg]
        - candidate equatorial coordinates (RA, Dec) and errors, in [deg]
        - candidate photons rate and error, in [ph/s]
        - candidate photons flux and error, in [ph/cm2/s]

    Args:
        log (Log):
            Log instance with IROS data output from `run_IROS()`.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl (DataLoader):
            Data container instance for chosen WFM coded-mask camera.

    Returns:
        output (Log):
            Log instance with computed parameters for each candidate.
    """
    # retrieve observation data (px area [cm^2], camera exposure [s])
    ups = np.prod(camera.upscale_f)
    px_area = (
        1e-2 * camera.specs["mask_deltax"] * camera.specs["mask_deltay"] / ups
    )
    exposure = sdl.header["EXPOSURE"]

    # insert new entries
    params = (
        LogEntry('y', 'J', 'px'), LogEntry('x', 'J', 'px'),
        LogEntry('anglex', 'D', 'deg'), LogEntry('danglex', 'D', 'deg'),
        LogEntry('angley', 'D', 'deg'), LogEntry('dangley', 'D', 'deg'),
        LogEntry('ra', 'D', 'deg'), LogEntry('dra', 'D', 'deg'),
        LogEntry('dec', 'D', 'deg'), LogEntry('ddec', 'D', 'deg'),
        LogEntry('rate', 'D', 'ph/s'), LogEntry('drate', 'D', 'ph/s'),
        LogEntry('flux', 'D', 'ph/cm2/s'), LogEntry('dflux', 'D', 'ph/cm2/s'),
    )
    log.insert(params)

    shifts_x, dshifts_x = log.log['shiftx'], log.log['dshiftx']
    shifts_y, dshifts_y = log.log['shifty'], log.log['dshifty']
    fluences, dfluences = log.log['fluence'], log.log['dfluence']

    def angle_error(shift: float, dshift: float) -> float:
        """Computes camera angular coordinate error."""
        top = shift2angle(camera, shift + dshift)
        bottom = shift2angle(camera, shift - dshift)
        return abs(top - bottom) / 4
    
    def eq_coords_errors(
        shiftx: float, dshiftx: float,
        shifty: float, dshifty: float,
        sdl: DataLoader,
    ) -> tuple[float, float]:
        """Computes RA/DEC source errors."""
        r_ra, up_dec = shift2equatorial(
            sdl, camera, shiftx + dshiftx, shifty + dshifty
        )
        l_ra, down_dec = shift2equatorial(
            sdl, camera, shiftx - dshiftx, shifty - dshifty
        )
        return (
            abs(l_ra - r_ra) / 4,
            abs(up_dec - down_dec) / 4,
        )
    
    def effective_area(sx: float, sy: float) -> float:
        """Computes detector area seen by the source."""

        def process_mask(i: float, j: float) -> NDArray:
            """Process mask pattern."""
            mask_maybe_vignetted = apply_vignetting(
                camera, camera.mask, i, j,
            ) if vignetting else camera.mask
            
            mask_maybe_vignetted_maybe_psfy = convolve(
                mask_maybe_vignetted, _wfm_psfy_kernel_cached(camera), mode="same",
            ) if psfy else mask_maybe_vignetted
            return mask_maybe_vignetted_maybe_psfy
    
        n, m = camera.shape_sky
        proj = np.zeros(camera.shape_detector)
        components = _rbilinear(sx, sy, camera.bins_sky.x, camera.bins_sky.y)
        i_min, i_max, j_min, j_max = _detector_footprint(camera)

        for (c_i, c_j), weight in components.items():
            r, c = (n // 2 - c_i), (m // 2 - c_j)
            mask_p = process_mask(camera.bins_sky.x[c_j], camera.bins_sky.y[c_i])
            sg = _shift(mask_p, (r, c))
            proj += sg[i_min:i_max, j_min:j_max] * weight
        proj *= camera.bulk

        return proj.sum() * px_area

    # compute parameters
    px_idxs = tuple(
        shift2pos(camera, sx, sy) for sx, sy in zip(shifts_x, shifts_y)
    )
    log.add_entry_values('y', [idx[0] for idx in px_idxs])
    log.add_entry_values('x', [idx[1] for idx in px_idxs])

    thetas_x, thetas_y = map(
        lambda shifts: tuple(shift2angle(camera, s) for s in shifts),
        (shifts_x, shifts_y),
    )
    dthetas_x, dthetas_y = map(
        lambda shifts, dshifts: tuple(angle_error(s, ds) for s, ds in zip(shifts, dshifts)),
        (shifts_x, dshifts_x),
        (shifts_y, dshifts_y),
    )
    log.add_entry_values('anglex', list(thetas_x))
    log.add_entry_values('angley', list(thetas_y))
    log.add_entry_values('danglex', list(dthetas_x))
    log.add_entry_values('dangley', list(dthetas_y))

    coords = tuple(
        shift2equatorial(sdl, camera, sx, sy) for sx, sy in zip(shifts_x, shifts_y)
    )
    dcoords = tuple(
        eq_coords_errors(sx, dsx, sy, dsy, sdl) for sx, dsx, sy, dsy in zip(
            shifts_x, dshifts_x, shifts_y, dshifts_y,
        )
    )
    log.add_entry_values('ra', [c.ra for c in coords])
    log.add_entry_values('dec', [c.dec for c in coords])
    log.add_entry_values('dra', [deq[0] for deq in dcoords])
    log.add_entry_values('ddec', [deq[1] for deq in dcoords])

    rates = [f / exposure for f in fluences]
    drates = [df / exposure for df in dfluences]
    log.add_entry_values('rate', rates)
    log.add_entry_values('drate', drates)

    fluxes = [
        f / (effective_area(sx, sy) * exposure) for f, sx, sy in zip(fluences, shifts_x, shifts_y)
    ]
    dfluxes = [
        df / (effective_area(sx, sy) * exposure) for df, sx, sy in zip(dfluences, shifts_x, shifts_y)
    ]
    log.add_entry_values('flux', fluxes)
    log.add_entry_values('dflux', dfluxes)

    return log




# end