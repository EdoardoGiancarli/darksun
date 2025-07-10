"""
IROS output data management and computation.
"""

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2angle
from bloodmoon.images import _shift
from bloodmoon.optim import iros

from .types import LogEntry
from .data import Log
from .data import create_log
from .data import DataLoader

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

        - shifts along the (x, y) axis with respective errors* in [mm]
        - fluence and respective error, in [ph]
        - extracted significance

    *The shifts errors are computed as the half-bin size of the interpolated shifts
    values in the `bm.mask.interpmax()` method, inside `bm.optim.optimize()`, where
    the binning grid is oversampled to a factor of (9, 9).
    
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
    # shifts errors along x and y in [mm] (`bm.mask.interpmax()` half-bin)
    dsx = 0.5 * abs(camera.bins_sky.x[0] - camera.bins_sky.x[1]) / 9
    dsy = 0.5 * abs(camera.bins_sky.y[0] - camera.bins_sky.y[1]) / 9

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
) -> Log:
    """
    Computes parameters for IROS reconstructed candidates.

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
    ...


# end