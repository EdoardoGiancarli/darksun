"""
IROS output data management and computation.
"""

from pathlib import Path
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pandas import DataFrame

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2theta
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.images import _shift
from bloodmoon.images import argmax
from bloodmoon.optim import iros

from .types import LogEntry
from .data import Log
from .data import create_log
from .data import DataLoader

__all__ = [""]


def perform_iros(
    camerasID: tuple[str, str],
    camera: CodedMaskCamera,
    sdl_camA: DataLoader,
    sdl_camB: DataLoader,
    max_iterations: int = 25,
    snr_threshold: int | float = 5,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[dict, tuple[NDArray, NDArray]]:
    """
    Runs the IROS (Iterative Removal of Sources) loop and stores the output.

    This function iteratively removes detected sources from the sky and updates 
    a log until either the maximum number of iterations is reached or the 
    SNR threshold is met.

    Args:
        camerasID (tuple[str, str]):
            Cameras A and B of the WFM being processed.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl_camA (DataLoader):
            DataLoader instance for camera A.
        sdl_camB (DataLoader):
            DataLoader instance for camera B.
        max_iterations (int, optional (default=25)):
            Maximum number of iterations for the IROS loop.
        snr_threshold (int | float, optional (default=5)):
            Minimum SNR value required to continue the iterative source removal process.
        vignetting (bool, optional (default=True)):
            If `True`, the model used for optimization will simulate vignetting.
        psfy (bool, optional (default=True)):
            If `True`, the model used for optimization will simulate detector
            position reconstruction effects.

    Returns:
        output (tuple[dict, tuple[NDArray, NDArray]]):
            - log (dict): Database with metadata and results from IROS.
            - residuals (tuple[NDArray, NDArray]): Sky residuals after IROS.
    """
    def store_output(
        log: Log,
        rec_source: tuple[tuple[float]],
        obs_counts: tuple[float],
        sub_counts: tuple[float],
    ) -> None:
        """Stores IROS output inside the log."""
        for idx, camID in enumerate(camerasID):
            sx, sy, f, signif = rec_source[idx]
            entries = (
                ('shiftx', sx), ('shifty', sy), ('fluence', f), ('snr', signif),
                ('obs_counts', obs_counts[idx]), ('sub_counts', sub_counts[idx]),
            )
            log.update(camID, entries)
    
    # generate IROS output log
    params = (
        LogEntry('shiftx', 'D', 'mm'), LogEntry('shifty', 'D', 'mm'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('snr', 'D', ''),
        LogEntry('obs_counts', 'D', 'ph'), LogEntry('sub_counts', 'D', 'ph'),
    )
    log = create_log(
        camA_ID=camerasID[0],
        camB_ID=camerasID[1],
        params=params,
    )

    # init and run IROS loop
    print("# Computing stuff...")
    detectors = tuple(count(camera, sdl.data)[0] for sdl in (sdl_camA, sdl_camB))
    skies = tuple(decode(camera, d) for d in detectors)
    skies_max = [tuple(np.max(sky) for sky in skies)]
    skies = [skies]

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
    for sources, residuals in tqdm(loop):
        skies.append(residuals)
        skies_max.append(tuple(np.max(r) for r in residuals))
        obs_counts = skies_max[0]
        sub_counts = tuple(s.max() - r[*argmax(s)] for s, r in zip(skies[0], skies[1]))
        skies.pop(0); skies_max.pop(0)
        store_output(log, sources, obs_counts, sub_counts)
    
    return log.log, residuals














def IROS_sources_log() -> Log:
    """
    
    """
    raise NotImplementedError



# end