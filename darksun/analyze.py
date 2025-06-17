"""
IROS output data management and computation.
"""

from collections.abc import Sequence
from pathlib import Path
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pandas import DataFrame

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.coords import shift2equatorial
from bloodmoon.coords import shift2pos
from bloodmoon.coords import shift2angle
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


def compute_params(
    iros_output: dict,
    camera: CodedMaskCamera,
    sdl_camA: DataLoader,
    sdl_camB: DataLoader,
) -> dict:
    """
    Computes parameters for IROS reconstructed sources.

    Args:
        iros_output (dict):
            IROS data output from `perform_iros()`.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        sdl_camA (DataLoader):
            Data container instance for camera A.
        sdl_camB (DataLoader):
            Data container instance for camera B.

    Returns:
        output (dict):
            Database with computed parameters for each reconstructed source.
    """
    # generate params output log
    params = (
        LogEntry('y', 'J', 'px'), LogEntry('x', 'J', 'px'),
        LogEntry('shiftx', 'D', 'mm'), LogEntry('dshiftx', 'D', 'mm'),
        LogEntry('shifty', 'D', 'mm'), LogEntry('dshifty', 'D', 'mm'),
        LogEntry('anglex', 'D', 'deg'), LogEntry('danglex', 'D', 'deg'),
        LogEntry('angley', 'D', 'deg'), LogEntry('dangley', 'D', 'deg'),
        LogEntry('ra', 'D', 'deg'), LogEntry('dra', 'D', 'deg'),
        LogEntry('dec', 'D', 'deg'), LogEntry('ddec', 'D', 'deg'),
        LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
        LogEntry('rate', 'D', 'ph/s'), LogEntry('drate', 'D', 'ph/s'),
        LogEntry('flux', 'D', 'ph/cm2/s'), LogEntry('dflux', 'D', 'ph/cm2/s'),
        LogEntry('obs_fluence', 'D', 'ph'), LogEntry('sub_fluence', 'D', 'ph'),
        LogEntry('simul_ph', 'D', 'ph'), LogEntry('snr', 'D', ''),
    )
    camA, camB = iros_output.keys()
    log = create_log(
        camA_ID=camA,
        camB_ID=camB,
        params=params,
    )

    # retrieve observation data
    sdls = (sdl_camA, sdl_camB)
    ups = np.prod(camera.upscale_f)                                   # px area [cm^2]
    px_area = (
        1e-2 * camera.specs["mask_deltax"] * camera.specs["mask_deltay"] / ups
    )
    exposure = [data.header["EXPOSURE"] for data in sdls]             # camera exposure [s]
    dshiftx = abs(camera.bins_sky.x[0] - camera.bins_sky.x[1]) / 2    # shift error along x (half-bin) [mm]
    dshifty = abs(camera.bins_sky.y[0] - camera.bins_sky.y[1]) / 2    # shift error along y (half-bin) [mm]

    def update_log(idx: int, cameraID: str) -> None:
        """Update the sources log with the computed parameters."""

        def _source_effective_area(sx: float, sy: float) -> float:
            """Computes detector area seen by the source."""          
            scalingy, scalingx = tuple(
                d / s for d, s in zip(camera.shape_detector, camera.shape_sky)
            )
            shifty_px = int(sy * scalingy / camera.specs["mask_deltay"])
            shiftx_px = int(sx * scalingx / camera.specs["mask_deltax"])
            # shift is opposed wrt source pos
            proj = _shift(camera.bulk, (-shifty_px, -shiftx_px)) * camera.bulk
            return px_area * proj.sum()

        def _error_angle(s: float, ds: float) -> float:
            """Computes camera angular coordinate error."""
            bottom = shift2angle(camera, s - ds)
            top = shift2angle(camera, s + ds)
            return abs(top - bottom) / 4

        def _eq_coords_errors(
            ss: float,
            dss: float,
            sdl: DataLoader,
        ) -> tuple[float, float]:
            """Computes RA/DEC source errors."""
            sx, sy = ss
            dsx, dsy = dss
            r_ra, up_dec = shift2equatorial(sdl, camera, sx + dsx, sy + dsy)
            l_ra, down_dec = shift2equatorial(sdl, camera, sx - dsx, sy - dsy)
            return (
                abs(l_ra - r_ra) / 4,
                abs(up_dec - down_dec) / 4,
            )

        def _simulated_photons(
            coords: tuple[float, float],
            dcoords: tuple[float, float],
            sdl: DataLoader,
            sigma: float = 3.0,
        ) -> int:
            """Retrieve source collected photons."""
            ra, dec = coords
            dra, ddec = dcoords
            l, r = ra - sigma * dra, ra + sigma * dra
            b, t = dec - sigma * ddec, dec + sigma * ddec
            collected_phs = (
                (sdl.DLdata["RA"] > l) & (sdl.DLdata["RA"] < r) &
                (sdl.DLdata["DEC"] > b) & (sdl.DLdata["DEC"] < t)
            )
            return len(sdl.DLdata[collected_phs])

        def get_params(
            shiftx: float,
            shifty: float,
            counts: float,
            snr: float,
            obs_counts: float,
            sub_counts: float,
        ) -> Sequence[tuple[str, int | float]]:
            """Computes sources parameters from specified camera output."""
            y, x = shift2pos(camera, shiftx, shifty)                 # px indexes from optimized shifts
            anglex = shift2angle(camera, shiftx)                     # instrument angle coords along x [deg]
            danglex = _error_angle(shiftx, dshiftx)                  # angle along x error [deg]
            angley = shift2angle(camera, shifty)                     # instrument angle coords along y [deg]
            dangley = _error_angle(shifty, dshifty)                  # angle along y error [deg]
            ra, dec = shift2equatorial(
                sdls[idx], camera, shiftx, shifty,
            )                                                        # RA, DEC [deg]
            dra, ddec = _eq_coords_errors(
                (shiftx, shifty), (dshiftx, dshifty), sdls[idx],
            )                                                        # RA, DEC errors [deg]
            dcounts = np.sqrt(counts)                                # fluence error (Poissonian) [ph]
            rate = counts / exposure[idx]                            # rate [ph/s]
            drate = dcounts / exposure[idx]                          # rate error (Poissonian) [ph/s]
            _eff_area = _source_effective_area(shiftx, shifty)
            flux = rate / _eff_area                                  # flux [ph/cm2/s]
            dflux = drate / _eff_area                                # flux error [ph/cm2/s]
            simph = _simulated_photons(
                (ra, dec), (dra, ddec), sdls[idx],
            )                                                        # source collected photons [ph]

            entries = (
                ('y', y), ('x', x),
                ('shiftx', shiftx), ('dshiftx', dshiftx),
                ('shifty', shifty), ('dshifty', dshifty),
                ('anglex', anglex), ('danglex', danglex),
                ('angley', angley), ('dangley', dangley),
                ('ra', ra), ('dra', dra),
                ('dec', dec), ('ddec', ddec),
                ('fluence', counts), ('dfluence', dcounts),
                ('rate', rate), ('drate', drate),
                ('flux', flux), ('dflux', dflux),
                ('obs_fluence', obs_counts), ('sub_fluence', sub_counts),
                ('simul_ph', simph), ('snr', snr),
            )
            return entries

        for sx, sy, f, snr, oc, sc in zip(*iros_output[cameraID].values()):
            entries = get_params(sx, sy, f, snr, oc, sc)
            log.update(cameraID, entries)


    for idx, camID in enumerate((camA, camB)):
        update_log(idx, camID)
    
    return log.log



# end