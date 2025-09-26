from typing import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import decode
from bloodmoon.mask import snratio
from bloodmoon.mask import variance
from bloodmoon.optim import optimize, model_sky

from darksun.data import DataLoader


def iros_singleCAM(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
    sky_start: NDArray | None = None,
) -> Iterable:
    """

    """    
    def find_candidate(sky: NDArray, snr: NDArray, batch: int = 1000) -> tuple:
        """Returns candidate."""
        reservoir = np.array([np.unravel_index(id, sky.shape) for id in np.argsort(sky, axis=None)[-batch:]])[::-1]
        for pos in reservoir:
            if (snr[*pos] > snr_threshold):
                return tuple(pos)
        return tuple()

    def subtract(
        candidate: tuple[int, int],
        sky: NDArray,
        snr: NDArray,
    ) -> tuple[tuple[float, float, float, float], NDArray]:
        """Runs optimizer and subtract source."""
        try:
            shiftx, shifty, fluence = optimize(
                camera=camera,
                sky=sky,
                arg_sky=candidate,
                vignetting=vignetting,
                psfy=psfy,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e

        significance = float(snr[*candidate])  # candidate significance at peak counts
        model = model_sky(
            camera=camera,
            shift_x=shiftx,
            shift_y=shifty,
            fluence=fluence,
            vignetting=vignetting,
            psfy=psfy,
        )
        residual = sky - model
        return (shiftx, shifty, fluence, significance), residual
    
    detector = count(camera, sdl.DLdata)[0]
    var_ = variance(camera, detector)
    skymap = decode(camera, detector) if sky_start is None else sky_start
    for i in range(max_iterations):
        snrmap = snratio(skymap, np.clip(var_, a_min=1, a_max=None))
        candidate = find_candidate(skymap, snrmap)
        if not candidate:
            break
        try:
            source, skymap = subtract(candidate, skymap, snrmap)
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield (source, skymap)


def run_IROS(
    camera: CodedMaskCamera,
    sdl: DataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
    sky_start: NDArray | None = None,
) -> tuple[dict, NDArray]:
    """
    """
    entries = (
        'shift_x', 'shift_y', 'fluence', 'snr',
    )
    db = {entry: [] for entry in entries}

    def store_output(values: tuple[float]) -> None:
        """Stores values in database."""
        for par, val in zip(entries, values):
            db[par].append(val)

    loop = iros_singleCAM(
        camera=camera,
        sdl=sdl,
        max_iterations=max_iterations,
        snr_threshold=snr_threshold,
        vignetting=vignetting,
        psfy=psfy,
        sky_start=sky_start,
    )

    print("# Looping around the FOV...")
    for source, residual in tqdm(loop):
        store_output(source)

    return db, residual


# end    