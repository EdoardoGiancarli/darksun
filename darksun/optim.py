"""
Data processing.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import model_shadowgram

from .types import Candidate
from .data import Log

__all__ = [
    "bkg_smoothing", "get_candidates", "retrieve_detector", "detector_smoothing",
]


def _apply_median_filter(
    detector: NDArray,
    bulk: NDArray,
    axis: int,
    size: int,
) -> NDArray:
    """
    Collapses the detector along the specified axis and
    applies a 1D median filter of the given size.
    """
    # collapse detector and bulk
    collapsed_det = detector.sum(axis=axis)
    collapsed_bulk = bulk.sum(axis=axis)
    # bulk zeros are ignored to avoid boundary effects
    filtered = collapsed_det.copy()
    bulk_mask = (collapsed_bulk > 0)
    filtered[bulk_mask] = median_filter(
        collapsed_det[bulk_mask], size=size, mode='nearest',
    )
    return filtered


def bkg_smoothing(
    detector: NDArray,
    camera: CodedMaskCamera,
    *,
    kernelsize_y: int = 11,
    kernelsize_x: int = 7,
) -> NDArray:
    """
    Performs a smoothing of the residual background from the coded-mask camera detector image.

    The smoothing is performed on the (y, x) axes independently. First, the detector image
    is collapsed along a direction, and then a 1D median filter is applied to remove the
    residual high frequencies on the collapsed array. The median filter is applied ignoring
    the detector sensitivity array zeroes to avoid boundary effects.

    The kernel has a default size along `(y, x)` of `(11 x 7)` at upscaling `(1, 1)`, equal
    to a physical size of `(11 * camera.specs.mask_deltay, 7 * camera.specs.mask_deltax) mm`
    for the LEM-X camera modules. Inside the method, the kernel size is automatically
    adjusted to the camera upscaling.

    This smoothing should be applied after removing the brightest sources from the original
    detector image (e.g., by processing the observed sky with the IROS algorithm).
    While the remaining (weaker) sources will be affected by the smoothing, it has been
    tested that their significance is reduced by a factor lower than `10%`, for both on-
    and off-axis sources with SNR between `5` and `50` sigmas.

    Args:
        detector (NDArray):
            Input coded-mask camera detector image.
        camera (CodedMaskCamera):
            Instance with detector geometry info.
        kernelsize_y (int, optional (default=`11`)):
            Kernel size along the y axis (upscaling 1).
        kernelsize_x (int, optional (default=`7`)):
            Kernel size along the x axis (upscaling 1).
    
    Returns:
        output (NDArray):
            Smoothed detector image. The array is rescaled to have
            the same counts of the original input detector image.
    
    ## Notes:
        - CFR with url: [`bkg_fitting_v3.ipynb`](
        https://github.com/yuri-evangelista/CodedMasks/blob/main/notebooks/bkg_fitting_v3.ipynb
        ).
    """
    # define median filter kernel size at given camera upscaling
    KERNEL_SIZE = {
        'y': int(kernelsize_y * camera.upscale_f.y),
        'x': int(kernelsize_x * camera.upscale_f.x),
    }
    # apply filter along the two axes independently
    # ! the smoothing is performed by collapsing the
    #   detector along the opposing direction
    smooth_y = _apply_median_filter(
        detector, camera.bulk, axis=1, size=KERNEL_SIZE['y'],
    )
    smooth_x = _apply_median_filter(
        detector, camera.bulk, axis=0, size=KERNEL_SIZE['x'],
    )
    # restore 2D profile through broadcasting (as suggested by np.tile doc)
    # - the smoothed array is masked with the bulk to remove artefacts
    # - the filtered array is rescaled to conserve the original counts
    smoothed = smooth_y[:, np.newaxis] * smooth_x[np.newaxis, :]
    smoothed *= (camera.bulk > 0)
    smoothed *= (detector.sum() / smoothed.sum())

    return smoothed


def get_candidates(
    log: Log,
    thresh: int | float,
    verbose: bool = True,
) -> tuple[Candidate, ...]:
    """
    Extracts the source candidates from the IROS log with a significance lower than
    the input threshold. This function finds the index of the first log entry where
    'snr' is strictly less than the provided threshold, and then returns all prior
    entries as Candidate objects.
    This assumes that the log data is sorted by 'snr' in descending order.

    Args:
        log (Log):
            The Log object containing the observation data. The `log` is expected
            to contain at least the sources shifts coords in [mm], the fluence [ph],
            the extracted significance and the association ID.
        thresh (int | float):
            The minimum Significance-to-Noise Ratio (SNR) required for an entry to
            be considered a candidate. Entries with `snr >= thresh` are returned.
        verbose (bool, optional (default=`True`)):
            If True, prints diagnostic information about the number of candidates
            found and the SNR values around the threshold index.

    Returns:
        output (tuple[Candidate, ...]):
            List of source candidates with significance higher than threshold.

    Raises:
        IndexError: If all sources have significance below the input threshold.
    """
    snrs = np.array(log.log['snr'])

    if np.all(snrs < thresh):
        raise IndexError("All sources have significance below the input threshold.")

    idx: int = np.argwhere(snrs < thresh)[0, 0]
    candidates: tuple[Candidate, ...] = tuple(
        Candidate(sx, sy, f, signf) for sx, sy, f, signf in zip(
            log.log['shift_x'][:idx],
            log.log['shift_y'][:idx],
            log.log['fluence'][:idx],
            log.log['snr'][:idx],
        )
    )
    if verbose:
        print(
            f"Number of candidates with SNR > {thresh}: {len(candidates)}\n"
            f"Last: {log.log['ID'][idx - 1]} (snr = {log.log['snr'][idx - 1]:.2f})\n"
            f"Following: {log.log['ID'][idx]} (snr = {log.log['snr'][idx]:.2f})\n"
        )
    
    return candidates


def retrieve_detector(
    candidates: tuple[Candidate, ...],
    camera: CodedMaskCamera,
    vignetting: bool,
    psfy: bool,
) -> NDArray:
    """
    Generates the reconstructed detector image by summing the individual, fluence-weighted
    model shadowgrams of all retrieved source candidates. This process effectively
    simulates the detector counts produced by the list of identified sources.

    Args:
        candidates (tuple[Candidate, ...]):
            Tuple of Candidate objects with candidates parameters info.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        vignetting (bool):
            Flag to include vignetting effects in the source modelling.
        psfy (bool):
            Flag to include detector spatial resolution effects in the source modelling.

    Returns:
        output (NDArray):
            A 2D NumPy array representing the final detector image
            based on the parameters of the retrieved candidates.
    """
    detector = np.zeros(camera.shape_detector)
    for (sx, sy, f, _) in candidates:
        sg = model_shadowgram(
            camera=camera,
            shift_x=sx,
            shift_y=sy,
            vignetting=vignetting,
            psfy=psfy,
        )
        detector += (f * sg)
    return detector


def detector_smoothing(
    detector: NDArray,
    candidates: tuple[Candidate, ...],
    camera: CodedMaskCamera,
    vignetting: bool,
    psfy: bool,
    kernelsize_y: int = 11,
    kernelsize_x: int = 7,
) -> NDArray:
    """
    Process the observed detector image by applying a median smoothing of the background.

    Args:
        detector (NDArray):
            Input coded-mask camera detector image.
        candidates (tuple[Candidate, ...]):
            Tuple of Candidate objects with candidates parameters info.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        vignetting (bool):
            Flag to include vignetting effects in the source modelling.
        psfy (bool):
            Flag to include detector spatial resolution effects in the source modelling.
        kernelsize_y (int, optional (default=`11`)):
            Kernel size along the y axis (upscaling 1).
        kernelsize_x (int, optional (default=`7`)):
            Kernel size along the x axis (upscaling 1).

    Returns:
        output (NDArray):
            A 2D NumPy array representing the final detector image
            based on the parameters of the retrieved candidates.
    """
    # get residual detector image
    retrieved = retrieve_detector(
        candidates=candidates,
        camera=camera,
        vignetting=vignetting,
        psfy=psfy,
    )
    res_detector = detector - retrieved
    # perform smoothing on residual detector image
    res_smoothed = bkg_smoothing(
        detector=res_detector,
        camera=camera,
        kernelsize_y=kernelsize_y,
        kernelsize_x=kernelsize_x,
    )
    # get smoothed detector image
    smoothed = detector - res_smoothed
    return smoothed


# end