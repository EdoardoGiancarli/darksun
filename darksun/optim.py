"""
Placeholder for IROS algorithm.
"""

from typing import Callable, Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from astropy.coordinates import angular_separation

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.mask import cutout
from bloodmoon.mask import decode
from bloodmoon.mask import snratio
from bloodmoon.mask import variance
from bloodmoon.optim import model_sky
from bloodmoon.optim import optimize

from .data import DataLoader


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
    to a physical size of `(11 * 0.4, 7 * 0.25) mm` for the Wide Field Monitor cameras.
    Inside the method, the kernel size is automatically adjusted to the camera upscaling. 

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
        kernelsize_y (int, optional (default=`7`)):
            Kernel size along the y axis (upscaling 1).
        kernelsize_x (int, optional (default=`11`)):
            Kernel size along the x axis (upscaling 1).
    
    Returns:
        output (NDArray):
            Smoothed detector image. The array is rescaled to have
            the same counts of the original input detector image.
    
    ## Notes:
        - CFR with url: [`bkg_fitting_v3.ipynb`](
        https://github.com/yuri-evangelista/CodedMasks/blob/main/notebooks/bkg_fitting_v3.ipynb
        ).
        - The kernel size may change depending on the selected mask pattern design of the WFM.
    """
    # define median filter kernel size at given camera upscaling
    UPX, UPY = camera.upscale_f
    KERNEL_SIZE = {
        'y': int(kernelsize_y * UPY),
        'x': int(kernelsize_x * UPX),
    }
    
    def apply_filter(axis: int, size: int) -> NDArray:
        """
        Collapses the detector along the specified axis and
        applies a 1D median filter of the given size.
        """
        # collapse detector and bulk
        collapsed_det = detector.sum(axis=axis)
        collapsed_bulk = camera.bulk.sum(axis=axis)
        # bulk zeros are ignored to avoid boundary effects
        bulk_mask = (collapsed_bulk > 0)
        filtered = collapsed_det.copy()
        filtered[bulk_mask] = median_filter(
            collapsed_det[bulk_mask], size=size, mode='nearest',
        )
        return filtered
    
    # apply filter along the two axes independently
    # ! the smoothing is performed by collapsing the
    #   detector along the opposing direction
    smooth_y = apply_filter(axis=1, size=KERNEL_SIZE['y'])
    smooth_x = apply_filter(axis=0, size=KERNEL_SIZE['x'])

    # restore 2D profile through broadcasting (as suggested by np.tile doc)
    # - the smoothed array is masked with the bulk to remove artefacts
    # - the filtered array is rescaled to conserve the original counts
    smoothed = smooth_y[:, np.newaxis] * smooth_x[np.newaxis, :]
    smoothed *= (camera.bulk > 0)
    smoothed *= (detector.sum() / smoothed.sum())

    return smoothed


def verify_camera_orientation(
    sdl_camA: DataLoader,
    sdl_camB: DataLoader,
) -> tuple[DataLoader, DataLoader]:
    """
    Verifies cameras are oriented orthogonally (90° rotation in azimuth).
    This is required for the source position matching algorithm in IROS.
    Then, sorts the data loaders into a tuple so that the second's data
    loader x axis is at +90° from the first one.
    """
    delta_rot_x = angular_separation(
        *map(np.deg2rad, (*sdl_camA.rotations["x"], *sdl_camB.rotations["x"]))
    )
    if (
        not np.isclose(
            angular_separation(
                *map(np.deg2rad, (*sdl_camA.rotations["z"], *sdl_camB.rotations["z"]))
            ),
            0.0,
        )
        or
        not np.isclose(
            np.abs(delta_rot_x),
            np.pi / 2,
        )
    ):
        raise ValueError("Cameras must be rotated by 90° degrees over azimuth.")

    if delta_rot_x > 0:
        sdls = (sdl_camA, sdl_camB)
    else:
        print("SDLs order inverted.")
        sdls = (sdl_camB, sdl_camA)
    
    return sdls


def iros(
    camera: CodedMaskCamera,
    sdl_cam1a: DataLoader,
    sdl_cam1b: DataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
) -> Iterable:
    """Performs Iterative Removal of Sources (IROS) for dual-camera WFM observations.

    This function implements an iterative source detection and removal algorithm for
    the WFM coded mask instrument. For each iteration, it:
    1. Ranks source candidates by SNR and integrated intensity
    2. Matches compatible source positions between orthogonal cameras
    3. Fits source parameters
    4. Removes fitted sources from the sky image
    5. Repeats until no significant sources remain or max iterations reached

    Args:
        camera: CodedMaskCamera instance containing mask/detector geometry and parameters
        sdl_cam1a: SimulationDataLoader for the first WFM  camera
        sdl_cam1b: SimulationDataLoader for the second WFM camera
        max_iterations: Maximum number of source removal iterations to perform
        snr_threshold: Optional float. If provided, iteration stops when maximum
            residual SNR falls below this value. Defaults to 0. (no threshold).
        vignetting: Optional bool. If `True`, the model used for optimization will simulate vignetting.
        psfy: Optional bool. If `True`, the model used for optimization will simulate detector
        position reconstruction effects.

    Yields:
        For each iteration, yields:
            - A tuple of two (x, y, fluence, significance) tuples, one for each camera's
              detected source, where x,y are sky-shift coordinates in mm, fluence is source intensity,
               significance in standard deviations.
            - A tuple of two residual sky images after source removal, one for each camera
            Note: Results are ordered to match sdl_cam1a, sdl_cam1b order

    Raises:
        ValueError: If cameras are not oriented orthogonally (90° rotation in azimuth)
        RuntimeError: If source parameter optimization fails (with detailed error message)

    ## Notes:
        Performance Considerations:
        - Computation scales with mask resolution. Keep upscaling factors low
          (upscale_x * upscale_y ~< 10) for reasonable performance

        Algorithm Details:
        - Requires orthogonal camera views (90° rotation) for source localization
        - Ranks candidates by SNR and integrated intensity within aperture
        - Optimizes source parameters in local windows around candidates
        - When using reconstructed data, accounts for vignetting and PSF effects

    Example:
    >>> for sources, residuals in iros(camera, sdl_cam1a, sdl_cam1b, max_iterations=2):
    >>>     source_1a, source_1b = sources
    >>>     residual_1a, residual_1b = residuals
    >>>     ...
    """
    sdls = verify_camera_orientation(
        sdl_camA=sdl_cam1a,
        sdl_camB=sdl_cam1b,
    )

    def direction_match(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> bool:
        """Determines if source positions from both cameras correspond to the same sky location.
        Compares source positions accounting for the 90° camera rotation. Positions are
        considered matching if they are within one slit width from each other after rotation.
        TODO: not urgent, but in a future we should make this work for arbitrary camera rotations.
        """
        ax, ay = camera.bins_sky.x[a[1]], camera.bins_sky.y[a[0]]
        # we apply -90deg rotation to camera b source
        bx, by = -camera.bins_sky.y[b[0]], camera.bins_sky.x[b[1]]
        min_slit = min(camera.mdl["slit_deltax"], camera.mdl["slit_deltay"])
        return abs(ax - bx) < min_slit and abs(ay - by) < min_slit

    def match(pending: tuple) -> tuple:
        """Cross-check the last entry in pending to match against all other pending directions"""
        pa, pb = pending
        if not pa or not pb:
            return tuple()

        # we are going to call this each time we get a new couple of candidate indices.
        # we avoid evaluating matches for all pairs at all calls, which would result in
        # repeated evaluations of the same pairs (would result in O(n^3) worst case for
        # `find_candidates()`
        *_, latest_a = pa
        for b in pb:
            if direction_match(latest_a, b):
                return latest_a, b

        *_, latest_b = pb
        for a in pa:
            if direction_match(a, latest_b):
                return a, latest_b
        return tuple()

    def init_get_arg(skies: tuple, snrs: tuple, batchsize: int = 1000) -> Callable:
        """This hides a reservoirs-batch mechanism for quickly selecting candidates,
        and initializes the data structures it relies on."""
        # we sort source directions by significance.
        # this is kind of costly because the sky arrays may be very large.
        # sorted directions are moved to a reservoir.
        reservoirs = [np.argsort(sky, axis=None) for sky in skies]

        # integrating source intensities over aperture for all matrix elements is
        # computationally unfeasable. To avoid this, we execute this computation over small batches.
        batches = [np.array([]), np.array([])]

        def slit_intensity():
            """Integrates source intensity over mask's aperture."""
            intensities = ([], [])
            for int_, sky, batch in zip(
                intensities,
                skies,
                batches,
            ):
                for arg in batch:
                    (min_i, max_i, min_j, max_j), _ = cutout(camera, arg)
                    slit = sky[min_i:max_i, min_j:max_j]
                    int_.append(np.sum(slit))
            return intensities

        def fill():
            """Fill the batches with sorted candidates"""
            for i, _ in enumerate(sdls):
                tail, head = reservoirs[i][:-batchsize], reservoirs[i][-batchsize:]
                batches[i] = np.array([np.unravel_index(id, skies[i].shape) for id in head])
                reservoirs[i] = tail

            # integrates over mask element aperture and sum between cameras
            argsort_intensities = np.argsort(np.sum(slit_intensity(), axis=0))

            # sort candidates in present batch by their integrated-combined intensity
            for i, _ in enumerate(sdls):
                batches[i] = batches[i][argsort_intensities]

        def empty():
            """Checks if batches are empty"""
            return all(not len(b) for b in batches)

        def get() -> tuple | None:
            """Think of this as a faucet getting you one decent direction combo at a time."""
            if empty():
                fill()
                if empty():
                    return None

            out = tuple(batch[-1] for batch in batches)
            for i, _ in enumerate(sdls):
                batches[i] = batches[i][:-1]
            return out
        
        return get if max(tuple(snr[*cand] for cand, snr in zip(get(), snrs))) > snr_threshold else lambda: None

    def find_candidates(skies: tuple, snrs: tuple, max_pending=6666) -> tuple:
        """Returns candidate, compatible sources for the two cameras.
        Worst case complexity is O(n^2) but amortized costs are much smaller."""
        get_arg = init_get_arg(skies, snrs)
        pending = ([], [])

        while not (matches := match(pending)):
            args = get_arg()
            if args is None:
                break
            for stack, arg in zip(pending, args):
                stack.append(arg)
                if len(stack) > max_pending:
                    stack.pop(0)
        return matches if matches else tuple()

    def subtract(
        arg: tuple[int, int],
        sky: NDArray,
        snr_map: NDArray,
    ) -> tuple[tuple[float, float, float, float], NDArray]:
        """Runs optimizer and subtract source."""
        try:
            shiftx, shifty, fluence = optimize(
                camera=camera,
                sky=sky,
                arg_sky=arg,
                vignetting=vignetting,
                psfy=psfy,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e

        significance = float(snr_map[*arg])  # candidate significance at peak counts
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

    def compute_snratios(
        skymaps: tuple[NDArray, NDArray],
        varmaps: tuple[NDArray, NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Computes skies SNR."""
        # variance is clipped to improve numerical stability for off-axis sources,
        # which may result in very few counts.
        # TODO: improve on this only sorting matrix elements over a threshold.
        snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skymaps, varmaps))
        return snrs

    detectors = tuple(count(camera, sdl.DLdata)[0] for sdl in sdls)
    variances = tuple(variance(camera, d) for d in detectors)
    skies = tuple(decode(camera, d) for d in detectors)
    for i in range(max_iterations):
        snrs = compute_snratios(skies, variances)
        candidates = find_candidates(skies, snrs)
        if not candidates:
            print("\nNo candidates left...")
            break
        try:
            sources, skies = zip(*(subtract(index, sky, snr) for index, sky, snr in zip(candidates, skies, snrs)))
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield ((sources, skies) if sdls == (sdl_cam1a, sdl_cam1b) else (sources[::-1], skies))


# end