"""
Module for images processing.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import model_sky

__all__ = [
    "upscale", "downscale", "crop",
    "make_sky", "WFM_composition"
]


def upscale(
    data: NDArray,
    upscale_y: int = 1,
    upscale_x: int = 1,
) -> NDArray:
    """
    Oversamples a 2D array by repeating elements along each axis and
    by interpolating array values.

    Args:
        data (NDArray):
            Input 2D array.
        upscale_y (int, optional (default=`1`)):
            Upscaling factor over the y direction.
        upscale_x (int, optional (default=`1`)):
            Upscaling factor over the x direction.

    Returns:
        output (NDArray): Oversampled array.

    Raises:
        ValueError: If upscale factors are not positive integers.
    
    ## Notes:
        - The array total sum is conserved through linear interpolation.
        - For N-dim arrays, consider using Astropy's `block_replicate()`.
    """
    def enlarge(m: NDArray, upscale_f: tuple[int, int]) -> NDArray:
        """
        Oversamples a 2D array by repeating elements along the axes.
        """    
        for i, f in enumerate(upscale_f):
            m = np.repeat(m, f, axis=i)
        return m

    if not (
        (isinstance(upscale_y, int) and upscale_y > 0) and
        (isinstance(upscale_x, int) and upscale_x > 0)
    ):
        raise ValueError("Upscaling factors must be positive integers.")
    
    upscaling = (upscale_y, upscale_x)
    return enlarge(data, upscaling) / np.prod(upscaling)


def downscale(
    data: NDArray,
    downscale_y: int = 1,
    downscale_x: int = 1,
) -> NDArray:
    """
    Downscales a 2D array by dividing the input array in blocks
    and adding over them to interpolate array values.

    Args:
        data (NDArray):
            Input 2D array.
        downscale_y (int, optional (default=`1`)):
            Downscaling factor over the y direction.
        downscale_x (int, optional (default=`1`)):
            Downscaling factor over the x direction.

    Returns:
        output (NDArray): Downsampled array.

    Raises:
        ValueError: If downscale factors are not positive integers.
    
    ## Notes:
        - The downsampling is performed through blocks subdivision, which
          represent the elements of the downsampled array. Each block is
          reduced by adding its elements for linear interpolation.
        - The total sum of the array is conserved.
        - For N-dim arrays, consider using Astropy's `block_reduce()`.
    """
    def decrease(m: NDArray, downscale_f: NDArray) -> NDArray:
        """Downsamples a 2D array"""

        def handle_axis(a: NDArray, idx: int) -> NDArray:
            """Redistributes cutted values in the block-adjusted axis."""
            return a[:idx] + a[idx:].sum(axis=0) / idx

        def handle_shape(data: NDArray, factors: NDArray) -> NDArray:
            """Adjusts array for blocks subdivision by cutting extra-rows/columns."""            
            adj_shape = (np.array(data.shape) // factors) * factors
            for ax in range(data.ndim):
                if data.shape[ax] != adj_shape[ax]:
                    data = data.swapaxes(0, ax)
                    data = handle_axis(data, adj_shape[ax])
                    data = data.swapaxes(0, ax)
            return data

        def to_blocks(data: NDArray, factors: NDArray) -> NDArray:
            """Reshapes input array into blocks."""
            assert not np.any(np.mod(data.shape, factors) != 0)
            nblocks = np.array(data.shape) // factors
            reshaping = tuple(dim for dims in zip(nblocks, factors) for dim in dims)
            return data.reshape(reshaping).transpose((0, 2, 1, 3))
        
        m = handle_shape(m, downscale_f)
        m = to_blocks(m, downscale_f)
        return m.sum(axis=(2, 3))
    
    if not (
        (isinstance(downscale_y, int) and downscale_y > 0) and
        (isinstance(downscale_x, int) and downscale_x > 0)
    ):
        raise ValueError("Downscaling factors must be positive integers.")
    
    downscaling = np.array((downscale_y, downscale_x))
    return decrease(data, downscaling)


def crop(
    image: NDArray,
    pos: tuple[int, int],
    crp: tuple[int, int],
    strict: bool = True,
) -> NDArray:
    """
    Crops 2D array at given position and with given cropping.

    Args:
        image (NDArray):
            2D array to crop.
        pos (tuple[int, int]):
            Center position for cropping.
        crp (tuple[int, int]):
            Size of the cropping along (y, x).
        strict (bool, optional (default=`True`)):
            If `False` allows for the cropping to be adapted
            wrt the array edges when they are exceeded.
    
    Returns:
        output (NDArray):
            Cropped 2D array. The cut is performed by centering the
            cropped array, so that the final shape is `2 * crp + 1`
            along the two axes.
    
    Raises:
        ValueError: If `crp` is not a positive int tuple.
        IndexError: If `crp` wrt indexes exceeds 2D array edges
                    (only if `strict` is `True`).
    
    ## Notes:
        - Negative indexes for `pos` are allowed.
    """
    n, m = image.shape
    y, x = pos
    cy, cx = crp
    boundary_x = (
        ((0 <= x - cx) and (x + cx < m - 1)) or ((cx - x <= m - 1) and (x + cx < 0))
    )
    boundary_y = (
        ((0 <= y - cy) and (y + cy < n - 1)) or ((cy - y <= n - 1) and (y + cy < 0))
    )

    if (cy <= 0) or (cx <= 0):
        raise ValueError("Cropping must be a tuple of positive integers.")
    if not (boundary_x and boundary_y):
        if not strict:
            # the crop extends up to the 2nd row/col from top/bottom/left/right
            if not boundary_x:
                cx = min(x - 2, m - x - 3) if x > 0 else min(x + m + 2, -x - 2)
            if not boundary_y:
                cy = min(y - 2, n - y - 3) if y > 0 else min(y + n + 2, -y - 2)
            print(f"Cropping {crp} at pos {pos} exceeds array edges, new cropping: {cy, cx}.")
        else:
            raise IndexError(f"Cropping {crp} at pos {pos} exceeds array edges.")
    
    return image[y - cy : y + cy + 1, x - cx : x + cx + 1]


def make_sky(
    data: dict,
    camera: CodedMaskCamera,
    *,
    vignetting: bool = True,
    psfy: bool = True,
    background: NDArray = None,
) -> NDArray:
    """
    Generates a skymap with the info retrieved by IROS.

    Args:
        data (dict):
            Database with parameters computed from IROS.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        vignetting (bool, optional (default=`True`)):
            Simulates vignetting effects.
        psfy (bool, optional (default=`True`)):
            Simulates detector reconstruction effects.
        background (NDArray, optional (default=`None`)):
            Background for the reconstructed sky.
    
    Returns:
        sky (NDArray):
            Resulting sky from IROS image reconstruction.
    
    Raises:
        ValueError: If `background` has an invalid shape.
    
    ## Notes:
        - The input database must contain at least the sources (i) sky
          coords shifts in [mm] wrt the camera optical axis; (ii) their
          fluences [ph] and (iii) the px indexes.
        - The background is optional (e.g. could be a Poissonian distr.
          of photons decoded from the detector or the IROS residuals).
    """
    def valid_BG(bkg: NDArray) -> bool:
        """Checks background shape."""
        if not (bkg.shape == camera.shape_sky):
            raise ValueError(f"Background must have same sky shape {camera.shape_sky}.")
        return True

    def make_source(
        shiftx: float,
        shifty: float,
        fluence: float,
        pos: tuple[int, int],
        crp: tuple[int, int],
    ) -> NDArray:
        """Generates a source shadowgram and returns a crop of the source."""
        model = model_sky(camera, shiftx, shifty, fluence, vignetting, psfy)
        return crop(model, pos, crp, strict=False)

    if background is None:
        sky = np.zeros(camera.shape_sky, dtype=np.int32)
    elif valid_BG(background):
        sky = np.int32(background)
    
    upx, upy = camera.upscale_f
    cropx, cropy = (
        int(camera.specs["slit_deltax"] * upx / camera.specs["mask_deltax"] + 5),
        int(camera.specs["slit_deltay"] * upy / camera.specs["mask_deltay"] + 5),
    )
    
    for shiftx, shifty, fluence, x, y in zip(
        data["shift_x"],
        data["shift_y"],
        data["fluence"],
        data["x"],
        data["y"],
    ):
        modeled = make_source(
            shiftx=shiftx,
            shifty=shifty,
            fluence=fluence,
            pos=(y, x),
            crp=(cropy, cropx)
        )
        p, q = modeled.shape
        sky[y - p // 2 : y + p // 2 + 1, x - q // 2 : x + q // 2 + 1] += np.int32(modeled)
    
    return sky


def WFM_composition(
    skyA_path: str | Path,
    skyB_path: str | Path,
) -> tuple[NDArray, NDArray, WCS]:
    """
    Performs the composition of the WFM cameras skies and significances,
    including the reprojection of the World Coordinates System for RA/Dec.

    Specifically, it:
        a. Opens the skies FITS file
        b. Finds the optimal WCS fit and sky shape for the composition
        c. Reprojects and sums the two skies making the composition
        d. Reprojects the two SNRs and takes the max

    Args:
        skyA_path (str, Path):
            File path for the camera A sky.
        skyB_path (str, Path):
            File path for the camera B sky.
    
    Returns:
        output (tuple[NDArray, NDArray, WCS]):
            - sky (NDArray):
                WFM cameras sky composition.
            - snr (NDArray):
                WFM composed sky significance computed by taking
                the max of the two cameras individual sky SNR.
            - wcs (WCS):
                Output reprojected WCS fit.

    ## Notes:
        - If the WCS fit keys are not present in the camera skies headers,
          a TypeError will be raised from `find_optimal_celestial_wcs()`:
        >>> TypeError: "WCS does not have celestial components."
    
    TODO:
        - optimize/improve array composition
    """
    with fits.open(skyA_path) as hduA, fits.open(skyB_path) as hduB:
        skies = (hduA[1], hduB[1])
        snrs = (hduA[2], hduB[2])
    
        print("# Composing WFM skies...")
        wcs_out, shape_out = find_optimal_celestial_wcs(input_data=skies)
        sky_comp, _ = reproject_and_coadd(
            input_data=skies,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="sum",
        )
        snr_comp, _ = reproject_and_coadd(
            input_data=snrs,
            output_projection=wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function="max",
        )
    
    print("# WFM composition completed!")
    return sky_comp, snr_comp, wcs_out


# end