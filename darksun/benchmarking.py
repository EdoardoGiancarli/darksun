"""
IROS sources reconstruction analysis and benchmarking.
"""

from typing import Any
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from astropy.io.fits.fitsrec import FITS_rec

from bloodmoon.types import CoordEquatorial
from bloodmoon.coords import equatorial2shift, shift2angle
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import count
from bloodmoon.optim import model_sky

from .data import Log, CatalogueLoader, DataLoader
from .filtering import select_source_photons
from .images import crop, upscale
from .show import map4image, image_plot, slices_plot
from .utils import savefile_to

__all__ = [
    "dict2df", "config_distr_limits", "pixels_angular_resolution",
    "psf_extension", "crop_source_psf", "source_catalogue_data",
    "source_angular_coords", "source_fluence", "extract_sources_info",
    "data_DF", "reconstructed_sources_profiles", "reconstruction_sources_heatmaps",
]


def dict2df(data: dict[str, dict[str, Any]]) -> DataFrame:
    """
    Flattens a two-level nested dictionary into a pandas DataFrame with MultiIndex columns.

    This utility function takes a dictionary where the first key typically represents 
    a category (e.g., 'camera ID') and the second key represents a parameter (e.g., 'shift_x').
    The function flattens this structure by creating a new dictionary where keys are
    two-element tuples (category, parameter), and the values are the data series.
    This resulting dictionary is then directly passed to the pandas DataFrame constructor,
    which automatically creates a DataFrame with a MultiIndex column structure.

    Args:
        data (dict[str, dict[str, Any]]):
            A two-level nested dictionary. The outer keys are expected to be strings
            representing categories (Level 0 of the column index), and the inner keys
            are strings representing parameters (Level 1 of the column index).
            The inner dictionary values must be array-like structures (lists, NumPy arrays, etc.)
            of the same length, which will form the rows of the DataFrame.

    Returns:
        output (DataFrame):
            A pandas DataFrame with a MultiIndex column structure. The column index levels
            are determined by the outer and inner keys of the input dictionary.
    """
    dmap = {
        (cam, param): values
        for cam, cam_data in data.items() 
        for param, values in cam_data.items()
    }
    return DataFrame(dmap)


def config_distr_limits(
    bins: NDArray,
    start: int | float | None,
    stop: int | float | None,
) -> tuple[slice, slice]:
    """
    Configures histogram distribution values to show.

    Args:
        bins (NDArray): Bins array values.
        start (int | float): Inf values limit.
        stop (int | float): Sup values limit.
    
    Returns:
        output (tuple[slice, slice]):
            - slice obj for bins values
            - slice obj for histogram values
    """
    i_min = (
        bisect_right(bins, start) - 1 if start is not None else None
    )
    i_max = (
        bisect_left(bins, stop) if stop is not None else None
    )
    slice_bins = slice(i_min, i_max)
    slice_hist = slice(i_min, i_max - 1 if i_max is not None else None)
    return slice_bins, slice_hist


def pixels_angular_resolution(
    camera: CodedMaskCamera,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Calculates the angular size, in [arcmin], of a single upscaled detector pixel.

    This function determines the effective physical size of a detector pixel after upscaling,
    then uses the mask-detector distance and simple trigonometry (arctan) to convert that
    physical size into an angular resolution on the sky plane. The result is converted
    from radians to arcminutes for convenience.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        verbose (bool, optional (default=`True`)):
            If True, prints diagnostic information including the upscaling factors and
            the calculated angular resolutions for both directions.

    Returns:
        output (tuple[float, float]):
            Camera images pixel angular resolution in [arcmin]
            along the fine and coarse directions.
    """
    pxs = (
        camera.specs.mask_deltax / camera.upscale_f.x,
        camera.specs.mask_deltay / camera.upscale_f.y,
    )
    dtheta_x, dtheta_y = map(
        lambda x: np.rad2deg(np.arctan(x / camera.specs.mask_detector_distance)) * 60,
        pxs,
    )
    if verbose:
        print(
            f"\n"
            f"Pixel angular resolution at upscaling (x, y): ({camera.upscale_f.x}, {camera.upscale_f.y})\n"
            f"  - fine direction: {dtheta_x:.4f} arcmin\n"
            f"  - coarse direction: {dtheta_y:.4f} arcmin"
            f"\n"
        )
    return dtheta_x, dtheta_y


def psf_extension(camera: CodedMaskCamera) -> tuple[int, int]:
    """
    Computes the coded-mask camera source PSF extension in pixels.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
    
    Returns:
        output (tuple[int, int]):
            Camera PSF extension in pixels along (fine, coarse) directions.
    """
    slit_fine, slit_coarse = (
        int(camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax),
        int(camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay),
    )
    return slit_fine, slit_coarse


def crop_source_psf(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
    offset_fine: int = 1,
    offset_coarse: int = 1,
) -> tuple[slice, slice]:
    """
    Crops a source PSF with the given offset along fine and coarse directions.

    Args:
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        pos (tuple[int, int]):
            2D array pixel position to centre the cropping.
        offset_fine (int, optional default=`1`):
            Offset in pixels for the camera fine direction.
        offset_coarse (int, optional default=`1`):
            Offset in pixels for the camera coarse direction.
    
    Returns:
        output (tuple[slice, slice]):
            - slice obj for 2D array rows
            - slice obj for 2D array cols
    """
    crop_fine, crop_coarse = psf_extension(camera)
    crop_fine_, crop_coarse_ = (
        crop_fine + offset_fine,
        crop_coarse + offset_coarse,
    )
    i, j = pos
    slice_rows, slice_cols = (
        slice(i - crop_coarse_, i + crop_coarse_ + 1),
        slice(j - crop_fine_, j + crop_fine_ + 1),
    )
    return slice_rows, slice_cols


def source_catalogue_data(
    sourceID: str,
    data: FITS_rec,
) -> FITS_rec:
    """
    Extracts the data record corresponding to a specific
    source from a catalogue table.

    Args:
        sourceID (str):
            The source ID (e.g., 'scox1', 'crab').
        data (FITS_rec):
            Record with the info on the sources.

    Returns:
        output (FITS_rec):
            Single row array with the source info.
    """
    return np.unique(data[(data['ID'] == sourceID)])[0]


def source_angular_coords(
    sourceID: str,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
) -> tuple[float, float]:
    """
    Retrieves the angular coordinates of a source in the
    instrument's local frame from the specified catalogue.

    Args:
        sourceID (str):
            The source ID (e.g., 'scox1', 'crab').
        catalogue (CatalogueLoader):
            The catalogue object with sources metadata.
        sdl (DataLoader):
            DataLoader instance with pointings information.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.

    Returns:
        output (tuple[float, float]):
            Camera local-frame angular coordinate of
            the source along the (x, y) axes in [deg].
    """
    data = source_catalogue_data(sourceID, catalogue.DLdata)
    shifts = equatorial2shift(sdl, camera, data['RA'], data['DEC'])
    thetax, thetay = map(lambda x: shift2angle(camera, x), shifts)
    return thetax, thetay


def source_fluence(
    sourceID: str,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
    verbose: bool = True,
) -> float:
    """
    Calculates the total true source fluence (total photon counts) detected
    by the instrument's *active* detector area for a specific source.

    This function accounts for instrument effects by discarding photons that
    fall onto the detector's dead zones (non-sensitive areas).

    Args:
        sourceID (str):
            The source ID (e.g., 'scox1', 'crab').
        catalogue (CatalogueLoader):
            The catalogue object with sources metadata.
        sdl (DataLoader):
            DataLoader instance with pointings information.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        verbose (bool, optional (default=`True`)):
            If `True`, prints out the selected events number with
            respect to the total number of photons in `data`.

    Returns:
        output (float):
            Total number of photons (fluence) attributed to the source
            landed on the sensitive (active) area of the detector.
    """
    data = source_catalogue_data(sourceID, catalogue.DLdata)
    coords = CoordEquatorial(data['RA'], data['DEC'])
    total_photons = select_source_photons(coords, sdl.DLdata, verbose)
    # remove photons fell in detector plane dead zone
    det_image = count(camera, total_photons)[0] * (camera.bulk > 0)
    return det_image.sum()


def extract_sources_info(
    cameraID: str,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
    verbose: bool = True,
) -> DataFrame:
    """
    Processes the entire source catalogue to compute the local angular coordinates and
    the true detected fluence for every unique source, returning the aggregated data
    in a structured pandas DataFrame.

    Args:
        cameraID (str):
            ID of the LEM-X module camera (e.g., 'cam1a' or 'cam1b').
        catalogue (CatalogueLoader):
            The catalogue object with sources metadata.
        sdl (DataLoader):
            DataLoader instance with pointings information.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with instrument geometry.
        verbose (bool, optional (default=`True`)):
            If `True`, prints out the selected events number with
            respect to the total number of photons in `data`.

    Returns:
        output (DataFrame):
            DataFrame in which are stored the camera local-frame angular coords and
            true observed fluence of all the sources in the simulated catalogue.
    """
    # compute sources camera local-frame angular coords and fluence
    sources, thetas_x, thetas_y, fluences = [], [], [], []
    for sourceID in np.unique(catalogue.DLdata['ID']):
        thx, thy = source_angular_coords(sourceID, catalogue, sdl, camera)
        fluence = source_fluence(sourceID, catalogue, sdl, camera, verbose)
        sources.append(sourceID)
        thetas_x.append(thx)
        thetas_y.append(thy)
        fluences.append(fluence)
    
    dmap = {
        cameraID: {
            'ID': sources,
            'angle_x': thetas_x,
            'angle_y': thetas_y,
            'fluence': fluences,
        },
    }
    return dict2df(dmap)


def data_DF(
    log: Log,
    catalogue: CatalogueLoader,
    sdl: DataLoader,
    camera: CodedMaskCamera,
) -> DataFrame:
    """
    Generates a Dataframe with output data from IROS.
    """
    raise NotImplementedError


def reconstructed_sources_profiles(
    true_sky: NDArray,
    log: Log,
    crp: tuple[int, int],
    camera: CodedMaskCamera,
    *,
    vignetting: bool = True,
    psfy: bool = True,
    save_to: str | Path | None = None,
) -> None:
    """
    Displays the IROS reconstruction effect wrt the original decoded sky.

    Specifically, it shows:
        - IROS reconstructed source slices wrt the true sky
        - source slices residues

    Args:
        true_sky (NDArray):
            True observed decoded sky.
        log (Log):
            Reconstructed sources database.
        crp (tuple[int, int]):
            Size of the cropping along fine/coarse directions.
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        vignetting (bool, optional (default=`True`)):
            Simulates vignetting effects.
        psfy (bool, optional (default=`True`)):
            Simulates detector reconstruction effects.
        save_to (str | Path | None, optional (default=`None`)):
            Path to the directory where to save the plots.
    
    ## Notes:
        - A copy of `true_sky` is used to avoid memory overwrite.
    """
    true_sky_ = true_sky.copy()
    res_ground = np.zeros_like(true_sky_)

    for source, sx, sy, f, x, y in zip(
        log.log['ID'],
        log.log['shift_x'],
        log.log['shift_y'],
        log.log['fluence'],
        log.log['x'],
        log.log['y'],
    ):
        POS = (y, x)

        # simulate source
        modeled = model_sky(
            camera=camera,
            shift_x=sx,
            shift_y=sy,
            fluence=f,
            vignetting=vignetting,
            psfy=psfy,
        ).astype(np.int32)

        # plot IROS vs True skies slices (profiles and residues)
        slices_plot(
            sky=(true_sky_, modeled),
            pos=POS,
            crp=crp,
            source=source,
            labels=('True', 'IROS'),
            cameraID=log.name,
            save_to=(
                savefile_to(save_to, f'{source.upper()}_profile_{log.name.upper()}')
                if save_to is not None else None
            ),
        )
        slices_plot(
            sky=(res_ground, true_sky_ - modeled),
            pos=POS,
            crp=crp,
            source=source,
            labels=('', 'true - IROS'),
            ylabel='residues [ph]',
            cameraID=log.name,
            save_to=(
                savefile_to(save_to, f'{source.upper()}_profile_res_{log.name.upper()}')
                if save_to is not None else None
            ),
        )

        # remove source from True sky
        true_sky_ -= modeled


def reconstruction_sources_heatmaps(
    true_sky: NDArray,
    log: Log,
    crp: tuple[int, int],
    camera: CodedMaskCamera,
    *,
    vignetting: bool = True,
    psfy: bool = True,
    save_to: str | Path | None = None,
) -> None:
    """
    Displays the IROS reconstruction effect wrt the original decoded sky.
    Specifically, it shows the heatmaps residues between the ground
    truth and the IROS reconstructed sources in the input `log`.

    Args:
        true_sky (NDArray):
            True observed decoded sky.
        log (Log):
            Reconstructed sources database.
        crp (tuple[int, int]):
            Size of the cropping along sky image (rows, cols).
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        vignetting (bool, optional (default=`True`)):
            Simulates vignetting effects.
        psfy (bool, optional (default=`True`)):
            Simulates detector reconstruction effects.
        save_to (str | Path | None, optional (default=`None`)):
            Path to the directory where to save the plots.
    
    ## Notes:
        - A copy of `true_sky` is used to avoid memory overwrite.
    """
    true_sky_ = true_sky.copy()

    for source, sx, sy, f, x, y in zip(
        log.log['ID'],
        log.log['shift_x'],
        log.log['shift_y'],
        log.log['fluence'],
        log.log['x'],
        log.log['y'],
    ):
        POS = (y, x)

        # simulate source
        modeled = model_sky(
            camera=camera,
            shift_x=sx,
            shift_y=sy,
            fluence=f,
            vignetting=vignetting,
            psfy=psfy,
        ).astype(np.int32)

        # plot IROS vs True residues heatmaps
        F_UPY, F_UPX = 2, 5

        true_crp, modeled_crp = map(
            lambda x: crop(x, pos=POS, crp=crp, strict=False),
            (true_sky_, modeled),
        )
        vlim = np.max(np.abs(true_crp - modeled_crp))
        image_plot(
            dmaps=map4image(
                img=upscale(true_crp - modeled_crp, F_UPY, F_UPX) * np.prod((F_UPY, F_UPX)),
                title=f'{source.upper()} True - IROS Residues {log.name}',
                cbarlabel='residues [ph]',
                img_kwargs={
                    'vmin': -vlim,
                    'vmax': vlim,
                    'cmap': 'bwr',
                    'extent': (-crp[1] * F_UPX, crp[1] * F_UPX, -crp[0] * F_UPY, crp[0] * F_UPY),
                },
            ),
            save_to=(
                savefile_to(save_to, f'{source.upper()}_resHM_{log.name.upper()}')
                if save_to is not None else None
            ),
        )

        # remove source from True sky
        true_sky_ -= modeled


# end