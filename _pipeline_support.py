"""
Support methods for the IROS pipeline.
"""

import os
from typing import Sequence
from pathlib import Path
from dataclasses import dataclass

from bloodmoon.types import CoordEquatorial


OS_SELECT = {
    'DEBIAN': '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data',
    'WSL': '/mnt/d/PhD_AASS/Coding/Images_fits',
}

def _handle_dirpaths(
    mask: str,
    skyfield: str,
    simul: str,
    run_name: str | None = None,
) -> tuple[str]:
    """Handles paths depending on the OS."""
    # TODO: use os.path.join to be more general (still, the root folder has to be written as `/mnt`...)

    # define paths for data and output files
    if Path(base_path := OS_SELECT['DEBIAN']).is_dir():
        mask_path = f"{base_path}/Simulations/{mask}"                 # dirpath to WFM mask file
        data_path = f"{base_path}/Simulations/{skyfield}/{simul}/"    # dirpath with simul files

        if run_name:
            save_path = f"{base_path}/Outputs/Out{skyfield}/{simul}"  # dirpath to save output data
            if not Path(f"{save_path}/{run_name}").is_dir():
                os.mkdir(f"{save_path}/{run_name}")
            save_path += f"/{run_name}/"
        else:
            save_path = None
        
    elif Path(base_path := OS_SELECT['WSL']).is_dir():
        mask_path = f"{base_path}/{mask}"
        data_path = f"{base_path}/{skyfield}/{simul}/"

        if run_name:
            save_path = base_path
            if not Path(f"{save_path}/{run_name}").is_dir():
                os.mkdir(f"{save_path}/{run_name}")
            save_path += f"/{run_name}/"
        else:
            save_path = None

    else:
        raise ValueError("A0, ma ndo sei finit*?")
    
    # check mask FITS file and paths
    if not Path(mask_path).is_file():
        raise ValueError(f"WFM mask '{mask}' does not exist.")
    for name, dirpath in zip(
            ("data_path", "save_path"),
            (data_path, save_path),
        ):
            if (dirpath and not Path(dirpath).is_dir()):
                raise ValueError(f"{name} '{dirpath}' does not exist.")

    return mask_path, data_path, save_path


def handle_instrument_effects(
    thin_mask: bool,
    dataset: str,
) -> tuple[bool, bool]:
    """Handles CAI vignetting and psf corrections along y-axis."""

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("'dataset' must be either 'detected' or 'reconstructed'.")
    
    vignetting = not thin_mask
    psfy = False if dataset == "detected" else True
    return vignetting, psfy


@dataclass(frozen=True)
class PipelineParams:
    """
    Container for the IROS pipeline parameters.

    Attributes:
        mask_file (str):
            Directory path to the mask FITS file.
        simul_data (str):
            Directory path to the simulated data FITS file.
        save_path (str):
            Output FITS files directory path.
        vignetting (bool):
            Flag for vignetting effect on the detector.
        psfy (bool):
            Flag for detector PSF effect along the y axis.
        wfm_cameras (tuple[str, str]):
            Name of the WFM cameras (e.g., `('cam1a', 'cam1b')`).
        dataset (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int, int]):
            Starting upscaling values.
        final_ups (tuple[int, int]):
            Final upscaling values.
        iros_max_iterations (int):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool):
            Flag for WFM sky compositions.
        simul_names (tuple[str, str]):
            Names for the simulated skies FITS files.
        simul_comp_name (str):
            Name for the simulated sky composition FITS file.
        iros_output_name (str):
            Name for the output IROS database FITS file.
        res_names (tuple[str, str]):
            Names for the IROS sky residuals FITS files.
        res_comp_name (str):
            Name for the IROS sky residuals composition FITS file.
        iros_data_name (str):
            Name for the output IROS sources parameters database FITS file.
        DB_name (str):
            Name for the output database with the identified sources parameters FITS file.
        out_names (tuple[str, str]):
            Names for the IROS reconstructed skies FITS files.
        out_comp_name (str):
            Name for the IROS reconstructed sky composition FITS file.
        E_min (int | float | None):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int] | None):
            Filtered interval of sources.
        flux_range (tuple[int | float | None, int | float | None]):
            Flux range in ph/cm2/s for the data filtering.
    """
    mask_file: str
    simul_data: str
    save_path: str
    vignetting: bool
    psfy: bool
    wfm_cameras: tuple[str, str]
    dataset: str
    start_ups: tuple[int, int]
    final_ups: tuple[int, int]
    iros_max_iterations: int
    iros_snr_threshold: int | float
    sky_compositions: bool
    simul_names: tuple[str, str]
    simul_comp_name: str
    iros_output_name: str
    res_names: tuple[str, str]
    res_comp_name: str
    iros_data_name: str
    DB_name: str
    out_names: tuple[str, str]
    out_comp_name: str
    E_min: int | float | None
    E_max: int | float | None
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None
    n: int | tuple[int, int] | None
    flux_range: tuple[int | float | None, int | float | None]


def config_parameters(
    *,
    mask: str,
    thin_mask: bool,
    skyfield: str,
    skydata: str,
    wfm_cameras: tuple[str, str],
    dataset: str,
    start_ups: tuple[int, int],
    final_ups: tuple[int, int],
    analysisID: str | None,
    iros_max_iterations: int,
    iros_snr_threshold: int | float,
    sky_compositions: bool,
    energy_range: tuple[int | float | None, int | float | None] | None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None,
    n: int | tuple[int, int] | None,
    flux_range: tuple[int | float | None, int | float | None] | None,
) -> PipelineParams:
    """
    Configures the IROS pipeline by processing input parameters.

    Args:
        mask (str):
            Name of the mask FITS file.
        thin_mask (bool):
            Indicates if the mask is infinite thin and/or absorbent or realistic.
        skyfield (str):
            Name of the sky-field simulation (e.g., 'Crab', 'GalacticCenter', ...).
        skydata (str):
            Name of the directory with the sky-data simulation.
        wfm_cameras (tuple[str, str]):
            Name of the WFM cameras (e.g., `('cam1a', 'cam1b')`).
        dataset (str):
            Photons position reconstruction effects. Either 'detected' or 'reconstructed'.
        start_ups (tuple[int, int]):
            Starting upscaling values (x, y).
        final_ups (tuple[int, int]):
            Final upscaling values (x, y).
        analysisID (str | None):
            Test name.
        iros_max_iterations (int):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool):
            Flag for WFM sky compositions.
        energy_range (tuple[int | float | None, int | float | None] | None):
            Energy range in keV for the data filtering, to be interpreted as (`E_min`, `E_max`).
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int, int] | None):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range (tuple[int | float | None, int | float | None] | None):
            Flux range in ph/cm2/s for the data filtering, to be interpreted as (`F_min`, `F_max`).
    
    Returns:
        params (PipelineParams):
            Output PipelineParams instance with all the parameters to run the pipeline.
    
    Raises:
        ValueError: If `n` or `flux_range` are both specified for catalogs filtering.
    """
    def report(params: PipelineParams) -> None:
        """Prints out IROS pipeline info."""
        print(
            f"\n# IROS Pipeline Report\n"
            f"  - Testing skyfield: '{skyfield}'\n"
            f"  - Output folder name: '{analysisID}'\n"
            f"  - Dataset type: '{dataset}'\n"
            f"  - Mask type: '{"ideal" if thin_mask else "realistic"}'\n"
            f"  - Vignetting: {params.vignetting}\n"
            f"  - Psfy: {params.psfy}\n"
            f"  - Starting upscaling (x, y): {start_ups}\n"
            f"  - Final upscaling (x, y): {final_ups}\n"
            f"  - Max IROS iteration: {iros_max_iterations}\n"
            f"  - Sky compositions: {sky_compositions}\n"
            f"  - Filtered photons energy range [keV]: {energy_range}\n"
            f"  - Excluded photons RA/Dec [deg]: {coords}\n"
            f"  - Catalog selected brighest sources: {n}\n"
            f"  - Catalog sources flux min/range [ph/cm2/s]: {flux_range}\n"
        )

    # configure n and flux_range for catalogs filtering
    if n and flux_range:
        raise ValueError("Specify either 'n' or 'flux_range' to filter the catalog.")

    # file directory paths (mask, simulated, where to save output files)
    mask_file, simul_data, save_path = _handle_dirpaths(
        mask=mask,
        skyfield=skyfield,
        simul=skydata,
        run_name=analysisID,
    )

    # mask/detector corrections
    vignetting, psfy = handle_instrument_effects(thin_mask, dataset)
    
    # output files names (simul skies, iros output DB and sky residuals, sources and catalog-compared DB, IROS skies)
    cam_a, cam_b = wfm_cameras

    simul_names = tuple(save_path + f"sky_SIMUL_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    simul_comp_name = save_path + f"COMPOSED_sky_SIMUL_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    iros_output_name = save_path + f"IROS_output_TEST_{analysisID}.fits"
    res_names = tuple(save_path + f"skyRES_IROS_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    res_comp_name = save_path + f"COMPOSED_skyRES_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    iros_data_name = save_path + f"IROS_data_TEST_{analysisID}.fits"
    DB_name = save_path + f"IROS_sources_database_TEST_{analysisID}.fits"

    out_names = tuple(save_path + f"OUTsky_IROS_{cam.upper()}_TEST_{analysisID}.fits" for cam in (cam_a, cam_b))
    out_comp_name = save_path + f"COMPOSED_OUTsky_IROS_{cam_a.upper()}_{cam_b.upper()}_TEST_{analysisID}.fits"

    # filters params
    if energy_range is None: E_min, E_max = None, None
    elif any(energy_range): E_min, E_max = energy_range

    params = PipelineParams(
        mask_file=mask_file,
        simul_data=simul_data,
        save_path=save_path,
        vignetting=vignetting,
        psfy=psfy,
        wfm_cameras=wfm_cameras,
        dataset=dataset,
        start_ups=start_ups,
        final_ups=final_ups,
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        sky_compositions=sky_compositions,
        simul_names=simul_names,
        simul_comp_name=simul_comp_name,
        iros_output_name=iros_output_name,
        res_names=res_names,
        res_comp_name=res_comp_name,
        iros_data_name=iros_data_name,
        DB_name=DB_name,
        out_names=out_names,
        out_comp_name=out_comp_name,
        E_min=E_min,
        E_max=E_max,
        coords=coords,
        n=n,
        flux_range=flux_range,
    )
    report(params)
    return params


def output_files(
    params: PipelineParams,
    check_out: bool = True,
) -> None:
    """Prints out which pipeline files have been saved."""

    def check(filename: str) -> str:
        return "SAVED" if Path(filename).is_file() else "MISSING"
    
    cam_a, cam_b = params.wfm_cameras
    pipeline_files = {
        "Simulation Files": [
            (f"Simulated Sky {cam_a.upper()}", params.simul_names[0]),
            (f"Simulated Sky {cam_b.upper()}", params.simul_names[1]),
            ("Simulated Sky (composed)", params.simul_comp_name),
        ],
        "Residuals Files": [
            (f"Residual Sky {cam_a.upper()}", params.res_names[0]),
            (f"Residual Sky {cam_b.upper()}", params.res_names[1]),
            ("Residual Sky (composed)", params.res_comp_name),
        ],
        "IROS Output Files": [
            (f"Output Sky {cam_a.upper()}", params.out_names[0]),
            (f"Output Sky {cam_b.upper()}", params.out_names[1]),
            ("Output Sky (composed)", params.out_comp_name),
        ],
        "Databases Files": [
            ("IROS output DB", params.iros_output_name),
            ("IROS sources parameters", params.iros_data_name),
            ("IROS-catalog comparison", params.DB_name),
        ],
    }
    check_list = []
    print("\n#### GENERATED FILES")
    for category, files_list in pipeline_files.items():
        print(f"# {category}")
        for name, path in files_list:
            check_list.append(status := check(path))
            print(f"  - {name}: {status}")
    
    if check_out and "MISSING" not in check_list:
        print("\nAll files present!")
        exit()


# end