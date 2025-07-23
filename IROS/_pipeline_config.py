"""
Configuration script for the IROS pipeline.
"""

from ._pipeline_support import PipelineParams
from ._pipeline_support import config_parameters
from ._pipeline_support import output_files

from typing import Sequence
from pathlib import Path

import numpy as np

from bloodmoon.io import simulation_files
from bloodmoon.mask import decode
from bloodmoon.mask import count
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.mask import codedmask

import darksun as ds


def run(params: PipelineParams) -> None:
    """
    IROS pipeline body.

    Args:
        params (PipelineParams):
            PipelineParams instance with the initialized parameters for the pipeline.
    """
    def is_file(filename: str | Path) -> bool:
        """Checks if input `filename` file exists."""
        if not isinstance(filename, Path):
            filename = Path(filename)
        return filename.is_file()
    
    CAM_A, CAM_B = params.wfm_cameras

    # upscaling setup
    UPSX_0, UPSY_0 = params.start_ups
    UPSX_FINAL, UPSY_FINAL = params.final_ups
    UPX_TO, UPY_TO = UPSX_FINAL - UPSX_0 + 1, UPSY_FINAL - UPSY_0 + 1

    # check on pipeline files
    output_files(params)

    # start pipeline
    with ds.timer("##### IROS PIPELINE #####"):

        # --- IROS SETUP
        print("\n#### IROS Setup...")
        with ds.timer("IROS Setup"):
            wfm = codedmask(
                mask_filepath=params.mask_file,
                upscale_x=UPSX_0,
                upscale_y=UPSY_0,
            )
            filepaths = simulation_files(params.simul_data)
            sdlA = ds.get_data(
                filepath=filepaths[CAM_A][params.dataset],
                E_min=params.E_min,
                E_max=params.E_max,
                coords=params.coords,
            )
            sdlB = ds.get_data(
                filepath=filepaths[CAM_B][params.dataset],
                E_min=params.E_min,
                E_max=params.E_max,
                coords=params.coords,
            )
            sdls = (sdlA, sdlB)

            with ds.timer("Compute dets/vars"):
                detectors = tuple(count(wfm, sdl.DLdata)[0] for sdl in sdls)
                variances = tuple(variance(wfm, d) for d in detectors)

            # WCS fit (here the camera is upscaled with the final upscaling)
            with ds.timer("WCS fit"):
                wfm_WCS = codedmask(
                    mask_filepath=params.mask_file,
                    upscale_x=UPSX_FINAL,
                    upscale_y=UPSY_FINAL,
                )
                wcs_fit = tuple(ds.fit_WCS(wfm_WCS, sdl) for sdl in sdls)
        

        # --- SAVING SIMULATED SKIES
        print("\n#### Saving Simulated Skies...")
        sim_camA, sim_camB = params.simul_names
        if (
            not is_file(sim_camA) or
            not is_file(sim_camB)
        ):
            skies = tuple(decode(wfm, d) for d in detectors)
            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPY_TO) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPY_TO) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.simul_names, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# Simulated skies already saved!")
        
        # sky composition
        if not is_file(params.simul_comp_name) and params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=sim_camA,
                    skyB_path=sim_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.simul_comp_name, comp_WCS)
        

        # --- RUN IROS AND SAVE OUTPUT + RESIDUES
        print("\n#### Running IROS...")
        res_camA, res_camB = params.res_names
        if not (
            is_file(params.iros_output_name) and
            is_file(res_camA) and
            is_file(res_camB)
        ):
            # IROS
            logs, skies = ds.run_IROS(
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                max_iterations=params.iros_max_iterations,
                snr_threshold=params.iros_snr_threshold,
                vignetting=params.vignetting,
                psfy=params.psfy,
                id_camA=CAM_A,
                id_camB=CAM_B,
            )
            # save output databases
            if not is_file(params.iros_output_name):
                log_camA, log_camB = logs
                ds.save_database(
                    log_camA=log_camA,
                    log_camB=log_camB,
                    sdlA=sdlA,
                    sdlB=sdlB,
                    save_to=params.iros_output_name,
                )
            # save IROS sky residues
            snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

            # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
            # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.res_names, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# IROS output data already saved!")
            log_camA, log_camB = ds.load_database(params.iros_output_name)
            skies = tuple(ds.load_sky(res)[0] for res in params.res_names)

        # sky composition
        if not is_file(params.res_comp_name) and params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=res_camA,
                    skyB_path=res_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.res_comp_name, comp_WCS)


        # --- COMPUTE SOURCES PARAMS WITH IROS OUTPUT
        print("\n#### Computing Sources Parameters...")
        if not is_file(params.iros_data_name):
            log_camA = ds.compute_parameters(
                log=log_camA,
                camera=wfm,
                sdl=sdlA,
                vignetting=params.vignetting,
                psfy=params.psfy,
            )
            log_camB = ds.compute_parameters(
                log=log_camB,
                camera=wfm,
                sdl=sdlB,
                vignetting=params.vignetting,
                psfy=params.psfy,
            )
            ds.save_database(
                log_camA=log_camA,
                log_camB=log_camB,
                sdlA=sdlA,
                sdlB=sdlB,
                save_to=params.iros_data_name,
            )
        else:
            print("# Candidates parameters already computed!")
            log_camA, log_camB = ds.load_database(params.iros_data_name)


        # --- CATALOG COMPARISON AND DATABASE UPDATE
        print("\n#### Performing Catalogue Comparison...")
        if not is_file(params.DB_name):
            catA = ds.get_catalogue(
                filepath=filepaths[CAM_A]["sources"],
                n=params.n,
                F_max=params.F_max,
                F_min=params.F_min,
            )
            log_camA = ds.catalogue_comparison(
                log=log_camA,
                catalogue=catA,
                sdl=sdlA,
                camera=wfm,
                screening=True,
            )
            catB = ds.get_catalogue(
                filepath=filepaths[CAM_B]["sources"],
                n=params.n,
                F_max=params.F_max,
                F_min=params.F_min,
            )
            log_camB = ds.catalogue_comparison(
                log=log_camB,
                catalogue=catB,
                sdl=sdlB,
                camera=wfm,
                screening=True,
            )
            ds.save_database(
                log_camA=log_camA,
                log_camB=log_camB,
                sdlA=sdlA,
                sdlB=sdlB,
                save_to=params.DB_name,
            )
        else:
            print("# Catalogue comparison already done!")
            log_camA, log_camB = ds.load_database(params.DB_name)


        # --- RECONSTRUCTING SKIES FROM IROS OUTPUT + RESIDUES
        print("\n#### Reconstructing and Saving IROS Output Skies...")
        out_camA, out_camB = params.out_names
        logs = (log_camA, log_camB)
        if (
            not is_file(out_camA) or
            not is_file(out_camB)
        ):
            with ds.timer("IROS reconstructed skies"):
                skies = tuple(
                    ds.make_sky(logID.log, wfm, params.vignetting, params.psfy, res)
                    for logID, res in zip(logs, skies)
                )
                snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skies, variances))

                # ups_skies = tuple(upscale(sky, upscale_y=UPSY_FINAL - UPSY_0 + 1) for sky in skies)
                # ups_snrs = tuple(upscale(snr, upscale_y=UPSY_FINAL - UPSY_0 + 1) for snr in snrs)

            for sky, snr, sdl, name, wcs in zip(skies, snrs, sdls, params.out_names, wcs_fit):
                if not is_file(name):
                    ds.save_sky(sky, snr, sdl, name, wcs)
        else:
            print("# IROS reconstructed skies already saved!")
        
        # sky composition
        if not is_file(params.out_comp_name) and params.sky_compositions:
            with ds.timer("Camera composition"):
                comp_sky, comp_snr, comp_WCS = ds.WFM_composition(
                    skyA_path=out_camA,
                    skyB_path=out_camB,
                )
                ds.save_sky(comp_sky, comp_snr, sdlA, params.out_comp_name, comp_WCS)
    

    # final check on pipeline files
    output_files(params, check_out=False)


def run_pipeline(
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
    iros_max_iterations: int = 20,
    iros_snr_threshold: int | float = 5,
    sky_compositions: bool = False,
    energy_range: tuple[int | float | None, int | float | None] | None = None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    n: int | tuple[int, int] | None = None,
    flux_range: tuple[int | float | None, int | float | None] | None = None,
) -> None:
    """
    Runs the IROS pipeline.
    This method initializes the parameters for the pipeline and then runs
    the analysis procedure, divided in blocks and with respective checkpoints.

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
        iros_max_iterations (int, optional (default=`20`)):
            Maximum number of iterations for the IROS loop.
        iros_snr_threshold (int | float, optional (default=`5`)):
            Minimum SNR value required to continue the iterative source removal process.
        sky_compositions (bool, optional (default=`False`)):
            Flag for WFM sky compositions.
        energy_range (tuple[int | float | None, int | float | None] | None, optional (default=`None`)):
            Energy range in keV for the data filtering, to be interpreted as (`E_min`, `E_max`).
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None, optional (default=`None`)):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
        n (int | tuple[int, int] | None, optional (default=`None`)):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range (tuple[int | float | None, int | float | None] | None, optional (default=`None`)):
            Flux range in ph/cm2/s for the data filtering, to be interpreted as (`F_min`, `F_max`).
    """
    params: PipelineParams = config_parameters(
        mask=mask,
        thin_mask=thin_mask,
        skyfield=skyfield,
        skydata=skydata,
        wfm_cameras=wfm_cameras,
        dataset=dataset,
        start_ups=start_ups,
        final_ups=final_ups,
        analysisID=analysisID,
        iros_max_iterations=iros_max_iterations,
        iros_snr_threshold=iros_snr_threshold,
        sky_compositions=sky_compositions,
        energy_range=energy_range,
        coords=coords,
        n=n,
        flux_range=flux_range,
    )
    run(params)


# end