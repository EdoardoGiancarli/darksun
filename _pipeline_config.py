"""
Configuration script for the IROS pipeline.
"""

from _pipeline_support import PipelineParams
from _pipeline_support import output_files

from pathlib import Path

import numpy as np

from bloodmoon.io import simulation_files
from bloodmoon.mask import decode
from bloodmoon.mask import count
from bloodmoon.mask import variance
from bloodmoon.mask import snratio
from bloodmoon.mask import codedmask

import darksun as ds


def run_pipeline(params: PipelineParams) -> None:
    """
    Runs the IROS pipeline.

    Args:
        params (PipelineParams):
            PipelineParams instance with the initialized parameters for the pipeline.
    """
    def is_file(filename: str | Path) -> bool:
        """Checks if input `filename` file exists."""
        if not isinstance(filename, Path):
            filename = Path(filename)
        return filename.is_file()

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
            cam_a, cam_b = params.wfm_cameras
            wfm = codedmask(
                mask_filepath=params.mask_file,
                upscale_x=UPSX_0,
                upscale_y=UPSY_0,
            )
            filepaths = simulation_files(params.simul_data)
            sdlA = ds.get_data(
                filepath=filepaths[cam_a][params.dataset],
                energy_range=params.energy_range,
                coords=params.coords,
            )
            sdlB = ds.get_data(
                filepath=filepaths[cam_b][params.dataset],
                energy_range=params.energy_range,
                coords=params.coords,
            )
            sdls = (sdlA, sdlB)

            with ds.timer("Compute dets/vars"):
                detectors = tuple(count(wfm, sdl.data)[0] for sdl in sdls)
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
            log_camA, log_camB, skies = ds.run_IROS(
                camera=wfm,
                sdl_camA=sdlA,
                sdl_camB=sdlB,
                max_iterations=params.iros_max_iterations,
                snr_threshold=params.iros_snr_threshold,
                vignetting=params.vignetting,
                psfy=params.psfy,
                id_camA=cam_a,
                id_camB=cam_b,
            )
            # save output databases
            if not is_file(params.iros_output_name):
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



        # --- CATALOG COMPARISON AND DATABASE UPDATE



        # --- GENERATING SKIES FROM IROS OUTPUT + RESIDUES
    


    # final check on pipeline files
    output_files(params, check_out=False)


# end