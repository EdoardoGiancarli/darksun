r"""
             ___   ____     ___    ____         ____    _                  _   _         
            |_ _| |  _ \   / _ \  / ___|       |  _ \  (_)  _ __     ___  | | (_)  _ __  
             | |  | |_) | | | | | \___ \       | |_) | | | | '_ \   / _ \ | | | | | '_ \   / _ \
             | |  |  _ <  | |_| |  ___) |      |  __/  | | | |_) | |  __/ | | | | | | | | |  __/
            |___| |_| \_\  \___/  |____/       |_|     |_| | .__/   \___| |_| |_| |_| |_|  \___|
                                                           |_|


Pipeline for running IROS, with checkpoints.

Layout:
    1. Pipeline params set up (in this module, through the variables below)
    2. IROS params set up
    3. Saving simulated skies and composition (with significance)
    4. Run IROS, saving output and residues (with significance + composition)
    5. Computing and saving candidates parameters for post-IROS analysis
    6. Candidates comparison with given catalogs, saving updated database
    7. Generating and saving output skies (IROS sources + IROS residues, with significance + composition)

Notes:
    - Variables are re-assigned to reduce memory cost
    - For each step there is a checkpoint, so that if saved FITS are already in the base directory, those
      steps are ignored (to handle compiler crashes).
      To repeat the step, just delete the output files of that step.

Dependencies for running the pipeline:
    - Change paths for data in `_handle_dirpaths()` in `_pipeline_support.py`

TODO:
    - fix skies upscaling for output visualisation
    - insert possibility to load residuals of proper shapes to act as BKG for output IROS skies (not oversampled)
    - generalize directory paths for all users
    - setup .json file to give it as input to this module
    - WARNING: new-sources association must be updated to consider repeating same new-source
"""

from IROS._pipeline_config import run_pipeline
from typing import Sequence
from bloodmoon.types import CoordEquatorial


"""
PIPELINE SET-UP.
"""
#### --- WIDE FIELD MONITOR MASK
MASK_FITS: str = "wfm_mask.fits"
THIN_MASK: bool = False             # selects if infinitely opaque and thin mask (removes vignetting effects)

#### --- OBSERVATION DATA
#SKYFIELD: str = "GalacticCenter"                                                          # skyfield selection
#DATA_FITS: str = "20250430_galctr_rxte_sax_2-50keV_1ks_realmask_infdet_sources_cxb"       # directory with FITS files from WFM

SKYFIELD: str = "IROSDummy"
DATA_FITS: str = "catalog_withCXB_1Crab_infdet_2-50keV_1ks"

ID_CAMERA_A: str = "cam1a"
ID_CAMERA_B: str = "cam1b"
DATASET: str = "detected"

#### --- IMAGES UPSCALING
UPSX_0: int = 5                     # initial upscaling (with which IROS is performed)
UPSY_0: int = 1

UPSX_FINAL: int = 5                 # final upscaling for skies and visualisation
UPSY_FINAL: int = 1

#### --- ANALYSIS ID
ANALYSIS_ID: str = "errorbox_wrt_coords"

#### --- IROS SETUP
MAX_ITERATIONS: int = 9
SNR_THRESHOLD: int | float = 10

WFM_SKY_COMPOSITION: bool = False   # selects if the WFM cameras are to be joined to get the composed sky

#### --- DATA FILTERS SETUP
# photons energy filter - [keV]
PHOTONS_ENERGY_RANGE: tuple[int | float | None, int | float | None] | None = None
# RA/Dec filter (sources to filter out) - [deg]
PHOTONS_COORDS: CoordEquatorial | Sequence[CoordEquatorial] | None = None

# number of sources in the catalog for comparison
CATALOGUE_NUM_BRIGHT_SOURCES: int | tuple[int, int] | None = None
# sources flux filter for the catalog comparison - [ph/cm2/s]
CATALOGUE_FLUX_RANGE: tuple[int | float | None, int | float | None] | None = None # (9e-2, None)



if __name__ == "__main__":

    """
    RUNNING PIPELINE.
    """
    print("\n#### RUNNING PIPELINE..")
    run_pipeline(
        mask=MASK_FITS,
        thin_mask=THIN_MASK,
        skyfield=SKYFIELD,
        skydata=DATA_FITS,
        wfm_cameras=(ID_CAMERA_A, ID_CAMERA_B),
        dataset=DATASET,
        start_ups=(UPSX_0, UPSY_0),
        final_ups=(UPSX_FINAL, UPSY_FINAL),
        analysisID=ANALYSIS_ID,
        iros_max_iterations=MAX_ITERATIONS,
        iros_snr_threshold=SNR_THRESHOLD,
        sky_compositions=WFM_SKY_COMPOSITION,
        energy_range=PHOTONS_ENERGY_RANGE,
        coords=PHOTONS_COORDS,
        n=CATALOGUE_NUM_BRIGHT_SOURCES,
        flux_range=CATALOGUE_FLUX_RANGE,
    )


# end