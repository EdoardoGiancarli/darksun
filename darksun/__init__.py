r"""
          ____                   _                          
         |  _ \    __ _   _ __  | | __  ___   _   _   _ __  
         | | | |  / _` | | '__| | |/ / / __| | | | | | '_ \ 
         | |_| | | (_| | | |    |   <  \__ \ | |_| | | | | |
         |____/   \__,_| |_|    |_|\_\ |___/  \__,_| |_| |_|


Darksun is a `bloodmoon` package for handling IROS-based analyses.
"""

__version__ = "0.1.0"
__author__ = "Edoardo Giancarli"


from .types import LogEntry
from .types import Tag
from .types import Candidate

# from .analyze import run_IROS              (when defined version)
from .analyze import compute_parameters
from .analyze import catalogue_comparison

from .data import create_log
from .data import get_data
from .data import get_catalogue
from .data import fit_WCS

from .filtering import select_source_photons
from .filtering import filter_data
from .filtering import source_filter
from .filtering import flux_filter
from .filtering import filter_catalogue

from .handle import save_database
from .handle import save_sky
from .handle import load_database
from .handle import load_sky

from .images import upscale
from .images import downscale
from .images import crop
from .images import unframe
from .images import collapse_view
from .images import make_sky
from .images import WFM_composition

from .optim import bkg_smoothing
from .optim import get_candidates
from .optim import retrieve_detector
from .optim import detector_smoothing
#from .optim import iros                     (when defined version)

from .show import map4plot
from .show import plot
from .show import distr_plot
from .show import map4image
from .show import image_plot
from .show import slices_plot
from .show import skyfield_map

from .benchmarking import config_distr_limits
from .benchmarking import pixels_angular_resolution
from .benchmarking import psf_extension
from .benchmarking import crop_source_psf
from .benchmarking import source_angular_coords
from .benchmarking import source_fluence
from .benchmarking import extract_sources_info
from .benchmarking import data_DF
from .benchmarking import reconstructed_sources_profiles
from .benchmarking import reconstruction_sources_heatmaps

from .utils import timer
from .utils import benchmark_func
from .utils import savefile_to


# end