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
# from .types import Candidate               (when defined version)

# from .analyze import run_IROS              (when defined version)
from .analyze import compute_parameters
from .analyze import catalogue_comparison

from .data import create_log
from .data import get_data
from .data import get_catalogue
from .data import fit_WCS

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
from .images import make_sky
from .images import WFM_composition

from .optim import bkg_smoothing
#from .optim import iros                     (when defined version)

from .show import map4plot
from .show import plot
from .show import distr_plot
from .show import map4image
from .show import image_plot
from .show import slices_plot
from .show import reconstruction_plot
from .show import skyfield_map

# from .stats import __all__

from .utils import timer
from .utils import benchmark_func


# end