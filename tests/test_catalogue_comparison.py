"""
Tests for IROS data comparison with Catalogue.
"""

import unittest
from unittest import TestCase

import numpy as np

from bloodmoon.mask import codedmask

from darksun.types import LogEntry
from darksun.data import create_log, get_data, get_catalogue
from darksun.analyze import catalogue_comparison

from .assets import _path_test_catalogue
from .assets import _path_test_SDL
from .assets import _path_test_mask


class TestCatalogueComparison(TestCase):
    """Tests for the `catalogue_comparison` method in `analyze.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask)
        self.sdl = get_data(_path_test_SDL)

        # sky-coords shifts errors in [mm]
        dsx, dsy = 250.0, 25.0
        run = np.rec.array([

            ('s1', 1347.0, dsx, 140.0, dsy, 5),     # associated directly
            ('s2', 1750.0, dsx, 109.0, dsy, 5),
            ('s3', 3388.0, dsx, 84.0, dsy, 5),
            ('s4', 72781.0, dsx, 122.0, dsy, 5),

            ('s6', -2950.0, dsx, 80.0, dsy, 5),      # associated through distance

            ('s9', -1440.0, dsx, 75.0, dsy, 5),      # associated through distance and removing 'gctr_diffuse'

            ('lemx-S1', -2110.0, dsx, -2180.0, dsy, 5),    # associated with new sources
            ('lemx-S2', -2120.0, dsx, -2170.0, dsy, 5),
            ('lemx-S3', -2130.0, dsx, -2160.0, dsy, 5),
            ('lemx-S4', -2140.0, dsx, -2150.0, dsy, 5),

            ('s2', 1750.0, dsx, 109.0, dsy, 3),      # repeating sources
            ('s2', 1750.0, dsx, 109.0, dsy, 1),
            ('s6', -2950.0, dsx, 80.0, dsy, 3),
            ('s6', -2950.0, dsx, 80.0, dsy, 1),
            ('s9', -1440.0, dsx, 75.0, dsy, 3),
            ('s9', -1440.0, dsx, 75.0, dsy, 1),
            ('lemx-S3', -2130.0, dsx, -2160.0, dsy, 3),
            ('lemx-S3', -2130.0, dsx, -2160.0, dsy, 1),

        ], dtype=[('ID', 'S20'), ('SHIFT_X', 'f8'), ('DSHIFT_X', 'f8'), ('SHIFT_Y', 'f8'), ('DSHIFT_Y', 'f8'), ('SNR', 'f8')])
            
        params = (
            LogEntry('shift_x', 'D', 'deg'),
            LogEntry('dshift_x', 'D', 'deg'),
            LogEntry('shift_y', 'D', 'deg'),
            LogEntry('dshift_y', 'D', 'deg'),
            LogEntry('snr', 'D', ''),
        )

        log = create_log(params)

        for entry in tuple(p.entry for p in log.params):
            log.add_entry_values(entry, list(run[entry.upper()]))
        
        self.log = log
        self.ids = [s.decode('utf-8') for s in run['ID']]  # convert b-str to str
    
    def test_complete_comparison(self):
        """Tests if `catalogue_comparison` correctly works."""
        catalogue = get_catalogue(_path_test_catalogue)
        log = catalogue_comparison(
            log=self.log,
            catalogue=catalogue,
            sdl=self.sdl,
            camera=self.wfm,
            screening=False,
        )
        #print(log.to_dataframe())
        np.testing.assert_array_equal(
            np.array(log.log['ID'][:-2]),
            np.array(self.ids[:-2]),
            strict=False,
        )
    
    def test_screening_comparison(self):
        """Tests if repeating sources are removed."""
        catalogue = get_catalogue(_path_test_catalogue)
        log = catalogue_comparison(
            log=self.log,
            catalogue=catalogue,
            sdl=self.sdl,
            camera=self.wfm,
        )
        #print(log.to_dataframe())
        np.testing.assert_array_equal(
            np.array(log.log['ID'][:-2]),
            np.array(
                ['s1', 's2', 's3', 's4', 's6', 's9', 'lemx-S1', 'lemx-S2', 'lemx-S3', 'lemx-S4']
            ),
            strict=False,
        )




if __name__ == "__main__":
    unittest.main()


# end