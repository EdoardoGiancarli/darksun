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
        dtx, dty = 1.0, 1.0
        run = np.rec.array([

            ('s1', 80.6, dtx, 34.0, dty, 5),         # associated directly
            ('s2', 84.0, dtx, 28.0, dty, 5),
            ('s3', 87.0, dtx, 23.0, dty, 5),
            ('s4', 90.0, dtx, 31.0, dty, 5),

            ('s6', -86.0, dtx, 21.2, dty, 5),        # associated through distance

            ('s9', -82.0, dtx, 18.9, dty, 5),        # associated through distance and removing 'gctr_diffuse'

            ('lemx-S1', 53.0, dtx, 68.0, dty, 5),    # associated with new sources
            ('lemx-S2', 23.0, dtx, 14.0, dty, 5),
            ('lemx-S3', 67.0, dtx, 35.0, dty, 5),
            ('lemx-S4', 15.0, dtx, 9.0, dty, 5),

            ('s2', 84.0, dtx, 28.0, dty, 3),         # repeating sources
            ('s2', 84.0, dtx, 28.0, dty, 1),
            ('s6', -86.0, dtx, 21.2, dty, 3),
            ('s6', -86.0, dtx, 21.2, dty, 1),
            ('s9', -82.0, dtx, 18.9, dty, 3),
            ('s9', -82.0, dtx, 18.9, dty, 1),
            ('lemx-S3', 67.0, dtx, 35.0, dty, 3),
            ('lemx-S3', 67.0, dtx, 35.0, dty, 1),

        ], dtype=[('ID', 'S20'), ('ANGLE_X', 'f8'), ('DANGLE_X', 'f8'), ('ANGLE_Y', 'f8'), ('DANGLE_Y', 'f8'), ('SNR', 'f8')])

        params = (
            LogEntry('angle_x', 'D', 'deg'),
            LogEntry('dangle_x', 'D', 'deg'),
            LogEntry('angle_y', 'D', 'deg'),
            LogEntry('dangle_y', 'D', 'deg'),
            LogEntry('snr', 'D', ''),
        )

        log = create_log(params)

        for entry in tuple(p.entry for p in log.params):
            log.add_entry_values(entry, list(run[entry.upper()]))
        
        self.log = log
        self.ids = [s.decode('utf-8') for s in run['ID']]  # convert b-str to str

        self.wfm = codedmask(_path_test_mask)
        self.sdl = get_data(_path_test_SDL)
    
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