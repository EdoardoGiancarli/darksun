"""
Tests for IROS data comparison with Catalogue.
"""

import unittest
from unittest import TestCase

import numpy as np

from darksun.types import LogEntry
from darksun.data import create_log, get_catalogue
from darksun.analyze import catalogue_comparison

from .assets import _path_test_catalogue


class TestCatalogueComparison(TestCase):
    """Tests for the `catalogue_comparison` methos in `analyze.py`."""

    def setUp(self):
        dra, ddec = 1.0, 1.0
        run = np.rec.array([

            ('s1', 257.0, dra, 52.0, ddec, 5),         # associated directly
            ('s2', 260.0, dra, 55.0, ddec, 5),
            ('s3', 263.0, dra, 58.0, ddec, 5),
            ('s4', 266.0, dra, 61.0, ddec, 5),

            ('s6', 270.0, dra, 65.0, ddec, 5),         # associated through distance

            ('s9', 274.0, dra, 69.0, ddec, 5),         # associated through distance and removing 'gctr_diffuse'

            ('lemx-s1', 278.0, dra, 73.0, ddec, 5),    # associated with new sources
            ('lemx-s2', 281.0, dra, 76.0, ddec, 5),
            ('lemx-s3', 284.0, dra, 79.0, ddec, 5),
            ('lemx-s4', 257.0, dra, 76.0, ddec, 5),

            ('s2', 260.0, dra, 55.0, ddec, 3),         # repeating sources
            ('s2', 260.0, dra, 55.0, ddec, 1),
            ('s6', 270.0, dra, 65.0, ddec, 3),
            ('s6', 270.0, dra, 65.0, ddec, 1),
            ('s9', 274.0, dra, 69.0, ddec, 3),
            ('s9', 274.0, dra, 69.0, ddec, 1),
            ('lemx-s3', 284.0, dra, 79.0, ddec, 3),
            ('lemx-s3', 284.0, dra, 79.0, ddec, 1),

        ], dtype=[('ID', 'S20'), ('RA', 'f8'), ('DRA', 'f8'), ('DEC', 'f8'), ('DDEC', 'f8'), ('SNR', 'f8')])

        params = (
            LogEntry('ra', 'D', 'deg'),
            LogEntry('dra', 'D', 'deg'),
            LogEntry('dec', 'D', 'deg'),
            LogEntry('ddec', 'D', 'deg'),
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
        )
        #print(log.to_dataframe())
        np.testing.assert_array_equal(
            np.array(log.log['ID'][:-2]),
            np.array(
                ['s1', 's2', 's3', 's4', 's6', 's9', 'lemx-s1', 'lemx-s2', 'lemx-s3', 'lemx-s4']
            ),
            strict=False,
        )




if __name__ == "__main__":
    unittest.main()


# end