"""
Tests for simulated data and catalog filters.
"""

import unittest
from unittest import TestCase

import numpy as np

from bloodmoon.types import CoordEquatorial
from darksun.filtering import filter_data
from darksun.filtering import flux_filter
from darksun.filtering import source_filter
from darksun.filtering import filter_catalog


class TestFilters(TestCase):
    """Tests for the filters in `filtering.py`."""
    
    def setUp(self):
        """Initialize the photons list and the catalog."""
        # simulated list of photons
        self.data = np.rec.array([
            (1,  10.684,  41.269, 22.5),
            (2,  83.822,  -5.391, 35.2),
            (3, 201.365, -43.019, 48.7),
            (4, 150.025,   2.312, 21.9),
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (7, 299.868,  40.733, 39.3),
            (8, 187.706,  12.391, 26.8),
            (9, 123.456, -10.123, 30.4),
            (10, 250.349,  36.467, 47.0),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])
        
        # simulated catalog for single run (e.g., 1ks exposure)
        self.catalog = np.rec.array([
            ('SRC_A', 12.4, 120),
            ('SRC_B', 3.5, 98),
            ('SRC_C', 87.2, 143),
            ('SRC_D', 0.95, 65),
            ('SRC_E', 56.7, 87),
            ('SRC_F', 23.1, 132),
            ('SRC_G', 71.8, 77),
            ('SRC_H', 99.9, 160),
            ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        # simulated catalog for multiple runs (e.g., 3ks exposure)
        self.catalog_mult_runs = np.rec.array([
            ('SRC_A', 12.4, 120), ('SRC_A', 12.4, 120), ('SRC_A', 12.4, 120),
            ('SRC_B', 3.5, 98), ('SRC_B', 3.5, 98), ('SRC_B', 3.5, 98),
            ('SRC_C', 87.2, 143), ('SRC_C', 87.2, 143),  ('SRC_C', 87.2, 143),
            ('SRC_D', 0.95, 65), ('SRC_D', 0.95, 65), ('SRC_D', 0.95, 65),
            ('SRC_E', 56.7, 87), ('SRC_E', 56.7, 87), ('SRC_E', 56.7, 87),
            ('SRC_F', 23.1, 132), ('SRC_F', 23.1, 132), ('SRC_F', 23.1, 132),
            ('SRC_G', 71.8, 77), ('SRC_G', 71.8, 77), ('SRC_G', 71.8, 77),
            ('SRC_H', 99.9, 160), ('SRC_H', 99.9, 160), ('SRC_H', 99.9, 160),
            ('SRC_I', 14.6, 101), ('SRC_I', 14.6, 101), ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110), ('SRC_J', 42.3, 110), ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])


    def test_data_energy_filter(self):
        """Tests for `filter_data()` in the energy channel."""
        E_min, E_max = None, 30
        filtered_data = filter_data(
            data=self.data,
            E_min=E_min,
            E_max=E_max,
            coords=None,
        )

        target = np.rec.array([
            (1,  10.684,  41.269, 22.5),
            (4, 150.025,   2.312, 21.9),
            (5,  53.125, -27.800, 29.5),
            (8, 187.706,  12.391, 26.8),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(filtered_data, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )

        E_min, E_max = (25, 45)
        filtered_data = filter_data(
            data=self.data,
            E_min=E_min,
            E_max=E_max,
            coords=None,
        )

        target = np.rec.array([
            (2,  83.822,  -5.391, 35.2),
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (7, 299.868,  40.733, 39.3),
            (8, 187.706,  12.391, 26.8),
            (9, 123.456, -10.123, 30.4),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(filtered_data, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )

    def test_data_coords_filter(self):
        """Tests for `filter_data()` in the RA/Dec channel."""
        coords = CoordEquatorial(ra=201.365, dec=-43.019)
        filtered_data = filter_data(
            data=self.data,
            E_min=None,
            E_max=None,
            coords=coords,
        )

        target = np.rec.array([
            (1,  10.684,  41.269, 22.5),
            (2,  83.822,  -5.391, 35.2),
            (4, 150.025,   2.312, 21.9),
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (7, 299.868,  40.733, 39.3),
            (8, 187.706,  12.391, 26.8),
            (9, 123.456, -10.123, 30.4),
            (10,250.349,  36.467, 47.0),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(filtered_data, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )

        coords = [
            CoordEquatorial(ra=299.868, dec=40.733),
            CoordEquatorial(ra=123.456, dec=-10.123),
            CoordEquatorial(ra=83.822,  dec=-5.391),
        ]
        filtered_data = filter_data(
            data=self.data,
            E_min=None,
            E_max=None,
            coords=coords,
        )

        target = np.rec.array([
            (1,  10.684,  41.269, 22.5),
            (3, 201.365, -43.019, 48.7),
            (4, 150.025,   2.312, 21.9),
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (8, 187.706,  12.391, 26.8),
            (10,250.349,  36.467, 47.0),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(filtered_data, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )

    def test_filter_data(self):
        """Tests for `filter_data()`."""
        E_min, E_max = (25, 45)
        coords = [
            CoordEquatorial(ra=299.868, dec=40.733),
            CoordEquatorial(ra=123.456, dec=-10.123),
            CoordEquatorial(ra=83.822,  dec=-5.391),
        ]
        filtered_data = filter_data(
            data=self.data,
            E_min=E_min,
            E_max=E_max,
            coords=coords,
        )

        target = np.rec.array([
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (8, 187.706,  12.391, 26.8),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(filtered_data, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )


    def test_catalog_flux_filter(self):
        """Tests for `flux_filter()`."""
        flux_range1 = (30, None)
        target1 = np.rec.array([
            ('SRC_C', 87.2, 143),
            ('SRC_E', 56.7, 87),
            ('SRC_G', 71.8, 77),
            ('SRC_H', 99.9, 160),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(flux_filter(self.catalog, *flux_range1), order="FLUX"),
            np.sort(target1, order="FLUX"),
        )

        flux_range2 = (None, 60)
        target2 = np.rec.array([
            ('SRC_A', 12.4, 120),
            ('SRC_B', 3.5, 98),
            ('SRC_D', 0.95, 65),
            ('SRC_E', 56.7, 87),
            ('SRC_F', 23.1, 132),
            ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(flux_filter(self.catalog, *flux_range2), order="FLUX"),
            np.sort(target2, order="FLUX"),
        )

        flux_range3 = (20, 80)
        target3 = np.rec.array([
            ('SRC_E', 56.7, 87),
            ('SRC_F', 23.1, 132),
            ('SRC_G', 71.8, 77),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(flux_filter(self.catalog, *flux_range3), order="FLUX"),
            np.sort(target3, order="FLUX"),
        )

    def test_catalog_sources_filter(self):
        """Tests for `source_filter()` on single run."""
        n = 3
        target = np.rec.array([
            ('SRC_C', 87.2, 143),
            ('SRC_F', 23.1, 132),
            ('SRC_H', 99.9, 160),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(source_filter(self.catalog, n), order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )

        n = (3, 6)
        target = np.rec.array([
            ('SRC_A', 12.4, 120),
            ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(source_filter(self.catalog, n), order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )

    def test_catalog_sources_filter2(self):
        """Tests for `source_filter()` on multiple runs."""
        n = 3
        target = np.rec.array([
            ('SRC_C', 87.2, 143),
            ('SRC_C', 87.2, 143),
            ('SRC_C', 87.2, 143),
            ('SRC_F', 23.1, 132),
            ('SRC_F', 23.1, 132),
            ('SRC_F', 23.1, 132),
            ('SRC_H', 99.9, 160),
            ('SRC_H', 99.9, 160),
            ('SRC_H', 99.9, 160),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(source_filter(self.catalog_mult_runs, n), order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )

        n = (3, 6)
        target = np.rec.array([
            ('SRC_A', 12.4, 120),
            ('SRC_A', 12.4, 120),
            ('SRC_A', 12.4, 120),
            ('SRC_I', 14.6, 101),
            ('SRC_I', 14.6, 101),
            ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110),
            ('SRC_J', 42.3, 110),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(source_filter(self.catalog_mult_runs, n), order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )

    def test_catalog_filter(self):
        """Test for `filter_catalog()`."""
        n = (3, 6)
        flux_range = (20, 90)

        # test for ValueError when both `n` and `flux_range` are given
        with self.assertRaises(ValueError):
            filter_catalog(self.catalog, n=n, flux_range=flux_range)
        
        # test for `n`
        filtered = filter_catalog(self.catalog, n=n)
        target = np.rec.array([
            ('SRC_A', 12.4, 120),
            ('SRC_I', 14.6, 101),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(filtered, order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )

        # test for `flux_range`
        filtered = filter_catalog(
            self.catalog, n=None, flux_range=flux_range,
        )
        target = np.rec.array([
            ('SRC_C', 87.2, 143),
            ('SRC_E', 56.7, 87),
            ('SRC_F', 23.1, 132),
            ('SRC_G', 71.8, 77),
            ('SRC_J', 42.3, 110),
        ], dtype=[('NAME', 'U10'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(filtered, order="NPHOTONS"),
            np.sort(target, order="NPHOTONS"),
        )



if __name__ == "__main__":
    unittest.main()


# end