import unittest
from unittest import TestCase

import numpy as np

from bloodmoon.types import CoordEquatorial
from darksun.data import get_data, get_catalogue

from tests.assets import _path_test_SDL
from tests.assets import _path_test_catalogue

class TestDataLoader(TestCase):
    """Tests for DataLoader and photons event list access."""

    def test_data_loading(self):
        """Tests if data is correctly loaded."""
        sdl1 = get_data(_path_test_SDL)

        emin, emax = 20, 100
        coords = CoordEquatorial(ra=299.868, dec=40.733)

        sdl2 = get_data(
            filepath=_path_test_SDL,
            E_min=emin,
            E_max=None,
            coords=None,
        )
        sdl3 = get_data(
            filepath=_path_test_SDL,
            E_min=None,
            E_max=emax,
            coords=None,
        )
        sdl4 = get_data(
            filepath=_path_test_SDL,
            E_min=None,
            E_max=None,
            coords=coords,
        )
        sdl5 = get_data(
            filepath=_path_test_SDL,
            E_min=emin,
            E_max=emax,
            coords=coords,
        )

    def test_filter_allowed(self):
        """Tests if SDL data can be filtered."""
        emin, emax = 20, 100
        coords = CoordEquatorial(ra=299.868, dec=40.733)

        sdl = get_data(
            filepath=_path_test_SDL,
            E_min=emin,
            E_max=emax,
            coords=coords,
        )
        data = sdl.DLdata
    
    def test_filtering(self):
        """Tests if filters are correctly applied."""
        emin, emax = (25, 45)
        coords = [
            CoordEquatorial(ra=299.868, dec=40.733),
            CoordEquatorial(ra=123.456, dec=-10.123),
            CoordEquatorial(ra=83.822,  dec=-5.391),
        ]

        sdl = get_data(
            filepath=_path_test_SDL,
            E_min=emin,
            E_max=emax,
            coords=coords,
        )

        target = np.rec.array([
            (5,  53.125, -27.800, 29.5),
            (6,  13.158, -72.800, 44.1),
            (8, 187.706,  12.391, 26.8),
        ], dtype=[('ID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('ENERGY', 'f4')])

        np.testing.assert_array_equal(
            np.sort(sdl.DLdata, order="ENERGY"),
            np.sort(target, order="ENERGY"),
        )

        self.assertTrue(len(sdl.data) != len(sdl.DLdata))




class TestCatalogueLoader(TestCase):
    """Tests for CatalogueLoader and catalogue data access."""

    def test_data_loading(self):
        """Tests if data is correctly loaded."""
        catalogue1 = get_catalogue(_path_test_catalogue)

        n = 4
        fmin, fmax = 20, 100

        catalogue2 = get_catalogue(
            filepath=_path_test_catalogue,
            n=n,
            flux_range=None,
        )
        catalogue3 = get_catalogue(
            filepath=_path_test_catalogue,
            n=None,
            flux_range=(fmin, None),
        )
        catalogue4 = get_catalogue(
            filepath=_path_test_catalogue,
            n=None,
            flux_range=(None, fmax),
        )

        with self.assertRaises(ValueError):
            catalogue5 = get_catalogue(
                filepath=_path_test_catalogue,
                n=n,
                flux_range=(fmin, fmax),
            )
    
    def test_filter_allowed(self):
        """Tests if SDL data can be filtered."""
        # test filter for n executable
        n = 4
        sdl1 = get_catalogue(
            filepath=_path_test_catalogue,
            n=n,
            flux_range=None,
        )
        data = sdl1.DLdata

        # test filter for flux executable
        fmin, fmax = 20, 100
        sdl2 = get_catalogue(
            filepath=_path_test_catalogue,
            n=None,
            flux_range=(fmin, fmax),
        )
        data = sdl2.DLdata
    
    def test_filtering(self):
        """Tests if filters are correctly applied."""
        n = (3, 6)
        flux_range = (20, 90)

        # test for `n`
        sdl1 = get_catalogue(
            filepath=_path_test_catalogue,
            n=n,
            flux_range=None,
        )
        target1 = np.rec.array([
            ('SRC_A', 10.684,  41.269, 12.4, 120),
            ('SRC_I', 123.456, -10.123, 14.6, 101),
            ('SRC_J', 250.349, 36.467, 42.3, 110),
        ], dtype=[('NAME', 'S10'), ('RA', 'f8'), ('DEC', 'f8'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(sdl1.DLdata, order="NPHOTONS"),
            np.sort(target1, order="NPHOTONS"),
        )
        self.assertTrue(len(sdl1.data) != len(sdl1.DLdata))

        # test for `flux_range`
        sdl2 = get_catalogue(
            filepath=_path_test_catalogue,
            n=None,
            flux_range=flux_range,
        )
        target2 = np.rec.array([
            ('SRC_C', 201.365, -43.019, 87.2, 143),
            ('SRC_E', 53.125, -27.800, 56.7, 87),
            ('SRC_F', 13.158, -72.800, 23.1, 132),
            ('SRC_G', 299.868, 40.733, 71.8, 77),
            ('SRC_J', 250.349, 36.467, 42.3, 110),
        ], dtype=[('NAME', 'S10'), ('RA', 'f8'), ('DEC', 'f8'), ('FLUX', 'f8'), ('NPHOTONS', 'i4')])

        np.testing.assert_array_equal(
            np.sort(sdl2.DLdata, order="NPHOTONS"),
            np.sort(target2, order="NPHOTONS"),
        )
        self.assertTrue(len(sdl2.data) != len(sdl2.DLdata))




if __name__ == "__main__":
    unittest.main()


# end