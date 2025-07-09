"""
Test for array cropping.
"""

import unittest
from unittest import TestCase

import numpy as np

from darksun.images import crop



class TestCropping(TestCase):
    """Test for the `crop()` method in `images.py`."""

    def test_errors(self):
        """Test for input values."""
        n, m = 20, 20
        pos = (5, 5)
        cr = 10
        a = np.zeros((n, m))

        with self.assertRaises(ValueError):
            crop(a, pos, (-cr, cr))
            crop(a, pos, (cr, -cr))
            crop(a, pos, (-cr, -cr))
        
        with self.assertRaises(IndexError):
            crop(a, (n // 4, m // 4), (cr, cr))
            crop(a, (n // 4, -m // 4), (cr, cr))
            crop(a, (-n // 4, m // 4), (cr, cr))
            crop(a, (-n // 4, -m // 4), (cr, cr))
    
    def test_strict_cropping(self):
        """Test for strict cropping."""
        n, m = 20, 20
        cr = 2
        a = np.zeros((n, m))
        target_shape = (2 * cr + 1, 2 * cr + 1)

        self.assertEqual(
            crop(a, (n // 4, m // 4), (cr, cr)).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (n // 4, -m // 4), (cr, cr)).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, m // 4), (cr, cr)).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, -m // 4), (cr, cr)).shape,
            target_shape,
        )
    
    def test_adaptable_cropping(self):
        """Test for adaptable cropping."""
        n, m = 20, 20
        cr = 10
        a = np.zeros((n, m))
        target_shape = (7, 7)

        self.assertEqual(
            crop(a, (n // 4, m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (n // 4, -m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, -m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )




if __name__ == "__main__":
    unittest.main()


# end