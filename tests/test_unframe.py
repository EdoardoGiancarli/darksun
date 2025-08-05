"""
Tests for arrays unframing.
"""

import unittest
from unittest import TestCase

import numpy as np
from numpy.typing import NDArray

from darksun.images import unframe

def array_isequal(x: NDArray, y: NDArray) -> None:
    np.testing.assert_array_equal(x, y)

class TestSampling(TestCase):
    """Test for the `unframe()` method in `images.py`."""

    def test_unframe(self):
        """Test array unframing."""
        a = np.random.randint(0, 10, (10, 10))

        array_isequal(
            unframe(a), a,
        )
        array_isequal(
            unframe(a, (None, None)), a,
        )
        array_isequal(
            unframe(a, unframe_y=2), a[2:-2, :],
        )
        array_isequal(
            unframe(a, unframe_y=2, unframe_x=3), a[2:-2, 3:-3],
        )
        array_isequal(
            unframe(a, unframe_y=(2, None)), a[2:, :],
        )
        array_isequal(
            unframe(a, unframe_y=(None, -2)), a[:-2, :],
        )

    def test_exceptions(self):
        """Test unframe exceptions."""
        a = np.random.randint(0, 10, (10, 10))
        with self.assertRaises(TypeError):
            unframe(a, unframe_y=2.0)
            unframe(a, unframe_y=(2.0, 5.0))
            unframe(a, unframe_x=2.0)
            unframe(a, unframe_x=(2.0, 5.0))
        



if __name__ == "__main__":
    unittest.main()


# end