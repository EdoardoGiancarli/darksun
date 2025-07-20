"""
Tests for IROS candidates parameters computing.
"""

from copy import deepcopy
import unittest
from unittest import TestCase

from bloodmoon.coords import shift2equatorial, shift2pos, shift2angle
from bloodmoon.mask import codedmask

from darksun.types import LogEntry
from darksun.analyze import compute_parameters
from darksun.data import create_log, get_data

from .assets import _path_test_SDL
from .assets import _path_test_mask


class TestComputeParameters(TestCase):
    """Tests for the `compute_parameters()` method in `analyze.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask)
        self.sdl = get_data(_path_test_SDL)
        self.sdl.header['EXPOSURE'] = 1e1

        self.shifts_x = [0.0] * 2
        self.shifts_y = [0.0] * 2
        self.fluences = [1e2] * 2
        self.snrs = [10.] * 2

        self.iros_log = create_log(
            params=(
                LogEntry('shift_x', 'D', 'mm'), LogEntry('shift_y', 'D', 'mm'),
                LogEntry('fluence', 'D', 'ph'), LogEntry('snr', 'D', ''),
            )
        )
        self.iros_log.add_entry_values('shift_x', self.shifts_x)
        self.iros_log.add_entry_values('shift_y', self.shifts_y)
        self.iros_log.add_entry_values('fluence', self.fluences)
        self.iros_log.add_entry_values('snr', self.snrs)

    def test_computing(self):
        """Tests if parameters are correctly computed."""
        log = compute_parameters(
            log=deepcopy(self.iros_log),
            camera=self.wfm,
            sdl=self.sdl,
        )

        self.assertEqual(
            shift2pos(self.wfm, self.shifts_x[0], self.shifts_y[0]),
            (log.log['y'][0], log.log['x'][0])
        )
        self.assertEqual(
            (shift2angle(self.wfm, self.shifts_x[0]), shift2angle(self.wfm, self.shifts_y[0])),
            (log.log['angle_x'][0], log.log['angle_y'][0])
        )
        self.assertEqual(
            shift2equatorial(self.sdl, self.wfm, self.shifts_x[0], self.shifts_y[0]),
            (log.log['ra'][0], log.log['dec'][0])
        )
        self.assertEqual(self.fluences[0] / 10, log.log['rate'][0])
    
    def test_angular_sensitivity(self):
        """Tests if cameras angular coords are correctly computed (in [deg])."""
        log = compute_parameters(
            log=deepcopy(self.iros_log),
            camera=self.wfm,
            sdl=self.sdl,
        )

        upx, upy = self.wfm.upscale_f
        self.assertEqual(
            (5 / upx / 60, 60 / upy / 60),
            (log.log['dangle_x'][0], log.log['dangle_y'][0])
        )

        upx, upy = (7, 3)
        wfm2 = codedmask(_path_test_mask, upx, upy)
        log = compute_parameters(
            log=deepcopy(self.iros_log),
            camera=wfm2,
            sdl=self.sdl,
        )
        self.assertEqual(
            (5 / upx / 60, 60 / upy / 60),
            (log.log['dangle_x'][0], log.log['dangle_y'][0])
        )




if __name__ == "__main__":
    unittest.main()


# end