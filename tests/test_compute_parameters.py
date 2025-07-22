"""
Tests for IROS candidates parameters computing.
"""

from copy import deepcopy
import unittest
from unittest import TestCase

import numpy as np

from bloodmoon.coords import shift2equatorial, shift2pos, shift2angle, angle2shift
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
        self.dfluences = [1e1] * 2
        self.snrs = [10.] * 2

        upx, upy = self.wfm.upscale_f
        self.dshifts_x = [angle2shift(self.wfm, 5.0 / upx / 60)] * 2
        self.dshifts_y = [angle2shift(self.wfm, 60.0 / upy / 60)] * 2

        self.iros_log = create_log(
            params=(
                LogEntry('shift_x', 'D', 'mm'), LogEntry('dshift_x', 'D', 'mm'),
                LogEntry('shift_y', 'D', 'mm'), LogEntry('dshift_y', 'D', 'mm'),
                LogEntry('fluence', 'D', 'ph'), LogEntry('dfluence', 'D', 'ph'),
                LogEntry('snr', 'D', ''),
            )
        )
        self.iros_log.add_entry_values('shift_x', self.shifts_x)
        self.iros_log.add_entry_values('dshift_x', self.dshifts_x)
        self.iros_log.add_entry_values('shift_y', self.shifts_y)
        self.iros_log.add_entry_values('dshift_y', self.dshifts_y)
        self.iros_log.add_entry_values('fluence', self.fluences)
        self.iros_log.add_entry_values('dfluence', self.dfluences)
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
        np.testing.assert_almost_equal(
            np.array((5 / upx / 60, 60 / upy / 60)),
            np.array((log.log['dangle_x'][0], log.log['dangle_y'][0])),
            decimal=5,
        )




if __name__ == "__main__":
    unittest.main()


# end