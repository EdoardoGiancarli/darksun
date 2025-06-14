"""
Tests for data logging.
"""

import unittest
from unittest import TestCase

from darksun.types import LogEntry
from darksun.data import Log, create_log


class TestLogging(TestCase):
    """Tests for the Log structure in `data.py`."""

    def setUp(self):
        self.camA = "cam1a"
        self.camB = "cam1b"

    def test_init(self):
        """Tests Log instance initialisation."""
        log = Log(camA_ID=self.camA, camB_ID=self.camB)
        self.assertIsNone(log.log)
        self.assertIsNone(log.params)
    
    def test_log_making(self):
        """Tests Log generation."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(
            camA_ID=self.camA, 
            camB_ID=self.camB,
            params=params,
        )

        expected = {
            self.camA: {
                "par1": {"data": [], "format": "J", "unit": "unit1"},
                "par2": {"data": [], "format": "J", "unit": "unit2"},
            },
            self.camB: {
                "par1": {"data": [], "format": "J", "unit": "unit1"},
                "par2": {"data": [], "format": "J", "unit": "unit2"},
            },
        }

        self.assertEqual(log.log, expected)

        with self.assertRaises(ValueError):
            create_log(
                camA_ID=1,
                camB_ID=self.camB,
                params=params,
            )
            create_log(
                camA_ID=self.camA,
                camB_ID=6.3,
                params=params,
            )
            create_log(
                camA_ID=1,
                camB_ID=5,
                params=params,
            )
    
    def test_log_update(self):
        """
        Tests Log update:
            - update procedure
            - independence of camera logs
        """
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(
            camA_ID=self.camA,
            camB_ID=self.camB,
            params=params,
        )

        run = {
            0: {
                self.camA: {"par1": 2, "par2": 8},
                self.camB: {"par1": 6, "par2": 9},
            },
            1: {
                self.camA: {"par1": 5, "par2": 3},
                self.camB: {"par1": 1, "par2": 0},
            },
            2: {
                self.camA: {"par1": 4, "par2": 9},
                self.camB: {"par1": 2, "par2": 5},
            },
            3: {
                self.camA: {"par1": 1, "par2": 0},
                self.camB: {"par1": 3, "par2": 7},
            },
        }

        checkpoint = {
            self.camA: {
                "par1": {"data": [2, 5, 4, 1], "format": "J", "unit": "unit1"},
                "par2": {"data": [8, 3, 9, 0], "format": "J", "unit": "unit2"},
            },
            self.camB: {
                "par1": {"data": [], "format": "J", "unit": "unit1"},
                "par2": {"data": [], "format": "J", "unit": "unit2"},
            },
        }

        expected = {
            self.camA: {
                "par1": {"data": [2, 5, 4, 1], "format": "J", "unit": "unit1"},
                "par2": {"data": [8, 3, 9, 0], "format": "J", "unit": "unit2"},
            },
            self.camB: {
                "par1": {"data": [6, 1, 2, 3], "format": "J", "unit": "unit1"},
                "par2": {"data": [9, 0, 5, 7], "format": "J", "unit": "unit2"},
            },
        }

        # update CAM1A and check
        for it in range(4):
            values = tuple((entry, val) for entry, val in run[it][self.camA].items())
            log.update(self.camA, values)
        
        self.assertEqual(log.log, checkpoint)

        # update CAM1B and check final Log
        for it in range(4):
            values = tuple((entry, val) for entry, val in run[it][self.camB].items())
            log.update(self.camB, values)
        
        self.assertEqual(log.log, expected)


if __name__ == "__main__":
    unittest.main()


# end