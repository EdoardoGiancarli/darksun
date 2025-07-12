"""
Tests for data logging.
"""

import unittest
from unittest import TestCase

import pandas as pd

from darksun.types import LogEntry
from darksun.data import Log, create_log


class TestLogging(TestCase):
    """Tests for the Log structure in `data.py`."""

    def test_init(self):
        """Tests Log instance initialisation."""
        log = Log()
        self.assertIsNone(log.log)
        self.assertIsNone(log.params)
    
    def test_log_making(self):
        """Tests Log generation."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)

        expected = {
            "par1": [],
            "par2": [],
        }

        self.assertEqual(log.params, params)
        self.assertEqual(log.log, expected)
    
    def test_single_entry(self):
        """Tests if a single-entry Log is correctly generated."""
        params = LogEntry("par1", "J", "unit1")
        log = create_log(params)

        expected = {"par1": []}
        self.assertEqual(log.params, params)
        self.assertEqual(log.log, expected)
    
    def test_log_update(self):
        """Tests Log update procedure."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)

        run = {
            0: {"par1": 2, "par2": 8},
            1: {"par1": 5, "par2": 3},
            2: {"par1": 4, "par2": 9},
            3: {"par1": 1, "par2": 0},
        }

        expected = {
            "par1": [2, 5, 4, 1],
            "par2": [8, 3, 9, 0],
        }

        for it in range(4):
            values = tuple((entry, val) for entry, val in run[it].items())
            log.update(values)
        
        self.assertEqual(log.log, expected)
    
    def test_new_entries(self):
        """Tests if new entries are correctly added to the Log."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)

        new_entries = (
            LogEntry("par3", "J", "unit3"),
            LogEntry("par4", "J", "unit4"),
        )

        expected = {
            "par1": [], "par2": [],
            "par3": [], "par4": [],
        }

        log.insert(new_entries)
        self.assertEqual(log.params, params + new_entries)
        self.assertEqual(log.log, expected)
    
    def test_insert_single_entry(self):
        """Tests if a single new entry is correctly added."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)

        new_entries = (
            LogEntry("par3", "J", "unit3"),
        )

        expected = {
            "par1": [], "par2": [], "par3": [],
        }

        log.insert(new_entries)
        self.assertEqual(log.params, params + new_entries)
        self.assertEqual(log.log, expected)
    
    def test_add_values(self):
        """Tests if the sequence of values is added."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)
        log.add_entry_values('par1', [0] * 10)
        log.add_entry_values('par2', [0] * 10)
        self.assertEqual(log.log['par1'], [0] * 10)
        self.assertEqual(log.log['par2'], [0] * 10)

        log.add_entry_values('par1', [0] * 10)
        log.add_entry_values('par2', [0] * 10)
        self.assertEqual(log.log['par1'], [0] * 20)
        self.assertEqual(log.log['par2'], [0] * 20)

    def test_replace_values(self):
        """Tests if the sequence of values is replaced."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)
        log.replace_entry_values('par1', [0] * 10)
        log.replace_entry_values('par2', [0] * 10)
        self.assertEqual(log.log['par1'], [0] * 10)
        self.assertEqual(log.log['par2'], [0] * 10)

        log.replace_entry_values('par1', [1] * 10)
        log.replace_entry_values('par2', [1] * 10)
        self.assertEqual(log.log['par1'], [1] * 10)
        self.assertEqual(log.log['par2'], [1] * 10)

    def test_to_dataframe(self):
        """Tests if the Log is correctly converted to a DataFrame."""
        params = (
            LogEntry("par1", "J", "unit1"),
            LogEntry("par2", "J", "unit2"),
        )
        log = create_log(params)
        log.log['par1'] = [2, 5, 4, 1]
        log.log['par2'] = [8, 3, 9, 0]

        expected = pd.DataFrame(
            {
                "par1": [2, 5, 4, 1],
                "par2": [8, 3, 9, 0],
            }
        )

        df = log.to_dataframe()
        self.assertTrue(isinstance(log.log, dict))
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue((df['par1'] == expected['par1']).all())
        self.assertTrue((df['par2'] == expected['par2']).all())




if __name__ == "__main__":
    unittest.main()


# end