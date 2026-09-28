from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from walk_forward import (
    EMBARGO_TRADING_DAYS,
    ROLLING_TRAIN_YEARS,
    TEST_YEARS,
    make_walk_forward_folds,
)

# A business-day calendar
def synthetic_calendar(start="2012-01-03", end="2024-12-31"):
    return pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()


class TestFoldConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = synthetic_calendar()
        cls.folds = make_walk_forward_folds(cls.calendar)

    def test_one_fold_per_test_year(self):
        self.assertEqual(len(self.folds), len(TEST_YEARS))
        self.assertEqual([f.fold_number for f in self.folds], [1, 2, 3, 4])

    def test_test_windows_are_single_calendar_years(self):
        for fold, year in zip(self.folds, sorted(TEST_YEARS)):
            self.assertEqual(fold.test_start[:4], str(year))
            self.assertEqual(fold.test_end[:4], str(year))

    def test_embargo_is_exactly_five_trading_days(self):
        for fold in self.folds:
            gap = [d for d in self.calendar if fold.train_end < d < fold.test_start]
            self.assertEqual(len(gap), EMBARGO_TRADING_DAYS, f"fold {fold.fold_number}: gap {len(gap)} days")

    def test_no_train_test_overlap(self):
        for fold in self.folds:
            self.assertLess(fold.train_end, fold.test_start)

    def test_rolling_window_spans_eight_years(self):
        for fold in self.folds:
            span_years = (
                pd.Timestamp(fold.train_end) - pd.Timestamp(fold.train_start)
            ).days / 365.25
            self.assertGreater(span_years, ROLLING_TRAIN_YEARS - 0.25)
            self.assertLessEqual(span_years, ROLLING_TRAIN_YEARS + 0.05)

    def test_folds_shift_forward_one_year_each(self):
        train_start_years = [int(f.train_start[:4]) for f in self.folds]
        self.assertEqual(train_start_years,[train_start_years[0] + i for i in range(4)])


class TestFoldEdgeCases(unittest.TestCase):
    def test_zero_embargo_leaves_no_gap(self):
        calendar = synthetic_calendar()
        folds = make_walk_forward_folds(calendar, embargo_trading_days=0)
        for fold in folds:
            gap = [d for d in calendar if fold.train_end < d < fold.test_start]
            self.assertEqual(len(gap), 0)

    def test_missing_test_year_raises(self):
        calendar = synthetic_calendar(end="2022-12-30")  # no 2023/2024 data
        with self.assertRaises(ValueError):
            make_walk_forward_folds(calendar, test_years=[2023])

    def test_unsorted_input_calendar_is_handled(self):
        calendar = synthetic_calendar()
        shuffled = list(reversed(calendar))
        self.assertEqual(make_walk_forward_folds(shuffled),make_walk_forward_folds(calendar))


if __name__ == "__main__":
    unittest.main()
