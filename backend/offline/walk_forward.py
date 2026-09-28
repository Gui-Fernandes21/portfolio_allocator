"""
Walk-forward fold construction following López de Prado (2018) paper.

Design choices:
-> Each fold trains on the most recent 'ROLLING_TRAIN_YEARS' years before its test window.
-> 4 test windows of one calendar year each: 2021, 2022, 2023, 2024.
-> The last 'EMBARGO_TRADING_DAYS' trading days before each test window are removed from that fold's training data, mitigating look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# FIXED SETTINGS
ROLLING_TRAIN_YEARS = 8
TEST_YEARS = [2021, 2022, 2023, 2024]
EMBARGO_TRADING_DAYS = 5


@dataclass(frozen=True)
class WalkForwardFold:
    """Date boundaries for one train/test fold."""

    fold_number: int
    train_start: str
    train_end: str  # after the embargo
    test_start: str
    test_end: str

    def describe(self) -> str:
        return (
            f"fold {self.fold_number}: "
            f"train {self.train_start}..{self.train_end} | "
            f"embargo {EMBARGO_TRADING_DAYS}d | "
            f"test {self.test_start}..{self.test_end}"
        )


def make_walk_forward_folds(
    trading_dates: list[str],
    test_years: list[int] = TEST_YEARS,
    rolling_train_years: int = ROLLING_TRAIN_YEARS,
    embargo_trading_days: int = EMBARGO_TRADING_DAYS,
) -> list[WalkForwardFold]:
    """
    Build folds from the dataset's trading calendar.

    'trading_dates' - sorted list of unique dates present in the dataset. The embargo is counted in trading days.
    """

    dates = sorted(trading_dates)
    date_index = pd.to_datetime(dates)
    folds = []

    for fold_number, test_year in enumerate(sorted(test_years), start=1):
        in_test_year = [
            date_string
            for date_string, timestamp in zip(dates, date_index)
            if timestamp.year == test_year
        ]
        
        if not in_test_year:
            raise ValueError(f"no trading dates found for test year {test_year}")
        
        test_start, test_end = in_test_year[0], in_test_year[-1]

        # All trading days before the test window
        candidate_train_dates = [d for d in dates if d < test_start]
        
        # Drop the last N trading days before the test (embargo).
        if embargo_trading_days > 0:
            candidate_train_dates = candidate_train_dates[:-embargo_trading_days]
        
        if not candidate_train_dates:
            raise ValueError(f"no training dates left for test year {test_year}")

        # Rolling window
        rolling_cutoff = (pd.Timestamp(test_start) - pd.DateOffset(years=rolling_train_years)).strftime("%Y-%m-%d")
        
        train_dates = [d for d in candidate_train_dates if d >= rolling_cutoff]
        
        if not train_dates:
            raise ValueError(
                f"rolling window of {rolling_train_years}y left no training dates for test year {test_year}"
            )

        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


if __name__ == "__main__":
    # Quick self-check against the cached dataset's real calendar.
    from data_cache import load_cached_dataset

    dataset = load_cached_dataset()
    for fold in make_walk_forward_folds(sorted(dataset.date.unique())):
        print(fold.describe())
