"""Walk-forward validation with purging.

Timeline (expanding window, n_splits = 3):

    |------ train ------|purge|- test 1 -|
    |----------- train -----------|purge|- test 2 -|
    |------------------ train ------------------|purge|- test 3 -|holdout|

* Test blocks are consecutive and never overlap.
* ``purge_days`` dates right before each test block are dropped from training,
  because their labels (forward returns over ``horizon`` days) reach into the
  test period. purge_days >= horizon removes that leakage.
* ``holdout_days`` at the very end are never used by any fold (kept for the
  final check / paper-trading replay).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Fold:
    fold: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def describe(self) -> str:
        return (f"fold {self.fold}: train {self.train_dates.min().date()}..{self.train_dates.max().date()} "
                f"({len(self.train_dates)} d) | test {self.test_dates.min().date()}.."
                f"{self.test_dates.max().date()} ({len(self.test_dates)} d)")


class WalkForwardSplitter:
    def __init__(self, n_splits: int, test_days: int, min_train_days: int, purge_days: int,
                 holdout_days: int = 0, max_train_days: Optional[int] = None):
        if n_splits < 1 or test_days < 1:
            raise ValueError("n_splits and test_days must be >= 1")
        if purge_days < 0 or holdout_days < 0:
            raise ValueError("purge_days and holdout_days must be >= 0")
        self.n_splits = n_splits
        self.test_days = test_days
        self.min_train_days = min_train_days
        self.purge_days = purge_days
        self.holdout_days = holdout_days
        self.max_train_days = max_train_days     # None = expanding window, int = rolling window

    @classmethod
    def from_config(cls, vcfg: dict) -> "WalkForwardSplitter":
        return cls(vcfg["n_splits"], vcfg["test_days"], vcfg["min_train_days"], vcfg["purge_days"],
                   vcfg.get("holdout_days", 0), vcfg.get("max_train_days"))

    def usable_dates(self, dates) -> pd.DatetimeIndex:
        """All dates except the hold-out period."""
        d = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
        return d[: len(d) - self.holdout_days] if self.holdout_days else d

    def holdout_dates(self, dates) -> pd.DatetimeIndex:
        d = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
        return d[len(d) - self.holdout_days:] if self.holdout_days else d[:0]

    def split(self, dates) -> List[Fold]:
        d = self.usable_dates(dates)
        first_test = len(d) - self.n_splits * self.test_days
        needed = self.min_train_days + self.purge_days
        if first_test < needed:
            raise ValueError(
                f"not enough history: {len(d)} usable dates, but need {needed} (train+purge) + "
                f"{self.n_splits}x{self.test_days} (test). Reduce n_splits/test_days/min_train_days."
            )
        folds = []
        for k in range(self.n_splits):
            t0 = first_test + k * self.test_days
            train_end = t0 - self.purge_days                # exclusive
            train_start = 0 if self.max_train_days is None else max(0, train_end - self.max_train_days)
            folds.append(Fold(k, d[train_start:train_end], d[t0:t0 + self.test_days]))
        return folds
