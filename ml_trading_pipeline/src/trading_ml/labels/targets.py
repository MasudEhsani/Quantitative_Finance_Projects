"""Targets (what the models learn to predict).

forward_return(t) = close(t + h) / close(t) - 1     -> looks h days into the future.
The last h rows of every ticker therefore have NO label (NaN) and are dropped
from training. Because labels overlap in time, the walk-forward splitter purges
``h`` days between train and test (see validation/splits.py).

Label kinds
  direction      1 if forward return > 0 else 0                          (Line A)
  three_class    1 = buy (> +dead_zone), 0 = hold, -1 = sell (< -dead_zone)
  cs_outperform  1 if forward return > cross-sectional median that date (Line B)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_KINDS = ("direction", "three_class", "cs_outperform")


def forward_return(prices: pd.DataFrame, horizon: int) -> pd.Series:
    close = prices["close"]
    fut = close.groupby(level="ticker").shift(-horizon)
    out = fut / close - 1
    out.name = f"fwd_ret_{horizon}"
    return out


def make_labels(prices: pd.DataFrame, horizon: int, kind: str, dead_zone: float = 0.0) -> pd.Series:
    if kind not in LABEL_KINDS:
        raise ValueError(f"unknown label kind '{kind}', choose from {LABEL_KINDS}")
    fwd = forward_return(prices, horizon)
    if kind == "direction":
        y = (fwd > 0).astype(float)
    elif kind == "three_class":
        y = pd.Series(np.select([fwd > dead_zone, fwd < -dead_zone], [1.0, -1.0], 0.0), index=fwd.index)
    else:  # cs_outperform
        med = fwd.groupby(level="date").transform("median")
        y = (fwd > med).astype(float)
    y = y.where(fwd.notna())          # no future data -> no label
    y.name = "label"
    return y
