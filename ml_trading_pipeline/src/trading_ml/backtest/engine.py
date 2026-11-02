"""Vectorised daily backtest.

Timing (the most common source of fake profits, so it is explicit):
  * target weight w[t] is decided at the CLOSE of day t (using data <= t)
  * it is executed at the OPEN of day t+1 -> held weight h[t+1] = w[t]
  * held weight h[d] earns the open-to-open return of day d:
        r_oo[d] = open[d+1] / open[d] - 1
  * trading costs are paid at the open of day d on the turnover
        turnover[d] = sum_i |h[d, i] - h[d-1, i]|,   cost[d] = turnover[d] * cost_bps / 1e4

Weights are held constant between decisions (daily rebalancing back to target).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from ..data.schema import to_wide
from .metrics import performance_metrics


@dataclass
class BacktestResult:
    returns: pd.Series            # net daily returns
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    held_weights: pd.DataFrame
    equity: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics)


def run_backtest(target_weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 10.0,
                 initial_capital: float = 100_000.0) -> BacktestResult:
    if target_weights.empty:
        raise ValueError("target_weights is empty")
    opens = to_wide(prices, "open")
    unknown = set(target_weights.columns) - set(opens.columns)
    if unknown:
        raise ValueError(f"weights for tickers without prices: {sorted(unknown)}")
    opens = opens[list(target_weights.columns)]

    all_dates = opens.index
    start, end = target_weights.index.min(), target_weights.index.max()
    later = all_dates[all_dates > end]
    last = later[0] if len(later) else end              # the day the final decision is executed
    dates = all_dates[(all_dates >= start) & (all_dates <= last)]

    target = target_weights.reindex(dates).ffill().fillna(0.0)
    held = target.shift(1).fillna(0.0)                  # executed next open
    r_oo = (opens.shift(-1) / opens - 1).reindex(dates)

    gross = (held * r_oo.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    costs = turnover * cost_bps / 1e4
    net = gross - costs

    valid = r_oo.notna().any(axis=1)                    # drop the final day with no next open
    net, gross, costs, turnover, held = net[valid], gross[valid], costs[valid], turnover[valid], held[valid]

    equity = initial_capital * (1 + net).cumprod()
    exposure = held.abs().sum(axis=1)
    metrics = performance_metrics(net, turnover=turnover, exposure=exposure)
    return BacktestResult(net.rename("net"), gross.rename("gross"), costs.rename("costs"),
                          turnover.rename("turnover"), held, equity.rename("equity"), metrics)
