"""From model scores to target portfolio weights.

Line A (single stock) - stateful threshold rule per ticker:
    score > buy_threshold   -> BUY  (+1)  -> go / stay long
    score < sell_threshold  -> SELL (-1)  -> go flat (or short if allow_short)
    otherwise               -> HOLD ( 0)  -> keep the previous position
Several Line-A tickers in one portfolio share capital equally (1/N each).

Line B (set of stocks) - cross-sectional ranking:
    every ``rebalance_every`` days: long the top_k scores (equal weight, sum = 1);
    with bottom_k > 0 also short the bottom_k -> dollar-neutral (+0.5 / -0.5).
    Between rebalance days the weights are held.

All weights are TARGETS decided at the close of their date; the backtest
executes them at the next open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUY, HOLD, SELL = 1, 0, -1


def scores_to_signals(scores: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:
    if sell_threshold > buy_threshold:
        raise ValueError("sell_threshold must be <= buy_threshold")
    sig = np.select([scores > buy_threshold, scores < sell_threshold], [BUY, SELL], HOLD)
    return pd.Series(sig, index=scores.index, name="signal").where(scores.notna(), HOLD).astype(int)


def signals_to_positions(signals: pd.Series, allow_short: bool = False, start_position: float = 0.0) -> pd.Series:
    """Stateful: BUY -> +1, SELL -> 0 (or -1), HOLD -> previous position. One ticker, date index."""
    out = np.empty(len(signals))
    pos = start_position
    for i, s in enumerate(signals.to_numpy()):
        if s == BUY:
            pos = 1.0
        elif s == SELL:
            pos = -1.0 if allow_short else 0.0
        out[i] = pos
    return pd.Series(out, index=signals.index, name="position")


def line_a_weights(scores: pd.Series, buy_threshold: float = 0.55, sell_threshold: float = 0.45,
                   allow_short: bool = False) -> pd.DataFrame:
    """scores: Series indexed (date, ticker) -> wide target weights (date x ticker)."""
    sig = scores_to_signals(scores, buy_threshold, sell_threshold)
    wide_sig = sig.unstack("ticker").sort_index()
    n = wide_sig.shape[1]
    pos = {}
    for tkr in wide_sig.columns:
        s = wide_sig[tkr].dropna().astype(int)
        pos[tkr] = signals_to_positions(s, allow_short)
    return (pd.DataFrame(pos).reindex(wide_sig.index).ffill().fillna(0.0) / n).sort_index()


def line_b_weights(scores: pd.Series, top_k: int = 3, bottom_k: int = 0, rebalance_every: int = 5) -> pd.DataFrame:
    """scores: Series indexed (date, ticker) -> wide target weights (date x ticker)."""
    if top_k < 1 or bottom_k < 0 or rebalance_every < 1:
        raise ValueError("need top_k >= 1, bottom_k >= 0, rebalance_every >= 1")
    wide = scores.unstack("ticker").sort_index()
    weights = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    long_w, short_w = (0.5, -0.5) if bottom_k > 0 else (1.0, 0.0)
    for i, d in enumerate(wide.index):
        if i % rebalance_every:
            continue
        row = wide.loc[d].dropna().sort_index()
        if row.empty:
            continue
        ranked = row.sort_values(ascending=False, kind="mergesort")   # stable -> deterministic ties
        k_long = min(top_k, len(ranked))
        w = pd.Series(0.0, index=wide.columns)
        w[ranked.index[:k_long]] = long_w / k_long
        if bottom_k > 0:
            k_short = min(bottom_k, len(ranked) - k_long)
            if k_short > 0:
                w[ranked.index[-k_short:]] = short_w / k_short
        weights.loc[d] = w
    return weights.ffill().fillna(0.0)


def equal_weight(dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Benchmark: buy and hold an equal-weight basket (rebalanced to equal weight daily)."""
    tickers = list(tickers)
    return pd.DataFrame(1.0 / len(tickers), index=pd.DatetimeIndex(dates), columns=tickers)
