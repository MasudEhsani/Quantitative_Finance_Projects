"""Feature engineering.

Rule: every feature at date t uses ONLY prices up to and including the close of t.
``tests/test_02_features.py`` enforces this with a truncation test: features for
dates <= T must be identical whether or not the data after T exists.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0).where(gain.notna())


def single_ticker_features(g: pd.DataFrame, fcfg: Dict[str, Any]) -> pd.DataFrame:
    """Features for ONE ticker. ``g`` is indexed by date with OHLCV columns."""
    close, high, low, volume = g["close"], g["high"], g["low"], g["volume"]
    logc = np.log(close)
    r1 = logc.diff()
    out = pd.DataFrame(index=g.index)

    for w in fcfg["return_windows"]:                       # momentum / returns
        out[f"ret_{w}"] = logc.diff(w)
    for w in fcfg["sma_windows"]:                          # trend
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1
    for w in fcfg["vol_windows"]:                          # realised volatility (annualised)
        out[f"vol_{w}"] = r1.rolling(w).std() * np.sqrt(252)

    out[f"rsi_{fcfg['rsi_window']}"] = _rsi(close, fcfg["rsi_window"]) / 100.0   # mean reversion

    w = fcfg["bb_window"]
    mid, sd = close.rolling(w).mean(), close.rolling(w).std()
    out[f"bb_pctb_{w}"] = (close - (mid - 2 * sd)) / (4 * sd)

    w = fcfg["atr_window"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out[f"atr_{w}_pct"] = tr.rolling(w).mean() / close

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    out.loc[out.index[:26], "macd_hist"] = np.nan             # warm-up period of the EMAs

    w = fcfg["volume_window"]
    lv = np.log1p(volume)
    out[f"volume_z_{w}"] = (lv - lv.rolling(w).mean()) / lv.rolling(w).std()
    out["hl_range"] = (high - low) / close

    if fcfg.get("calendar", True):
        out["day_of_week"] = g.index.dayofweek.astype(float)
        out["month"] = g.index.month.astype(float)
    return out


def build_features(prices: pd.DataFrame, fcfg: Dict[str, Any]) -> pd.DataFrame:
    """All per-ticker features + market-relative features. Same (date, ticker) index as prices."""
    parts = []
    for tkr, g in prices.groupby(level="ticker", sort=True):
        f = single_ticker_features(g.droplevel("ticker"), fcfg)
        f["ticker"] = tkr
        parts.append(f.reset_index().set_index(["date", "ticker"]))
    feats = pd.concat(parts).sort_index()

    # market-relative return: stock return minus the universe average on the same date
    for w in fcfg.get("relative_windows", []):
        col = f"ret_{w}"
        if col not in feats:
            continue
        mkt = feats[col].groupby(level="date").transform("mean")
        feats[f"rel_ret_{w}"] = feats[col] - mkt
    return feats.replace([np.inf, -np.inf], np.nan)


def add_cross_sectional_ranks(feats: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Percentile rank (0..1] of each feature across tickers on the same date (Line B)."""
    columns = columns or [c for c in feats.columns if c not in ("day_of_week", "month")]
    ranks = feats[columns].groupby(level="date").rank(pct=True)
    ranks.columns = [f"cs_rank_{c}" for c in columns]
    return pd.concat([feats, ranks], axis=1)


def feature_columns(feats: pd.DataFrame) -> List[str]:
    return list(feats.columns)
