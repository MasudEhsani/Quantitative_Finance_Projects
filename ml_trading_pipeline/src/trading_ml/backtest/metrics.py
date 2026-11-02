"""Performance metrics from a daily return series."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def performance_metrics(returns: pd.Series, turnover: Optional[pd.Series] = None,
                        exposure: Optional[pd.Series] = None) -> Dict[str, float]:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"n_days": 0}
    total = float((1 + r).prod() - 1)
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    downside = float(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS))
    sortino = float(r.mean() * TRADING_DAYS / downside) if downside > 0 else 0.0
    mdd = max_drawdown(r)
    active = r[r != 0]
    m = {
        "n_days": n,
        "total_return": total,
        "cagr": float(cagr),
        "ann_vol": std * np.sqrt(TRADING_DAYS),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "hit_rate": float((active > 0).mean()) if len(active) else 0.0,
    }
    if turnover is not None:
        m["avg_daily_turnover"] = float(turnover.reindex(r.index).fillna(0).mean())
        m["n_trading_days"] = int((turnover.reindex(r.index).fillna(0) > 1e-12).sum())
    if exposure is not None:
        m["avg_exposure"] = float(exposure.reindex(r.index).fillna(0).mean())
    return m


def format_metrics(m: Dict[str, float]) -> str:
    keys = ["total_return", "cagr", "sharpe", "max_drawdown", "hit_rate", "avg_daily_turnover", "avg_exposure"]
    return "  ".join(f"{k}={m[k]:+.3f}" for k in keys if k in m)
