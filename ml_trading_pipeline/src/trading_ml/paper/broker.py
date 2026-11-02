"""A simulated ("paper") broker with a persistent JSON ledger.

Daily cycle, same timing as the backtest:
    evening of day t : queue_target_weights(t, {...})      <- decision at the close of t
    day t+1          : process_day(t+1, opens, closes)
                         1. orders fill at the OPEN of t+1 (fractional shares, costs in bps)
                         2. the portfolio is marked to market at the CLOSE of t+1
Between orders the share counts stay fixed (weights drift with prices).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class PaperBroker:
    def __init__(self, name: str, initial_cash: float = 100_000.0, cost_bps: float = 10.0,
                 meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.initial_cash = float(initial_cash)
        self.cost_bps = float(cost_bps)
        self.cash = float(initial_cash)
        self.positions: Dict[str, float] = {}          # ticker -> shares
        self.pending: Optional[Dict[str, Any]] = None   # {"decision_date": str, "weights": {...}}
        self.history: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.last_date: Optional[str] = None
        self.meta: Dict[str, Any] = dict(meta or {})

    # ------------------------------------------------------------------ orders
    def queue_target_weights(self, decision_date, weights: Dict[str, float]) -> None:
        gross = sum(abs(w) for w in weights.values())
        if gross > 1.0 + 1e-9:
            raise ValueError(f"{self.name}: gross exposure {gross:.3f} > 1 (no leverage in paper trading)")
        self.pending = {"decision_date": str(pd.Timestamp(decision_date).date()),
                        "weights": {k: float(v) for k, v in weights.items()}}

    def equity_at(self, prices: Dict[str, float]) -> float:
        return self.cash + sum(sh * prices[t] for t, sh in self.positions.items() if t in prices)

    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity_at(prices)
        return {t: sh * prices[t] / eq for t, sh in self.positions.items() if t in prices and eq > 0}

    def _execute(self, date: str, opens: Dict[str, float]) -> List[Dict[str, Any]]:
        targets = self.pending["weights"]
        tradable = {t for t in set(targets) | set(self.positions) if t in opens}
        equity_open = self.equity_at(opens)
        # reserve room for costs, so a 100% target never pushes cash below zero
        investable = equity_open / (1 + self.cost_bps / 1e4)
        fills = []
        for t in sorted(tradable):
            target_sh = targets.get(t, 0.0) * investable / opens[t]
            delta = target_sh - self.positions.get(t, 0.0)
            if abs(delta * opens[t]) < 1e-6:
                continue
            value = delta * opens[t]
            cost = abs(value) * self.cost_bps / 1e4
            self.cash -= value + cost
            new_sh = self.positions.get(t, 0.0) + delta
            if abs(new_sh) < 1e-12:
                self.positions.pop(t, None)
            else:
                self.positions[t] = new_sh
            fills.append({"date": date, "decision_date": self.pending["decision_date"], "ticker": t,
                          "side": "BUY" if delta > 0 else "SELL", "shares": delta, "price": opens[t],
                          "value": value, "cost": cost})
        skipped = sorted(set(targets) - set(opens))
        if skipped:
            print(f"[paper] {self.name}: no open price on {date} for {skipped} -> not traded")
        self.pending = None
        self.trades.extend(fills)
        return fills

    # ------------------------------------------------------------------ daily step
    def process_day(self, date, opens: Dict[str, float], closes: Dict[str, float]) -> Dict[str, Any]:
        d = str(pd.Timestamp(date).date())
        if self.last_date is not None and d <= self.last_date:
            raise ValueError(f"{self.name}: day {d} already processed (last {self.last_date})")
        equity_open = self.equity_at(opens)
        fills = []
        if self.pending is not None and self.pending["decision_date"] < d:
            fills = self._execute(d, opens)
        equity = self.equity_at(closes)
        prev = self.history[-1]["equity"] if self.history else self.initial_cash
        gross_exposure = sum(abs(sh * closes[t]) for t, sh in self.positions.items() if t in closes)
        row = {"date": d, "equity": equity, "equity_open": equity_open, "cash": self.cash,
               "daily_return": equity / prev - 1, "total_return": equity / self.initial_cash - 1,
               "exposure": gross_exposure / equity if equity else 0.0, "n_positions": len(self.positions),
               "n_fills": len(fills), "costs": sum(f["cost"] for f in fills)}
        self.history.append(row)
        self.last_date = d
        return row

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "initial_cash", "cost_bps", "cash", "positions", "pending",
                                              "history", "trades", "last_date", "meta")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperBroker":
        b = cls(d["name"], d["initial_cash"], d["cost_bps"], d.get("meta"))
        for k in ("cash", "positions", "pending", "history", "trades", "last_date"):
            setattr(b, k, d[k])
        return b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        tmp.replace(path)                       # atomic: never leaves a half-written ledger

    @classmethod
    def load(cls, path: Path) -> "PaperBroker":
        return cls.from_dict(json.loads(path.read_text()))

    def history_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
