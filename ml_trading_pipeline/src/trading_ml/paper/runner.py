"""Daily paper trading of many strategies side by side.

A *strategy* = one trained model from a training session (Line A per ticker, or
Line B universe) or a benchmark (buy & hold / equal weight). Each strategy has
its own PaperBroker ledger in ``paper.state_dir``.

For each new trading day d, ``PaperTrader.step(d)``:
  1. every broker fills yesterday's queued orders at d's open, marks to d's close
  2. features are computed from prices up to d (never later)
  3. every model scores d -> target weights -> queued for d+1's open
  4. a daily report row per strategy is written (+ metrics to the tracker)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest import performance_metrics
from ..config import resolve_path
from ..features import add_cross_sectional_ranks, build_features
from ..models import BaseModel
from ..signals import BUY, SELL, line_b_weights, scores_to_signals
from ..tracking import BaseTracker, get_tracker
from .broker import PaperBroker


@dataclass
class Strategy:
    name: str
    line: str                     # "A" or "B"
    target: str                   # ticker (A) or "universe" (B)
    model_name: str
    model: Optional[BaseModel]    # None -> benchmark
    run_id: Optional[str] = None  # training run it came from

    @property
    def is_benchmark(self) -> bool:
        return self.model is None


# ----------------------------------------------------------------------------- strategies
def benchmark_strategies(line_a_targets: List[str], include_b: bool = True) -> List[Strategy]:
    out = [Strategy(f"A|{t}|buy_and_hold", "A", t, "buy_and_hold", None) for t in line_a_targets]
    if include_b:
        out.append(Strategy("B|universe|equal_weight", "B", "universe", "equal_weight", None))
    return out


def latest_session(trackers: List[BaseTracker]) -> Optional[str]:
    best: Tuple[Optional[pd.Timestamp], Optional[str]] = (None, None)
    for t in trackers:
        runs = t.search_runs()
        if len(runs) and "tags.session_id" in runs:
            r = runs.sort_values("start_time").iloc[-1]
            ts = pd.Timestamp(r["start_time"])
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            if best[0] is None or ts > best[0]:
                best = (ts, r["tags.session_id"])
    return best[1]


def load_session_strategies(cfg: Dict[str, Any], session_id: Optional[str] = None, verbose: bool = True
                            ) -> Tuple[str, List[Strategy]]:
    """All models of a training session (+ benchmarks), loaded from the tracker."""
    trackers = {line: get_tracker(cfg, f"line{line}", verbose=False) for line in ("A", "B")}
    session_id = session_id or latest_session(list(trackers.values()))
    if session_id is None:
        raise RuntimeError("no training session found - run scripts/train.py first")
    strategies: List[Strategy] = []
    for line, tr in trackers.items():
        runs = tr.search_runs({"session_id": session_id})
        for _, r in runs.sort_values("run_name").iterrows() if len(runs) else []:
            target, model_name = r["tags.target"], r["tags.model"]
            strategies.append(Strategy(f"{line}|{target}|{model_name}", line, target, model_name,
                                       tr.load_model(r["run_id"]), r["run_id"]))
    a_targets = sorted({s.target for s in strategies if s.line == "A"})
    has_b = any(s.line == "B" for s in strategies)
    strategies += benchmark_strategies(a_targets, include_b=has_b)
    if verbose:
        print(f"[paper] session {session_id}: {len(strategies)} strategies "
              f"({sum(not s.is_benchmark for s in strategies)} models + benchmarks)")
    return session_id, strategies


# ----------------------------------------------------------------------------- trader
class PaperTrader:
    def __init__(self, cfg: Dict[str, Any], strategies: List[Strategy], state_dir: Optional[str] = None,
                 track: bool = True, verbose: bool = True):
        self.cfg = cfg
        self.pcfg = cfg["paper"]
        self.strategies = strategies
        self.state_dir = resolve_path(state_dir or self.pcfg["state_dir"])
        self.verbose = verbose
        self.lookback = int(self.pcfg.get("lookback_days", 400))
        self.tracker = get_tracker(cfg, "paper", verbose=verbose) if track else None
        self.brokers: Dict[str, PaperBroker] = {}
        for s in strategies:
            p = self._path(s.name)
            if p.exists():
                # existing ledger continues; after a retrain the strategy simply trades the newer model
                b = PaperBroker.load(p)
                if s.run_id and b.meta.get("train_run_id") != s.run_id:
                    b.meta.setdefault("model_history", []).append(
                        {"from_date": b.last_date, "train_run_id": s.run_id})
                    b.meta["train_run_id"] = s.run_id
                self.brokers[s.name] = b
            else:
                self.brokers[s.name] = PaperBroker(s.name, self.pcfg["initial_cash"], self.pcfg["cost_bps"],
                                                   meta={"line": s.line, "target": s.target, "model": s.model_name,
                                                         "train_run_id": s.run_id, "n_decisions": 0})
        self._warned_in_sample = set()

    def _path(self, name: str) -> Path:
        return self.state_dir / (name.replace("|", "__") + ".json")

    def reset(self) -> None:
        for s in self.strategies:
            self._path(s.name).unlink(missing_ok=True)
            self.brokers[s.name] = PaperBroker(s.name, self.pcfg["initial_cash"], self.pcfg["cost_bps"],
                                               meta={"line": s.line, "target": s.target, "model": s.model_name,
                                                     "train_run_id": s.run_id, "n_decisions": 0})
        (self.state_dir / "daily_report.csv").unlink(missing_ok=True)

    # ------------------------------------------------------------------ features
    def _features_at(self, prices: pd.DataFrame, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        dates = prices.index.get_level_values("date")
        window_dates = dates[dates <= date].unique()[-self.lookback:]
        hist = prices[dates.isin(window_dates)]
        feats = build_features(hist, self.cfg["features"])
        out = {"A": feats.xs(date, level="date", drop_level=False)}
        if any(s.line == "B" for s in self.strategies):
            out["B"] = add_cross_sectional_ranks(feats).xs(date, level="date", drop_level=False)
        return out

    # ------------------------------------------------------------------ decisions
    def _decide(self, s: Strategy, broker: PaperBroker, X: Dict[str, pd.DataFrame], date: pd.Timestamp,
                closes: Dict[str, float], tickers: List[str]) -> Tuple[Optional[Dict[str, float]], str, float]:
        """-> (target weights or None = no order, action text, score)"""
        if s.is_benchmark:
            if broker.positions:                      # already invested: buy & hold, no re-orders
                return None, "HOLD", np.nan
            if s.line == "A":
                return ({s.target: 1.0}, "BUY", np.nan) if s.target in closes else (None, "NO DATA", np.nan)
            avail = [t for t in tickers if t in closes]
            return {t: 1.0 / len(avail) for t in avail}, "BUY", np.nan

        train_end = getattr(s.model, "train_end_", None)
        if train_end is not None and date <= pd.Timestamp(train_end):
            if s.name not in self._warned_in_sample:
                print(f"[paper] WARNING {s.name}: {date.date()} is inside the model's training period "
                      f"(trained until {pd.Timestamp(train_end).date()}) -> results are in-sample")
                self._warned_in_sample.add(s.name)

        Xl = X[s.line]
        if s.line == "A":
            row = Xl[Xl.index.get_level_values("ticker") == s.target].dropna()
            if row.empty:
                return None, "NO DATA", np.nan
            score = float(s.model.predict_score(row)[0])
            a = self.cfg["line_a"]
            sig = int(scores_to_signals(pd.Series([score]), a["buy_threshold"], a["sell_threshold"]).iloc[0])
            cur = broker.current_weights(closes).get(s.target, 0.0)
            pos = 1.0 if cur > 1e-9 else (-1.0 if cur < -1e-9 else 0.0)
            if sig == BUY:
                new = 1.0
            elif sig == SELL:
                new = -1.0 if a["allow_short"] else 0.0
            else:
                return None, "HOLD", score
            if new == pos:
                return None, "HOLD", score
            return {s.target: new}, ("BUY" if new > pos else "SELL"), score

        # Line B
        b = self.cfg["line_b"]
        n = broker.meta.get("n_decisions", 0)
        broker.meta["n_decisions"] = n + 1
        if n % b["rebalance_every"]:
            return None, "HOLD", np.nan
        rows = Xl.dropna()
        if rows.empty:
            return None, "NO DATA", np.nan
        scores = pd.Series(s.model.predict_score(rows), index=rows.index)
        w = line_b_weights(scores, b["top_k"], b["bottom_k"], rebalance_every=1).iloc[0]
        return {t: float(v) for t, v in w.items() if v != 0}, "REBALANCE", float(scores.max())

    # ------------------------------------------------------------------ one day
    def step(self, date, prices: pd.DataFrame) -> pd.DataFrame:
        date = pd.Timestamp(date)
        bar = prices.xs(date, level="date")
        opens, closes = bar["open"].to_dict(), bar["close"].to_dict()
        tickers = sorted(bar.index)
        X = None
        rows = []
        for s in self.strategies:
            broker = self.brokers[s.name]
            if broker.last_date is not None and str(date.date()) <= broker.last_date:
                continue                                          # already processed (catch-up safety)
            day = broker.process_day(date, opens, closes)
            if X is None and not s.is_benchmark:
                X = self._features_at(prices, date)
            weights, action, score = self._decide(s, broker, X or {}, date, closes, tickers)
            if weights is not None:
                broker.queue_target_weights(date, weights)
            rows.append({"date": date.date(), "strategy": s.name, "line": s.line, "target": s.target,
                         "model": s.model_name, "equity": round(day["equity"], 2),
                         "daily_return": day["daily_return"], "total_return": day["total_return"],
                         "exposure": day["exposure"], "fills_today": day["n_fills"], "score": score,
                         "action_for_tomorrow": action,
                         "target_weights": "" if weights is None else
                         ", ".join(f"{t}:{w:.2f}" for t, w in sorted(weights.items()))})
            broker.save(self._path(s.name))
            self._track(s, broker, day)
        report = pd.DataFrame(rows)
        if len(report):
            p = self.state_dir / "daily_report.csv"
            report.to_csv(p, mode="a", header=not p.exists(), index=False)
        return report

    def _track(self, s: Strategy, broker: PaperBroker, day: Dict[str, Any]) -> None:
        if self.tracker is None:
            return
        tags = {"strategy": s.name, "line": s.line, "target": s.target, "model": s.model_name,
                "train_run_id": s.run_id or "benchmark"}
        with self.tracker.start_run(s.name, tags=tags, run_id=broker.meta.get("paper_run_id")) as t:
            broker.meta["paper_run_id"] = t.run_id
            t.log_metrics({k: day[k] for k in ("equity", "daily_return", "total_return", "exposure")},
                          step=len(broker.history))
        broker.save(self._path(s.name))

    # ------------------------------------------------------------------ many days
    def run(self, prices: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        """Process every trading day in [start, end] not yet processed by the ledgers."""
        dates = prices.index.get_level_values("date").unique().sort_values()
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        done = [b.last_date for b in self.brokers.values() if b.last_date]
        if done:
            dates = dates[dates > pd.Timestamp(min(done))]
        reports = []
        for d in dates:
            rep = self.step(d, prices)
            reports.append(rep)
            if self.verbose and len(rep):
                lead = rep.sort_values("total_return", ascending=False).iloc[0]
                acts = rep["action_for_tomorrow"].value_counts().to_dict()
                print(f"[paper] {d.date()}  best so far: {lead['strategy']:28s} {lead['total_return']:+.2%}  "
                      f"orders: {acts}")
        return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()

    # ------------------------------------------------------------------ comparison
    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for s in self.strategies:
            h = self.brokers[s.name].history_frame()
            if h.empty:
                continue
            m = performance_metrics(h["daily_return"])
            rows.append({"strategy": s.name, "line": s.line, "target": s.target, "model": s.model_name,
                         "days": len(h), "equity": h["equity"].iloc[-1], "total_return": h["total_return"].iloc[-1],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                         "n_trades": len(self.brokers[s.name].trades),
                         "costs": sum(t["cost"] for t in self.brokers[s.name].trades)})
        lb = pd.DataFrame(rows)
        return lb.sort_values(["line", "target", "total_return"], ascending=[True, True, False]) if len(lb) else lb

    def equity_curves(self) -> pd.DataFrame:
        return pd.DataFrame({s.name: self.brokers[s.name].history_frame()["equity"]
                             for s in self.strategies if self.brokers[s.name].history})
